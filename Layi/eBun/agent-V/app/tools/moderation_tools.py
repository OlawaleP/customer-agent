"""
Image moderation for platform uploads -- separate concern from the
customer support ticket pipeline in graph.py/nodes.py. This module is
called directly by the /uploads/moderate-image route in main.py, not
by the LangGraph pipeline.

Set these env vars for live behavior:
  OPENAI_API_KEY                a real OpenAI platform key -- NOT your Groq
                                 key. Groq only mimics OpenAI's chat-completions
                                 API shape for compatibility; it doesn't proxy
                                 OpenAI's own moderation endpoint. This is a
                                 second, separate credential.
  CHATWOOT_MODERATION_CONVERSATION_ID
                                 the id of ONE pre-existing Chatwoot conversation
                                 your moderator has already opened with your
                                 WhatsApp inbox. Every flagged image gets posted
                                 as a new message into this same conversation.

Reuses CHATWOOT_BASE_URL / CHATWOOT_API_TOKEN / CHATWOOT_ACCOUNT_ID from
ticketing_tools.py's env contract -- same Chatwoot account, different
conversation.

If either OPENAI_API_KEY or the Chatwoot vars are unset, this falls back
to mock behavior the same way ticketing_tools.py does, so local dev works
without live credentials.

Why a FIXED conversation instead of creating a new one per flagged image:
proactively creating a brand-new conversation through Chatwoot's API is
built around the API-channel contact_inbox/source_id pattern, not real
WhatsApp inboxes -- and WhatsApp Business API itself restricts who you can
message first (template-only outside an already-open session). Posting
into one conversation the moderator already opened with your business
sidesteps all of that: every flagged item is just a new message in a
conversation Chatwoot already considers open, using the exact same
"post a message with an attachment" mechanism as any other reply.
"""
from __future__ import annotations
import base64
import os
from typing import Optional

import requests
# from openai import OpenAI
# from transformers import MllamaForConditionalGeneration, AutoProcessor
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import torch
import torchvision
import io

# from transformers import pipeline

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# _moderation_client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# classifier = pipeline(
#     "image-classification",
#     model="Falconsai/nsfw_image_detection"
# )
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

processor = AutoProcessor.from_pretrained(model_id)

# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype="auto",
#     device_map="auto"
# )

# Same Chatwoot account as ticketing_tools.py, different fixed conversation.
BASE_URL = os.getenv("CHATWOOT_BASE_URL")
API_TOKEN = os.getenv("CHATWOOT_API_TOKEN")
ACCOUNT_ID = os.getenv("CHATWOOT_ACCOUNT_ID")
MODERATION_CONVERSATION_ID = os.getenv("CHATWOOT_MODERATION_CONVERSATION_ID")

_CHATWOOT_LIVE = bool(BASE_URL and API_TOKEN and ACCOUNT_ID and MODERATION_CONVERSATION_ID)


class ModerationResult:
    """Wraps the OpenAI moderation verdict for one image."""

    def __init__(self, flagged: bool, categories: dict, scores: dict):
        self.flagged = flagged
        self.categories = categories  # {"sexual": True, "violence": False, ...}
        self.scores = scores          # {"sexual": 0.94, "violence": 0.02, ...}

    def summary(self) -> str:
        """e.g. 'sexual (0.94), sexual/minors (0.31)' -- for the human reviewer,
        so they're not opening the image with zero context on why it's here."""
        hits = [c for c, is_flagged in self.categories.items() if is_flagged]
        if not hits:
            return "flagged (no individual category over threshold)"
        return ", ".join(f"{c} ({self.scores.get(c, 0):.2f})" for c in hits)


# def check_image(image_bytes: bytes, content_type: str = "image/jpeg") -> ModerationResult:
#     """Runs one image through OpenAI's omni-moderation-latest model.
#     Free endpoint, but requires OPENAI_API_KEY -- see module docstring."""
#     if _moderation_client is None:
#         raise RuntimeError(
#             "OPENAI_API_KEY is not set -- image moderation needs a real OpenAI "
#             "platform key (the endpoint itself is free to call, the key is just "
#             "for identifying the account). This is separate from your Groq key."
#         )

#     data_url = f"data:{content_type};base64,{base64.b64encode(image_bytes).decode()}"
#     response = _moderation_client.moderations.create(
#         model="omni-moderation-latest",
#         input=[{"type": "image_url", "image_url": {"url": data_url}}],
#     )
#     result = response.results[0]
#     return ModerationResult(
#         flagged=result.flagged,
#         categories=dict(result.categories),
#         scores=dict(result.category_scores),
#     )

# def check_image(image_bytes: bytes, content_type: str = "image/jpeg") -> ModerationResult:
#     image = Image.open(io.BytesIO(image_bytes))

#     predictions = classifier(image)

#     # Convert to lowercase labels for robustness
#     scores = {
#         p["label"].lower(): p["score"]
#         for p in predictions
#     }

#     nsfw_score = scores.get("nsfw", 0.0)
#     safe_score = scores.get("safe", 1.0)

#     threshold = 0.70
#     flagged = nsfw_score >= threshold

#     return ModerationResult(
#         flagged=flagged,
#         categories={"nsfw": flagged},
#         scores={
#             "nsfw": nsfw_score,
#             "safe": safe_score,
#         },
#     )
def check_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    prompt = """
You are an image moderation system.

Determine whether this image should be blocked.

Return ONLY valid JSON.

{
  "flagged": true or false,
  "categories": {
      "nudity": true,
      "violence": false,
      "gore": false,
      "drugs": false,
      "weapons": false
  },
  "reason": "..."
}
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = inputs.to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=256
    )

    print(processor.decode(output[0]))

def escalate_flagged_upload(image_bytes: bytes, filename: str, content_type: str,
                             uploader_id: str, result: ModerationResult) -> dict:
    """Posts the flagged image + why it was flagged into the fixed moderation
    conversation, as a message with an attachment, for a human to review."""
    note = f"\u26a0\ufe0f Flagged upload pending review\nUploader: {uploader_id}\nReason: {result.summary()}"

    if not _CHATWOOT_LIVE:
        print(f"[MOCK escalate_flagged_upload] uploader={uploader_id} reason={result.summary()}")
        return {"mock": True, "success": True}

    url = f"{BASE_URL}/api/v1/accounts/{ACCOUNT_ID}/conversations/{MODERATION_CONVERSATION_ID}/messages"
    # NOTE: no "Content-Type" header here on purpose -- requests sets its own
    # multipart/form-data boundary header when `files=` is used. Setting
    # Content-Type: application/json (like ticketing_tools._headers() does
    # for every other call) would break this specific request.
    headers = {"api_access_token": API_TOKEN}
    data = {"content": note, "message_type": "outgoing", "private": "false"}
    files = {"attachments[]": (filename, image_bytes, content_type)}

    r = requests.post(url, headers=headers, data=data, files=files, timeout=15)
    r.raise_for_status()
    return r.json()
