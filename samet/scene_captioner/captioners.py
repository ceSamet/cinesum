from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image


def _clean_assistant_text(text: str) -> str:
    text = text.strip()
    marker = "ASSISTANT:"
    if marker in text:
        text = text.split(marker, maxsplit=1)[-1].strip()
    return text


class Captioner(ABC):
    @abstractmethod
    def caption(self, image_path: Path, prompt: str | None = None) -> str:
        """Return a natural-language caption for one image."""


class MockCaptioner(Captioner):
    def caption(self, image_path: Path, prompt: str | None = None) -> str:
        with Image.open(image_path) as image:
            width, height = image.size
        return (
            "Mock caption: a sampled video frame is ready for visual-language "
            f"analysis ({width}x{height})."
        )


class BlipCaptioner(Captioner):
    """Practical lightweight BLIP image captioning backend."""

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def caption(self, image_path: Path, prompt: str | None = None) -> str:
        image = Image.open(image_path).convert("RGB")
        if prompt:
            inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=50)

        return self.processor.decode(out[0], skip_special_tokens=True).strip()


class Blip2Captioner(Captioner):
    """BLIP-2 backend for stronger captions when local hardware can run it."""

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    def caption(self, image_path: Path, prompt: str | None = None) -> str:
        image = Image.open(image_path).convert("RGB")
        prompt = prompt or "Question: Describe this video frame in one sentence. Answer:"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=70)

        return self.processor.decode(out[0], skip_special_tokens=True).strip()


class LlavaCaptioner(Captioner):
    """LLaVA image understanding backend for frame-level scene captions."""

    def __init__(
        self,
        model_name: str = "llava-hf/llava-interleave-qwen-0.5b-hf",
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

    def caption(self, image_path: Path, prompt: str | None = None) -> str:
        image = Image.open(image_path).convert("RGB")
        user_prompt = prompt or (
            "Describe this video frame in one concise sentence. Mention visible "
            "people, objects, place, and action if present."
        )
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": user_prompt}
        ]}]
        try:
            prompt_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except (AttributeError, ValueError):
            prompt_text = f"USER: <image>\n{user_prompt} ASSISTANT:"
        inputs = self.processor(text=prompt_text, images=image, return_tensors="pt").to(
            self.device
        )

        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                repetition_penalty=1.16,
                no_repeat_ngram_size=3,
            )

        generated = out[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return _clean_assistant_text(text)


class VideoLlavaFrameCaptioner(Captioner):
    """Video-LLaVA backend used in image mode for sampled video frames."""

    def __init__(
        self,
        model_name: str = "LanguageBind/Video-LLaVA-7B-hf",
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = VideoLlavaProcessor.from_pretrained(model_name)
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

    def caption(self, image_path: Path, prompt: str | None = None) -> str:
        image = Image.open(image_path).convert("RGB")
        user_prompt = prompt or (
            "Describe this video frame in one concise sentence. Mention visible "
            "people, objects, place, and action if present."
        )
        prompt_text = f"USER: <image>\n{user_prompt} ASSISTANT:"
        inputs = self.processor(text=prompt_text, images=image, return_tensors="pt").to(
            self.device
        )

        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=80)

        text = self.processor.batch_decode(
            out,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return _clean_assistant_text(text)


class GeminiCaptioner(Captioner):
    """Gemini API backend for low-friction remote vision captioning."""

    def __init__(
        self,
        model_name: str = "gemini-3.6-flash",
        api_key: str | None = None,
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "Gemini backend requires google-genai. Install with: "
                "pip install -r requirements-gemini.txt"
            ) from exc

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Create a .env file from .env.example "
                "or set it in your shell before running the Gemini backend."
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def caption(self, image_path: Path, prompt: str | None = None) -> str:
        image = Image.open(image_path).convert("RGB")
        prompt = prompt or (
            "Describe this video frame in one concise sentence. Mention visible "
            "people, objects, place, and action if present."
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, image],
        )

        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError(f"Gemini returned no text for frame: {image_path}")

        return text.strip()


def create_captioner(
    backend: str,
    model_name: str | None = None,
    device: str | None = None,
) -> Captioner:
    backend = backend.lower()

    if backend == "mock":
        return MockCaptioner()
    if backend == "blip":
        return BlipCaptioner(
            model_name=model_name or "Salesforce/blip-image-captioning-base",
            device=device,
        )
    if backend == "blip2":
        return Blip2Captioner(
            model_name=model_name or "Salesforce/blip2-opt-2.7b",
            device=device,
        )
    if backend == "llava":
        return LlavaCaptioner(
            model_name=model_name or "llava-hf/llava-interleave-qwen-0.5b-hf",
            device=device,
        )
    if backend == "video-llava":
        return VideoLlavaFrameCaptioner(
            model_name=model_name or "LanguageBind/Video-LLaVA-7B-hf",
            device=device,
        )
    if backend == "gemini":
        return GeminiCaptioner(
            model_name=model_name or os.getenv("GEMINI_MODEL") or "gemini-3.6-flash",
        )

    raise ValueError(f"Unsupported backend: {backend}")
