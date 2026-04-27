"""Image recognition service built on public multimodal models.

IMPORTANT: This module ALWAYS performs recognition in English. The recognized labels
are always in English, regardless of the frontend language selection. Language translation
(English -> Chinese) is handled separately in the TranslationService.

This design ensures:
1. Consistent and accurate AI model performance (models are trained primarily on English)
2. Better multilingual support (can easily add more language translations)
3. Separation of concerns (recognition logic is independent of presentation language)

The flow is:
1. Image -> AI Recognition (always in English) -> English labels
2. English labels -> TranslationService -> Chinese/Other languages (if requested)
3. Final labels are returned with appropriate language
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration, pipeline
from transformers.modeling_utils import PreTrainedModel

logger = logging.getLogger(__name__)

# Florence-2 remote code can expect this attribute on older/newer transformer mixes.
if not hasattr(PreTrainedModel, "_supports_sdpa"):
    PreTrainedModel._supports_sdpa = False

STOPWORDS = {
    "a", "an", "and", "at", "background", "blue", "by", "close", "day", "for",
    "foreground", "from", "green", "group", "in", "large", "near", "of", "on",
    "or", "photo", "scene", "small", "the", "to", "view", "white", "with"
}

FALLBACK_CANDIDATE_LABELS = [
    "person", "portrait", "face", "group", "child", "baby",
    "dog", "cat", "bird", "horse", "cow", "sheep", "fish",
    "car", "truck", "bus", "train", "bicycle", "motorcycle", "airplane", "boat",
    "building", "house", "bridge", "street", "city", "skyline",
    "beach", "sea", "lake", "river", "mountain", "forest", "tree", "flower", "garden", "snow",
    "food", "fruit", "cake", "coffee", "drink", "table",
    "book", "document", "screen", "laptop", "computer", "phone", "television", "keyboard",
    "sofa", "chair", "bed", "room", "kitchen", "bathroom",
    "artwork", "painting", "poster", "logo", "text",
]


class ImageRecognitionService:
    """Extract candidate elements and validate them with a confidence score."""

    def __init__(
        self,
        model_name: str,
        validation_model_name: str,
        device_preference: str = "auto",
        use_fast: bool = True,
    ) -> None:
        self.model_name = model_name
        self.validation_model_name = validation_model_name
        self.device = self._resolve_device(device_preference)
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model_source = self._resolve_model_source(self.model_name)
        self.validation_model_source = self._resolve_model_source(self.validation_model_name)
        self.processor: Optional[AutoProcessor] = None
        self.caption_model: Optional[Florence2ForConditionalGeneration] = None
        self.validation_pipe = None
        self.device_preference = device_preference
        self.use_fast = use_fast
        self._load_models()

    def _load_models(self) -> None:
        logger.info(
            "Loading recognition models on %s: %s + %s (use_fast=%s)",
            self.device,
            self.model_source,
            self.validation_model_source,
            self.use_fast,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_source,
            local_files_only=self._is_local_model_source(self.model_source),
        )
        self.caption_model = Florence2ForConditionalGeneration.from_pretrained(
            self.model_source,
            torch_dtype=self.dtype,
            attn_implementation="eager",
            local_files_only=self._is_local_model_source(self.model_source),
        ).to(self.device)
        self.caption_model.eval()
        self.validation_pipe = pipeline(
            "zero-shot-image-classification",
            model=self.validation_model_source,
            device=0 if self.device == "cuda" else -1,
            local_files_only=self._is_local_model_source(self.validation_model_source),
            use_fast=self.use_fast,
        )

    def recognize_image(
        self,
        image_path: str,
        confidence_threshold: float = 0.60,
        max_labels: int = 3,
    ) -> dict:
        try:
            image = Image.open(image_path).convert("RGB")
            candidates = self._generate_candidates(image, max_labels=max_labels)
            candidates = candidates or FALLBACK_CANDIDATE_LABELS[:]

            validated = self.validation_pipe(
                image,
                candidate_labels=candidates,
            )


            def _similar(a: str, b: str) -> float:
                # 简单相似度：Levenshtein距离/最大长度，返回1-归一化距离
                try:
                    import difflib
                    return difflib.SequenceMatcher(None, a, b).ratio()
                except Exception:
                    return 0.0

            labels = []
            for item in validated:
                score = float(item["score"])
                if score < confidence_threshold:
                    continue
                cleaned = self._clean_label(item["label"])
                if not cleaned:
                    continue
                # 新增：与已选标签做相似度去重
                is_similar = False
                for existing in labels:
                    if _similar(existing["label"], cleaned) > 0.5:
                        is_similar = True
                        break
                if is_similar:
                    continue
                labels.append(
                    {
                        "label": cleaned,
                        "score": round(score, 4),
                        "confidence_percentage": round(score * 100, 2),
                    }
                )
                if len(labels) >= max_labels:
                    break

            return {
                "success": True,
                "labels": labels,
                "label_count": len(labels),
                "model_used": self.model_name,
                "validation_model": self.validation_model_name,
                "candidate_count": len(candidates),
                "confidence_threshold": confidence_threshold,
                "device": self.device,
                "model_source": self.model_source,
                "validation_model_source": self.validation_model_source,
            }
        except Exception as exc:
            logger.exception("Error recognizing image %s", image_path)
            return {
                "success": False,
                "labels": [],
                "label_count": 0,
                "error": str(exc),
            }

    def _generate_candidates(self, image: Image.Image, max_labels: int) -> list[str]:
        candidates: list[str] = []

        detection_result = self._safe_run_florence_task(image, "<OD>")
        if isinstance(detection_result, dict):
            labels = detection_result.get("<OD>", {}).get("labels", [])
            candidates.extend(labels)

        dense_region_result = self._safe_run_florence_task(image, "<DENSE_REGION_CAPTION>")
        candidates.extend(self._extract_region_keywords(dense_region_result, "<DENSE_REGION_CAPTION>"))

        caption_result = self._safe_run_florence_task(image, "<MORE_DETAILED_CAPTION>")
        if isinstance(caption_result, dict):
            caption = caption_result.get("<MORE_DETAILED_CAPTION>", "")
            candidates.extend(self._extract_caption_keywords(caption))

        detailed_caption_result = self._safe_run_florence_task(image, "<DETAILED_CAPTION>")
        if isinstance(detailed_caption_result, dict):
            caption = detailed_caption_result.get("<DETAILED_CAPTION>", "")
            candidates.extend(self._extract_caption_keywords(caption))

        ocr_result = self._safe_run_florence_task(image, "<OCR>")
        if isinstance(ocr_result, dict):
            ocr_text = ocr_result.get("<OCR>", "")
            candidates.extend(self._extract_ocr_keywords(ocr_text))

        normalized: list[str] = []
        for label in candidates:
            cleaned = self._clean_label(label)
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
            if len(normalized) >= max_labels * 8:
                break

        if len(normalized) < max_labels * 2:
            for fallback in FALLBACK_CANDIDATE_LABELS:
                cleaned = self._clean_label(fallback)
                if cleaned not in normalized:
                    normalized.append(cleaned)
                if len(normalized) >= max(max_labels * 6, 18):
                    break

        return normalized[: max(max_labels * 6, 18)]

    def _run_florence_task(self, image: Image.Image, task_prompt: str) -> dict:
        if not self.processor or not self.caption_model:
            raise RuntimeError("Recognition models are not initialized")

        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        pixel_values = inputs.get("pixel_values")

        if input_ids is None:
            raise RuntimeError(f"Florence-2 did not produce input_ids for task {task_prompt}.")
        if pixel_values is None:
            raise RuntimeError(
                f"Florence-2 did not produce pixel_values for task {task_prompt}. "
                "Please confirm the input file is a readable RGB image."
            )

        model_inputs = {"input_ids": input_ids.to(self.device)}
        pixel_values = pixel_values.to(self.device)
        if self.device == "cuda":
            pixel_values = pixel_values.to(self.dtype)
        model_inputs["pixel_values"] = pixel_values
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask.to(self.device)

        with torch.no_grad():
            generated_ids = self.caption_model.generate(
                **model_inputs,
                max_new_tokens=128,
                num_beams=3,
                do_sample=False,
            )
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )
        if parsed is None:
            raise RuntimeError(f"Florence-2 returned an empty parsed result for task {task_prompt}.")
        return parsed

    def _safe_run_florence_task(self, image: Image.Image, task_prompt: str) -> dict:
        try:
            return self._run_florence_task(image, task_prompt)
        except Exception as exc:
            logger.warning("Florence task %s failed: %s", task_prompt, exc)
            return {}

    def _extract_caption_keywords(self, caption: str) -> list[str]:
        words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", caption.lower())
        keywords: list[str] = []
        for word in words:
            if word in STOPWORDS:
                continue
            cleaned = self._clean_label(word)
            if cleaned and cleaned not in keywords:
                keywords.append(cleaned)
            if len(keywords) >= 8:
                break
        return keywords

    def _extract_region_keywords(self, result: dict, key: str) -> list[str]:
        if not isinstance(result, dict):
            return []
        region_payload = result.get(key, {})
        if not isinstance(region_payload, dict):
            return []
        labels = list(region_payload.get("labels", []))
        region_captions = list(region_payload.get("bboxes_labels", []))
        candidates: list[str] = []
        candidates.extend(labels)
        for caption in region_captions:
            candidates.extend(self._extract_caption_keywords(caption))
        return candidates

    def _extract_ocr_keywords(self, ocr_text: str) -> list[str]:
        words = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-]{2,}", ocr_text.lower())
        keywords: list[str] = []
        for word in words:
            cleaned = self._clean_label(word)
            if cleaned and cleaned not in keywords:
                keywords.append(cleaned)
            if len(keywords) >= 6:
                break
        return keywords



    @staticmethod
    def _clean_label(label: str) -> str:
        if "," in label:
            label = label.split(",")[0]
        return re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")[:50]

    @staticmethod
    def _resolve_device(device_preference: str) -> str:
        preference = (device_preference or "auto").lower()
        has_cuda = torch.cuda.is_available()
        if preference == "cuda":
            if not has_cuda:
                raise RuntimeError(
                    "DEVICE_PREFERENCE is set to 'cuda', but CUDA is not available in the current PyTorch build."
                )
            return "cuda"
        if preference == "cpu":
            return "cpu"
        return "cuda" if has_cuda else "cpu"

    @staticmethod
    def _resolve_model_source(model_name: str) -> str:
        cache_root = Path.home() / ".cache" / "huggingface" / "hub"
        model_dir = cache_root / f"models--{model_name.replace('/', '--')}"
        refs_main = model_dir / "refs" / "main"
        if refs_main.exists():
            revision = refs_main.read_text(encoding="utf-8").strip()
            snapshot_dir = model_dir / "snapshots" / revision
            if snapshot_dir.exists():
                return str(snapshot_dir)
        return model_name

    @staticmethod
    def _is_local_model_source(model_source: str) -> bool:
        return os.path.exists(model_source)


_recognition_service: Optional[ImageRecognitionService] = None


def get_recognition_service(
    model_name: str,
    validation_model_name: str,
    device_preference: str = "auto",
    use_fast: bool = True,
) -> ImageRecognitionService:
    """Get or create a singleton image recognition service."""
    global _recognition_service
    # Support a fake recognition service for quick end-to-end demos without heavy model downloads.
    if os.getenv("FAKE_RECOGNITION", "false").lower() == "true":
        class FakeRecognitionService:
            def __init__(self):
                self.device = "cpu"

            def recognize_image(self, image_path: str, confidence_threshold: float = 0.6, max_labels: int = 3) -> dict:
                # deterministic fake labels useful for demos
                fake_labels = [
                    {"label": "person", "score": 0.95, "confidence_percentage": 95.0},
                    {"label": "sunset", "score": 0.87, "confidence_percentage": 87.0},
                    {"label": "mountain", "score": 0.78, "confidence_percentage": 78.0},
                ]
                return {
                    "success": True,
                    "labels": fake_labels[:max_labels],
                    "label_count": min(len(fake_labels), max_labels),
                    "model_used": model_name,
                    "validation_model": validation_model_name,
                    "candidate_count": len(fake_labels),
                    "confidence_threshold": confidence_threshold,
                    "device": "cpu",
                    "model_source": model_name,
                    "validation_model_source": validation_model_name,
                }

        return FakeRecognitionService()

    if _recognition_service is None:
        _recognition_service = ImageRecognitionService(
            model_name,
            validation_model_name,
            device_preference=device_preference,
            use_fast=use_fast,
        )
    return _recognition_service
