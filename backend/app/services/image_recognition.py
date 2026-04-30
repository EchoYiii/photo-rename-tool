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
from typing import Optional, Any

# Optimize CUDA memory usage - MUST SET BEFORE IMPORTING torch!
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Use HuggingFace mirror for faster downloads in China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    Blip2ForConditionalGeneration,
    Florence2ForConditionalGeneration,
    pipeline,
)
from transformers.modeling_utils import PreTrainedModel

logger = logging.getLogger(__name__)

# Florence-2 remote code can expect this attribute on older/newer transformer mixes.
if not hasattr(PreTrainedModel, "_supports_sdpa"):
    PreTrainedModel._supports_sdpa = True

# Standard candidate labels for common elements in photos (in English).
# These are used for zero-shot validation.
CANDIDATE_LABELS = [
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
        use_auth_token: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.validation_model_name = validation_model_name
        self.device = self._resolve_device(device_preference)
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model_source = self._resolve_model_source(model_name)
        self.validation_model_source = self._resolve_model_source(validation_model_name)
        self.processor: Optional[AutoProcessor] = None
        self.caption_model: Optional[Any] = None
        self.caption_pipe: Optional[Any] = None
        self.validation_pipe = None
        self.auth_token = use_auth_token
        # Detect backend type: Florence vs BLIP-2 vs use pipeline fallback
        self.is_florence = "florence" in (self.model_name or "").lower()
        self.is_blip2 = "blip2" in (self.model_name or "").lower()
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

        hf_kwargs = {}
        if self.auth_token:
            hf_kwargs["use_auth_token"] = self.auth_token
        
        # Set network timeout configuration
        import os
        hf_timeout = int(os.getenv("HF_TIMEOUT", "600"))  # 10 minutes default
        hf_kwargs["timeout"] = hf_timeout
        logger.info("HuggingFace timeout set to %d seconds", hf_timeout)

        # Model loading parameters (no timeout to avoid compatibility issues)
        model_kwargs = {}
        if self.auth_token:
            model_kwargs["use_auth_token"] = self.auth_token

        # Load validation pipeline (zero-shot classifier) - used by all backends
        try:
            logger.info("Loading validation pipeline: %s", self.validation_model_source)
            self.validation_pipe = pipeline(
                "zero-shot-image-classification",
                model=self.validation_model_source,
                device=0 if self.device == "cuda" else -1,
                local_files_only=self._is_local_model_source(self.validation_model_source),
                use_fast=self.use_fast,
                **hf_kwargs,
            )
            logger.info("Validation pipeline loaded successfully")
        except Exception:
            logger.warning("Failed to load validation pipeline %s; continuing without it.", self.validation_model_source, exc_info=True)

        # Florence-2 has a specialized processor + generation class with multi-task support.
        if "florence" in (self.model_source or "").lower():
            logger.info("=== Stage 1/3: Loading Florence-2 processor: %s ===", self.model_source)
            self.processor = AutoProcessor.from_pretrained(
                self.model_source,
                local_files_only=self._is_local_model_source(self.model_source),
                trust_remote_code=True,
                **hf_kwargs,
            )
            logger.info("=== Stage 2/3: Processor loaded, now loading Florence-2 model (this may take several minutes on first run) ===")
            self.caption_model = Florence2ForConditionalGeneration.from_pretrained(
                self.model_source,
                torch_dtype=self.dtype,
                attn_implementation="eager",
                local_files_only=self._is_local_model_source(self.model_source),
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                **model_kwargs,
            ).to(self.device)
            self.caption_model.eval()
            logger.info("=== Stage 3/3: Florence-2 model loaded successfully ===")
        elif self.is_blip2:
            # BLIP-2 models require specialized loading with Blip2ForConditionalGeneration
            try:
                logger.info("=== Stage 1/2: Loading BLIP-2 processor: %s ===", self.model_source)
                self.processor = AutoProcessor.from_pretrained(
                    self.model_source,
                    use_fast=self.use_fast,
                    **hf_kwargs,
                )
                logger.info("=== Stage 2/2: Processor loaded, now loading BLIP-2 model (this may take several minutes on first run) ===")
                self.caption_model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_source,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                    **model_kwargs,
                ).to(self.device)
                self.caption_model.eval()
                self.caption_pipe = None
                logger.info("=== BLIP-2 model loaded successfully ===")
            except Exception as exc:
                logger.exception("Failed to load BLIP-2 model for %s: %s", self.model_source, exc)
                raise
        else:
            # Fallback strategy: first try AutoModelForCausalLM, if fails use image-to-text pipeline
            try:
                logger.info("=== Strategy 1: Trying AutoModelForCausalLM for %s ===", self.model_source)
                self.processor = AutoProcessor.from_pretrained(
                    self.model_source,
                    use_fast=self.use_fast,
                    **hf_kwargs,
                )
                self.caption_model = AutoModelForCausalLM.from_pretrained(
                    self.model_source,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                    **model_kwargs,
                ).to(self.device)
                self.caption_model.eval()
                self.caption_pipe = None
                logger.info("=== Model loaded successfully with AutoModelForCausalLM ===")
            except Exception as exc1:
                logger.warning("AutoModelForCausalLM failed: %s, trying pipeline fallback", exc1)
                try:
                    logger.info("=== Strategy 2: Trying image-to-text pipeline ===")
                    self.caption_pipe = pipeline(
                        "image-to-text",
                        model=self.model_source,
                        device=0 if self.device == "cuda" else -1,
                        **hf_kwargs,
                    )
                    logger.info("=== Model loaded successfully with image-to-text pipeline ===")
                except Exception as exc2:
                    logger.exception("Both strategies failed for %s", self.model_source)
                    raise

    def recognize_image(
        self,
        image_path: str,
        confidence_threshold: float = 0.60,
        max_labels: int = 3,
    ) -> dict[str, Any]:
        """Recognize elements in an image.

        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence score (0-1)
            max_labels: Maximum number of labels to return

        Returns:
            Dictionary containing recognition results
        """
        try:
            with Image.open(image_path).convert("RGB") as image:
                # Generate initial captions/descriptions
                raw_caption = self._generate_caption(image)
                logger.debug("Raw caption for %s: %s", image_path, raw_caption)

                # Extract candidate labels from caption
                candidates = self._extract_candidate_labels(raw_caption)
                logger.debug("Candidate labels for %s: %s", image_path, candidates)

                if not candidates:
                    return {"labels": [], "raw_caption": raw_caption}

                # Validate candidates with zero-shot classifier
                validated = self._validate_candidates(image, candidates)

                # Filter and format results
                results = []
                for item in validated:
                    score = item.get("score", 0.0)
                    if score >= confidence_threshold:
                        results.append({
                            "label": item.get("label", ""),
                            "score": score,
                            "confidence_percentage": round(score * 100, 2),
                        })

                results = results[:max_labels]
                logger.debug("Final labels for %s: %s", image_path, results)
                return {"labels": results, "raw_caption": raw_caption}

        except Exception as exc:
            logger.exception("Error recognizing image %s: %s", image_path, exc)
            return {"labels": [], "raw_caption": str(exc), "error": True}

    def _generate_caption(self, image: Image.Image) -> str:
        """Generate a raw caption/description for the image."""
        # First check if we're using pipeline fallback
        if self.caption_pipe is not None:
            try:
                output = self.caption_pipe(image)
                if output and isinstance(output, list) and len(output) > 0:
                    return output[0].get("generated_text", "")
                return ""
            except Exception as exc:
                logger.exception("Pipeline captioning failed: %s", exc)
                return ""
        
        if self.is_florence:
            return self._run_florence_task(image, task_prompt="<MORE_DETAILED_CAPTION>")
        elif self.is_blip2:
            return self._run_blip2_caption(image)
        else:
            return self._run_generic_caption(image)

    def _run_florence_task(self, image: Image.Image, task_prompt: str) -> str:
        """Run a Florence-2 task (e.g. captioning)."""
        if not self.processor or not self.caption_model:
            return ""

        try:
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.caption_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=512,
                    num_beams=3,
                    early_stopping=True,
                    do_sample=False,
                    repetition_penalty=1.2,
                )

            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]

            # Florence prefixes the answer with the task prompt; strip it.
            cleaned_text = generated_text.replace(task_prompt, "").strip()
            return cleaned_text
        except Exception as exc:
            logger.exception("Florence task failed for prompt %s: %s", task_prompt, exc)
            return ""

    def _run_blip2_caption(self, image: Image.Image) -> str:
        """Run BLIP-2 image captioning."""
        if not self.processor or not self.caption_model:
            return ""

        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device, self.dtype)

            with torch.no_grad():
                generated_ids = self.caption_model.generate(**inputs, max_length=200)

            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]

            return generated_text
        except Exception as exc:
            logger.exception("BLIP-2 captioning failed: %s", exc)
            return ""

    def _run_generic_caption(self, image: Image.Image) -> str:
        """Run generic image captioning (ViT-GPT2, etc.)."""
        if not self.processor or not self.caption_model:
            return ""

        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.caption_model.generate(**inputs, max_length=100)

            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]

            return generated_text
        except Exception as exc:
            logger.exception("Generic captioning failed: %s", exc)
            return ""

    def _extract_candidate_labels(self, raw_caption: str) -> list[str]:
        """Extract candidate labels from raw caption using intelligent filtering."""
        if not raw_caption:
            return []

        # Common stop words to filter out
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "each", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "just",
            "and", "but", "if", "or", "because", "until", "while", "although",
            "this", "that", "these", "those", "it", "its", "they", "them",
            "their", "what", "which", "who", "whom", "we", "our", "you", "your",
            "he", "she", "him", "her", "his", "hers", "himself", "herself",
            "my", "me", "i", "us", " myself", "ours", "themselves",
            "photo", "image", "picture", "photograph", "scene", "view",
            "showing", "shown", "depicting", "depicts", "displaying", "displays",
            "appears", "appear", "including", "contains", "contain", "featuring",
        }

        # Common adjectives that are not useful as labels
        noise_words = {
            "beautiful", "lovely", "pretty", "nice", "good", "great", "wonderful",
            "amazing", "awesome", "fantastic", "excellent", "perfect", "wonderful",
            "bright", "colorful", "vibrant", "dark", "light", "small", "large",
            "big", "little", "tiny", "huge", "giant", "massive", "tall", "short",
            "long", "round", "square", "old", "new", "young", "ancient", "modern",
            "traditional", "classic", "famous", "popular", "familiar", "famous",
            "real", "really", "very", "extremely", "incredibly", "actually",
            "simply", "basically", "literally", "totally", "completely", "entirely",
        }

        # Clean and split into words
        cleaned = re.sub(r"[^\w\s]", " ", raw_caption.lower())
        words = [w.strip() for w in cleaned.split() if w.strip() and len(w) > 2]

        # Filter out stop words and noise
        candidates = [
            w for w in words
            if w not in stop_words
            and w not in noise_words
            and not w.isdigit()
            and not any(c.isdigit() for c in w)
        ]

        # Extract meaningful 2-word phrases (only common object combinations)
        common_object_pairs = {
            "red flower", "blue sky", "green grass", "white cloud", "tall tree",
            "stone wall", "brick wall", "wooden fence", "blue water", "clear sky",
            "green leaf", "brown dog", "black cat", "white shirt", "blue jeans",
            "wearing glasses", "holding phone", "sitting chair", "standing next",
            "street light", "traffic light", "fire hydrant", "parking meter",
            "coffee table", "living room", "bedroom floor", "kitchen counter",
            "front door", "back yard", "front yard", "stone path", "dirt road",
        }

        phrases = []
        for i in range(len(words) - 1):
            two_word = f"{words[i]} {words[i+1]}"
            if two_word in common_object_pairs:
                phrases.append(two_word)

        # Combine and deduplicate
        all_candidates = candidates + phrases

        # Remove duplicates and filter again
        all_candidates = list(set(all_candidates))
        all_candidates = [c for c in all_candidates if len(c) > 2 and c not in stop_words]

        # Sort by length (longer phrases first) and limit
        all_candidates.sort(key=len, reverse=True)
        return all_candidates[:15]

    def _validate_candidates(self, image: Image.Image, candidates: list[str]) -> list[dict]:
        """Validate candidate labels with zero-shot classifier."""
        if not self.validation_pipe:
            return [{"label": c, "score": 0.75} for c in candidates]

        try:
            results = self.validation_pipe(image, candidate_labels=candidates)
            return sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
        except Exception:
            logger.warning("Validation failed, returning candidates without validation")
            return [{"label": c, "score": 0.75} for c in candidates]

    def _resolve_device(self, device_preference: str) -> str:
        """Resolve which device to use (cuda or cpu)."""
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

    def _resolve_model_source(self, model_name: str) -> str:
        """Resolve model source - check for local path or use HuggingFace hub."""
        model_path = Path(model_name)
        if model_path.exists() and model_path.is_dir():
            return model_name
        return model_name

    def _is_local_model_source(self, model_source: str) -> bool:
        """Check if model source is a local path."""
        return Path(model_source).exists() and Path(model_source).is_dir()


# Cache for loaded recognition services
_recognition_service_cache: dict[str, ImageRecognitionService] = {}


def get_recognition_service(
    model_name: str,
    validation_model_name: str,
    device_preference: str = "auto",
    use_fast: bool = True,
    force_reload: bool = False,
) -> ImageRecognitionService:
    """Get or create a recognition service instance.

    Args:
        model_name: Name of the captioning model to use
        validation_model_name: Name of the zero-shot validation model
        device_preference: Preferred device ('cuda', 'cpu', 'auto')
        use_fast: Whether to use fast tokenizers
        force_reload: Force reload even if cached

    Returns:
        ImageRecognitionService instance
    """
    cache_key = f"{model_name}::{validation_model_name}::{device_preference}::{use_fast}"
    if not force_reload and cache_key in _recognition_service_cache:
        return _recognition_service_cache[cache_key]

    service = ImageRecognitionService(
        model_name=model_name,
        validation_model_name=validation_model_name,
        device_preference=device_preference,
        use_fast=use_fast,
    )
    _recognition_service_cache[cache_key] = service
    return service