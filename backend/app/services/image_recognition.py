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
    BlipForConditionalGeneration,
    BlipProcessor,
    Florence2ForConditionalGeneration,
    GPT2Tokenizer,
    ViTImageProcessor,
    pipeline,
    VisionEncoderDecoderModel,
)
from transformers.modeling_utils import PreTrainedModel

logger = logging.getLogger(__name__)

# ======================
# 照片大类（第二类标签 CLIP分类器
# ======================
# 方案中定义的14个照片大类（英文标签
PHOTO_CATEGORIES = [
    "a landscape photo of nature, mountains or ocean",
    "a portrait photo of a person's face",
    "a wildlife photo of wild animals or birds",
    "a macro photo of insects or flowers close-up",
    "a photo of astrophotography, stars or milky way at night",
    "a street photography or humanistic documentary photo",
    "a photo of architecture or buildings",
    "a food photography photo",
    "a sports or action photography photo",
    "an underwater photography photo",
    "an aerial drone photography photo",
    "an abstract or creative photography photo",
    "a still life or product photography photo",
    "a flora photography of plants or flowers",
]

# 英文标签 -> 中文标签映射
PHOTO_CATEGORY_LABELS = {
    0:  ("Landscape", "风光"),
    1:  ("Portrait", "人像"),
    2:  ("Wildlife", "野生动物"),
    3:  ("Macro", "微距"),
    4:  ("Astrophotography", "星空摄影"),
    5:  ("Street", "人文纪实"),
    6:  ("Architecture", "建筑"),
    7:  ("Food", "美食"),
    8:  ("Sports", "运动"),
    9:  ("Underwater", "水下"),
    10: ("Aerial", "航拍"),
    11: ("Abstract", "抽象"),
    12: ("Still Life", "静物"),
    13: ("Flora", "植物"),
}

# 回退大类
DEFAULT_CATEGORY = ("General", "综合")

# Florence-2 remote code can expect this attribute on older/newer transformer mixes.
if not hasattr(PreTrainedModel, "_supports_sdpa"):
    PreTrainedModel._supports_sdpa = True

# Standard candidate labels for common elements in photos (in English).
# These are used for zero-shot validation.
CANDIDATE_LABELS = [
    "person", "portrait", "face", "group", "child", "baby",
    "car", "truck", "bus", "train", "bicycle", "motorcycle", "airplane", "boat",
    "building", "house", "bridge", "street", "city", "skyline",
    "beach", "sea", "lake", "river", "mountain", "forest", "tree", "flower", "garden", "snow",
    "food", "fruit", "cake", "coffee", "drink", "table",
    "book", "document", "screen", "laptop", "computer", "phone", "television", "keyboard",
    "sofa", "chair", "bed", "room", "kitchen", "bathroom",
    "artwork", "painting", "poster", "logo", "text",
]

# Detailed species labels for enhanced animal and plant recognition
SPECIES_LABELS = [
    # Dogs
    "golden retriever", "labrador", "bulldog", "poodle", "beagle", "German shepherd", "husky", "corgi", "shiba inu", "dalmatian",
    # Cats
    "persian cat", "siamese cat", "british shorthair", "ragdoll", "maine coon", "sphynx cat",
    # Birds
    "sparrow", "eagle", "owl", "parrot", "penguin", "flamingo", "peacock", "crow", "pigeon", "swan",
    # Fish & Marine
    "goldfish", "koi", "shark", "dolphin", "whale", "jellyfish", "turtle", "crab", "lobster",
    # Animals
    "elephant", "lion", "tiger", "giraffe", "zebra", "monkey", "bear", "panda", "koala", "fox", "rabbit", "deer", "wolf", "horse", "cow", "pig", "sheep", "goat",
    # Insects
    "butterfly", "bee", "ladybug", "dragonfly", "spider", "ant", "beetle", "moth", "grasshopper",
    # Flowers
    "rose", "sunflower", "tulip", "daisy", "lily", "orchid", "lotus", "cherry blossom", "lavender", "marigold", "hibiscus", "jasmine",
    # Trees
    "pine tree", "oak tree", "maple tree", "palm tree", "cherry tree", "bamboo", "willow", "cypress", "cedar",
    # Plants
    "grass", "cactus", "fern", "moss", "bamboo plant", "bush", "succulent", "vine", "mushroom",
    # Fruits
    "apple", "banana", "orange", "grape", "strawberry", "watermelon", "pineapple", "mango", "peach", "cherry",
    # Vegetables
    "carrot", "tomato", "potato", "broccoli", "lettuce", "corn", "cucumber", "pepper", "onion", "garlic",
]

# Species labels organized by category for better targeted validation
BIRD_LABELS = [
    "sparrow", "eagle", "owl", "parrot", "penguin", "flamingo", "peacock", "crow", "pigeon", "swan",
    "robin", "hummingbird", "woodpecker", "seagull", "dove", "chicken", "duck", "goose", "turkey",
    "falcon", "hawk", "osprey", "canary", "finch", "warbler", "jay", "magpie", "raven",
    "heron", "egret", "ibis", "stork", "crane", "pelican", "cormorant", "albatross",
    "ostrich", "emu", "cassowary", "rhea", "kiwi", "toucan", "macaw", "cockatoo",
    "lovebird", "parakeet", "budgerigar", "quail", "pheasant", "peafowl", "turkey",
    "grouse", "ptarmigan", "prairie chicken", "guinea fowl", "turaco", "cuckoo",
    "roadrunner", "hoatzin", "potoo", "nightjar", "swift", "kingfisher", "bee-eater",
    "roller", "hoopoe", "hornbill", "barbet", "trogon", "quetzal", "motmot",
    "jacamar", "puffbird", "toucan", "barbet", "woodpecker", "wryneck", "honeyguide",
    "lyrebird", "scrubbird", "bowerbird", "riflebird", "bird of paradise",
    "passerine", "songbird", "warbler", "thrush", "blackbird", "starling",
    "myna", "mockingbird", "catbird", "wren", "nuthatch", "treecreeper",
    "creeper", "dipper", "waxwing", "silky flycatcher", "shrike", "vireo",
    "oriole", "blackbird", "grackle", "cowbird", "meadowlark", "bobolink",
    "cardinal", "tanager", "grosbeak", "bunting", "siskin", "redpoll",
    "crossbill", "linnet", "bullfinch", "chaffinch", "brambling", "hawfinch",
    "greenfinch", "goldfinch", "siskin", "serin", "canary", "wild canary",
    "house sparrow", "tree sparrow", "song sparrow", "white-throated sparrow"
]
DOG_LABELS = ["golden retriever", "labrador", "bulldog", "poodle", "beagle", "German shepherd", "husky", "corgi", "shiba inu", "dalmatian", "chihuahua", "yorkshire terrier", "dachshund", "pomeranian"]
CAT_LABELS = ["persian cat", "siamese cat", "british shorthair", "maine coon", "sphynx cat", "bengal cat", "british shorthair", "scottish fold", "ragdoll", "exotic shorthair"]
FLOWER_LABELS = ["rose", "sunflower", "tulip", "daisy", "lily", "orchid", "lotus", "cherry blossom", "lavender", "marigold", "hibiscus", "jasmine", "carnation", "violet", "dandelion", "peony"]
TREE_LABELS = ["pine tree", "oak tree", "maple tree", "palm tree", "cherry tree", "bamboo", "willow", "cypress", "cedar", "apple tree", "birch tree"]

GENERIC_TO_SPECIFIC = {
    "dog": set(DOG_LABELS),
    "dogs": set(DOG_LABELS),
    "puppy": set(DOG_LABELS),
    "puppies": set(DOG_LABELS),
    "cat": set(CAT_LABELS),
    "cats": set(CAT_LABELS),
    "kitten": set(CAT_LABELS),
    "kittens": set(CAT_LABELS),
    "bird": set(BIRD_LABELS),
    "birds": set(BIRD_LABELS),
    "flower": set(FLOWER_LABELS),
    "flowers": set(FLOWER_LABELS),
    "blossom": set(FLOWER_LABELS),
    "bloom": set(FLOWER_LABELS),
    "tree": set(TREE_LABELS),
    "trees": set(TREE_LABELS),
    "plant": set(SPECIES_LABELS),
    "plants": set(SPECIES_LABELS),
    "animal": set(SPECIES_LABELS),
    "animals": set(SPECIES_LABELS),
    "wildlife": set(SPECIES_LABELS),
    "pet": set(DOG_LABELS) | set(CAT_LABELS),
    "pets": set(DOG_LABELS) | set(CAT_LABELS),
    "insect": set(SPECIES_LABELS),
    "butterfly": set(SPECIES_LABELS),
}


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
        self.image_processor: Optional[ViTImageProcessor] = None
        self.text_processor: Optional[GPT2Tokenizer] = None
        self.caption_model: Optional[Any] = None
        self.caption_pipe: Optional[Any] = None
        self.validation_pipe = None
        self.auth_token = use_auth_token
        # CLIP model for photo category classification
        self.clip_model: Optional[Any] = None
        self.clip_processor: Optional[Any] = None
        # Detect backend type: Florence vs BLIP-2 vs BLIP vs YOLOv8 vs use pipeline fallback
        self.is_florence = "florence" in (self.model_name or "").lower()
        self.is_blip2 = "blip2" in (self.model_name or "").lower()
        self.is_blip = "blip" in (self.model_name or "").lower() and not self.is_blip2
        self.is_yolov8 = "yolov8" in (self.model_name or "").lower()
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
        elif self.is_blip:
            # BLIP models require specialized loading with BlipForConditionalGeneration
            try:
                logger.info("=== Stage 1/2: Loading BLIP processor: %s ===", self.model_source)
                self.processor = BlipProcessor.from_pretrained(
                    self.model_source,
                    **hf_kwargs,
                )
                logger.info("=== Stage 2/2: Processor loaded, now loading BLIP model (this may take several minutes on first run) ===")
                self.caption_model = BlipForConditionalGeneration.from_pretrained(
                    self.model_source,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                    **model_kwargs,
                ).to(self.device)
                self.caption_model.eval()
                self.caption_pipe = None
                logger.info("=== BLIP model loaded successfully ===")
            except Exception as exc:
                logger.exception("Failed to load BLIP model for %s: %s", self.model_source, exc)
                raise
        elif self.is_yolov8:
            # YOLOv8 uses ultralytics for object detection
            try:
                from ultralytics import YOLO
                logger.info("=== Loading YOLOv8 model (this may take a few minutes on first run) ===")
                # Use yolov8n.pt (nano) for speed, can upgrade to yolov8s.pt, yolov8m.pt etc.
                yolo_model_name = os.getenv("YOLO_MODEL", "yolov8n.pt")
                self.yolo_model = YOLO(yolo_model_name)
                # Set device
                if self.device == "cuda":
                    self.yolo_model.to("cuda")
                logger.info("=== YOLOv8 model loaded successfully on %s ===", self.device)
            except Exception as exc:
                logger.exception("Failed to load YOLOv8 model: %s", exc)
                raise
        else:
            # Fallback strategy: first try VisionEncoderDecoderModel (for ViT-GPT2, etc.),
            # if fails try AutoModelForCausalLM, then image-to-text pipeline
            try:
                logger.info("=== Strategy 1: Trying VisionEncoderDecoderModel for %s ===", self.model_source)
                self.image_processor = ViTImageProcessor.from_pretrained(self.model_source)
                self.text_processor = GPT2Tokenizer.from_pretrained(self.model_source)
                self.caption_model = VisionEncoderDecoderModel.from_pretrained(
                    self.model_source,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                    **model_kwargs,
                ).to(self.device)
                self.caption_model.eval()
                self.caption_pipe = None
                logger.info("=== Model loaded successfully with VisionEncoderDecoderModel ===")
            except Exception as exc0:
                logger.warning("VisionEncoderDecoderModel failed: %s, trying AutoModelForCausalLM", exc0)
                try:
                    logger.info("=== Strategy 2: Trying AutoModelForCausalLM for %s ===", self.model_source)
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
                        logger.info("=== Strategy 3: Trying image-to-text pipeline ===")
                        self.caption_pipe = pipeline(
                            "image-to-text",
                            model=self.model_source,
                            device=0 if self.device == "cuda" else -1,
                            **hf_kwargs,
                        )
                        logger.info("=== Model loaded successfully with image-to-text pipeline ===")
                    except Exception as exc2:
                        logger.exception("All strategies failed for %s", self.model_source)
                        raise

        # Load CLIP model for photo category classification (第二类标签
        try:
            logger.info("=== Loading CLIP model for photo category classification...")
            from transformers import CLIPProcessor, CLIPModel
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-large-patch14",
                **hf_kwargs
            )
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=self.dtype,
                **model_kwargs
            ).to(self.device)
            self.clip_model.eval()
            logger.info("=== CLIP model loaded successfully ===")
        except Exception as exc:
            logger.warning("Failed to load CLIP model: %s, category classification disabled", exc)

    def classify_photo_category(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.5,
    ) -> tuple[tuple[str, str], float]:
        """
        照片大类分类（第二类标签

        Returns:
            ((英文标签, 中文标签), 置信度
            如果置信度低于阈值，返回DEFAULT_CATEGORY和置信度
        """
        if not self.clip_model or not self.clip_processor:
            logger.warning("CLIP model not available, returning default category")
            return DEFAULT_CATEGORY, 0.0
        try:
            inputs = self.clip_processor(
                text=PHOTO_CATEGORIES,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            with torch.no_grad():
                logits = self.clip_model(**inputs).logits_per_image
                probs = logits.softmax(dim=-1)
                best_idx = probs.argmax().item()
                confidence = probs[0][best_idx].item()
            if confidence < confidence_threshold:
                return DEFAULT_CATEGORY, confidence
            label_en, label_zh = PHOTO_CATEGORY_LABELS.get(best_idx, DEFAULT_CATEGORY)
            return (label_en, label_zh), confidence
        except Exception as exc:
            logger.exception("Photo category classification failed: %s", exc)
            return DEFAULT_CATEGORY, 0.0

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
            Dictionary containing recognition results including photo category (第二类标签 and elements (第三类标签
        """
        try:
            with Image.open(image_path).convert("RGB") as image:
                # Step 1: 进行照片大类分类（第二类标签
                (category_en, category_zh), category_conf = self.classify_photo_category(
                    image,
                    confidence_threshold=0.5
                )

                # Special handling for YOLOv8 - direct object detection with species labels
                if self.is_yolov8 and hasattr(self, "yolo_model") and self.yolo_model is not None:
                    yolov8_result = self._recognize_with_yolov8(image, image_path, confidence_threshold, max_labels)
                    yolov8_result["category_en"] = category_en
                    yolov8_result["category_zh"] = category_zh
                    yolov8_result["category_confidence"] = category_conf
                    return yolov8_result

                # Generate initial captions/descriptions
                raw_caption = self._generate_caption(image)
                logger.debug("Raw caption for %s: %s", image_path, raw_caption)

                # Extract candidate labels from caption
                candidates = self._extract_candidate_labels(raw_caption)
                logger.debug("Candidate labels for %s: %s", image_path, candidates)

                if not candidates:
                    return {"labels": [], "raw_caption": raw_caption}

                # Detect and add species-specific labels for animals and plants
                species_candidates = self._detect_species(raw_caption, candidates)
                if species_candidates:
                    candidates = list(set(candidates + species_candidates))
                    logger.debug("After species detection: %s", candidates)

                # If animal/plant category detected, use TARGETED species validation
                if self._should_validate_species(candidates):
                    # Get targeted labels based on detected category
                    targeted_labels = self._get_targeted_species_labels(candidates, raw_caption)
                    # IMPORTANT: When bird category is detected, extend candidates with ALL bird species
                    # from BIRD_LABELS for comprehensive zero-shot validation
                    if "bird" in candidates or "bird" in raw_caption.lower():
                        bird_labels = [b for b in BIRD_LABELS if b not in candidates]
                        if bird_labels:
                            candidates = list(set(candidates + bird_labels))
                            logger.debug("Extended candidates with %d bird species for validation", len(bird_labels))
                    # Use lower threshold for species to improve recall
                    species_threshold = 0.1
                    species_validated = self._validate_species_labels(image, targeted_labels)
                    if species_validated:
                        # If bird category detected, only keep the top 1 bird species
                        is_bird_detected = "bird" in candidates or "bird" in raw_caption.lower()
                        if is_bird_detected:
                            bird_species = [sv for sv in species_validated if sv.get("label", "").lower() in [b.lower() for b in BIRD_LABELS]]
                            if bird_species:
                                top_bird = bird_species[0]
                                candidates = [top_bird["label"]]
                                logger.debug("Bird species detected, only keeping top match: %s", top_bird["label"])
                        else:
                            for sv in species_validated:
                                if sv.get("score", 0) >= species_threshold:
                                    candidates.append(sv["label"])
                            candidates = list(set(candidates))
                        logger.debug("After targeted species zero-shot validation: %s", candidates)

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

                results = self._deduplicate_similar_labels(results)
                results = results[:max_labels]
                logger.debug("Final labels for %s: %s", image_path, results)
                return {
                    "labels": results, 
                    "raw_caption": raw_caption,
                    "category_en": category_en,
                    "category_zh": category_zh,
                    "category_confidence": category_conf
                }

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
        elif self.is_blip:
            return self._run_blip_caption(image)
        elif self.is_yolov8:
            return self._run_yolov8_detection(image)
        else:
            return self._run_generic_caption(image)

    def _run_yolov8_detection(self, image: Image.Image) -> str:
        """Run YOLOv8 object detection and return species-level labels."""
        if not hasattr(self, "yolo_model") or self.yolo_model is None:
            return ""
        try:
            results = self.yolo_model(image, verbose=False)
            if not results or len(results) == 0:
                return ""
            result = results[0]
            names = result.names
            detected_labels = []
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls.cpu().numpy()[0])
                    conf = float(box.conf.cpu().numpy()[0])
                    if conf >= 0.25:
                        label = names[cls_id]
                        detected_labels.append(f"{label} ({conf:.2f})")
            return "; ".join(detected_labels) if detected_labels else ""
        except Exception as exc:
            logger.warning("YOLOv8 detection failed: %s", exc)
            return ""

    def _recognize_with_yolov8(
        self,
        image: Image.Image,
        image_path: str,
        confidence_threshold: float = 0.60,
        max_labels: int = 3,
    ) -> dict[str, Any]:
        """Recognize objects in an image using YOLOv8 directly.

        YOLOv8 provides direct object detection with bounding boxes and confidence scores.
        This method extracts species-level labels directly from the detection results,
        and sorts them by (area × confidence) to prioritize more significant objects.
        """
        try:
            results = self.yolo_model(image, verbose=False)
            if not results or len(results) == 0:
                return {"labels": [], "raw_caption": "No objects detected"}

            result = results[0]
            names = result.names

            detected_objects = []
            if result.boxes is not None and len(result.boxes) > 0:
                seen_labels = {}
                for box in result.boxes:
                    cls_id = int(box.cls.cpu().numpy()[0])
                    conf = float(box.conf.cpu().numpy()[0])
                    if conf >= confidence_threshold:
                        label = names[cls_id]
                        # 计算边界框面积
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                        area = (x2 - x1) * (y2 - y1)
                        # 计算显著性分数 = 面积 × 置信度
                        saliency = area * conf
                        if label not in seen_labels or seen_labels[label]["saliency"] < saliency:
                            seen_labels[label] = {
                                "score": conf,
                                "saliency": saliency
                            }

                # 按显著性排序（面积×置信度）
                sorted_items = sorted(
                    seen_labels.items(),
                    key=lambda x: x[1]["saliency"],
                    reverse=True
                )
                for label, data in sorted_items:
                    detected_objects.append({
                        "label": label,
                        "score": data["score"],
                        "confidence_percentage": round(data["score"] * 100, 2),
                    })

            detected_objects = detected_objects[:max_labels]
            raw_caption = ", ".join([f"{obj['label']} ({obj['confidence_percentage']}%)" for obj in detected_objects])

            logger.debug("YOLOv8 detection for %s: %s", image_path, detected_objects)
            return {"labels": detected_objects, "raw_caption": raw_caption}

        except Exception as exc:
            logger.exception("Error in YOLOv8 recognition for %s: %s", image_path, exc)
            return {"labels": [], "raw_caption": str(exc), "error": True}

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

    def _run_blip_caption(self, image: Image.Image) -> str:
        """Run BLIP image captioning."""
        if not self.processor or not self.caption_model:
            return ""

        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.caption_model.generate(**inputs, max_length=200)

            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]

            return generated_text
        except Exception as exc:
            logger.exception("BLIP captioning failed: %s", exc)
            return ""

    def _run_generic_caption(self, image: Image.Image) -> str:
        """Run generic image captioning (ViT-GPT2, etc.)."""
        if not self.image_processor or not self.text_processor or not self.caption_model:
            return ""

        try:
            image_inputs = self.image_processor(image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.caption_model.generate(
                    pixel_values=image_inputs.pixel_values,
                    max_length=100
                )

            generated_text = self.text_processor.batch_decode(
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

    def _detect_species(self, raw_caption: str, candidates: list[str]) -> list[str]:
        """Detect specific species from caption when animal/plant categories are detected.

        Args:
            raw_caption: The raw caption text from the model
            candidates: Already extracted candidate labels

        Returns:
            List of specific species labels found in the caption
        """
        detected_species = []
        caption_lower = raw_caption.lower()

        # Define category keywords to check
        animal_categories = {"dog", "cat", "bird", "fish", "horse", "cow", "sheep", "pig", "animal", "pet", "mammal", "wildlife", "creature", "beast"}
        plant_categories = {"flower", "tree", "plant", "fruit", "vegetable", "grass", "leaf", "garden", "blossom", "bloom", "foliage", "botanical"}

        # Check candidates for category keywords
        has_animal_cat = any(cat in candidates for cat in animal_categories)
        has_plant_cat = any(cat in candidates for cat in plant_categories)

        # Also check raw caption for category keywords
        has_animal_in_caption = any(cat in caption_lower for cat in animal_categories)
        has_plant_in_caption = any(cat in caption_lower for cat in plant_categories)

        # If category keywords found, do species detection
        should_detect = has_animal_cat or has_plant_cat or has_animal_in_caption or has_plant_in_caption

        if not should_detect:
            return []

        # Check each species label in SPECIES_LABELS for matches in the caption
        for species in SPECIES_LABELS:
            species_lower = species.lower()
            # Direct substring match
            if species_lower in caption_lower:
                detected_species.append(species)
                continue
            # For multi-word species like "golden retriever", check if all words match
            if " " in species:
                words = species_lower.split()
                word_matches = sum(1 for w in words if w in caption_lower)
                if word_matches >= 2 and word_matches == len(words):
                    detected_species.append(species)

        # For bird detection, also check BIRD_LABELS for specific bird species in caption
        # This catches birds that are not in the smaller SPECIES_LABELS
        if "bird" in candidates or "bird" in caption_lower or has_animal_cat or has_animal_in_caption:
            for bird in BIRD_LABELS:
                bird_lower = bird.lower()
                if bird_lower in caption_lower:
                    detected_species.append(bird)
                    continue
                # For multi-word bird names like "house sparrow", check if all words match
                if " " in bird:
                    words = bird_lower.split()
                    word_matches = sum(1 for w in words if w in caption_lower)
                    if word_matches >= 2 and word_matches == len(words):
                        detected_species.append(bird)

        # Remove duplicates while preserving order
        seen = set()
        unique_species = []
        for s in detected_species:
            if s not in seen:
                seen.add(s)
                unique_species.append(s)

        logger.debug("Detected species: %s", unique_species)
        return unique_species

    def _should_validate_species(self, candidates: list[str]) -> bool:
        """Check if we should do species-level validation based on candidates."""
        animal_categories = {"dog", "cat", "bird", "fish", "horse", "cow", "sheep", "pig", "animal", "pet", "mammal", "wildlife", "creature", "beast"}
        plant_categories = {"flower", "tree", "plant", "fruit", "vegetable", "grass", "leaf", "garden", "blossom", "bloom", "foliage", "botanical"}
        return any(cat in candidates for cat in animal_categories | plant_categories)

    def _get_targeted_species_labels(self, candidates: list[str], caption: str) -> list[str]:
        """Get targeted species labels based on detected category."""
        caption_lower = caption.lower()
        targets = []

        # Check for bird
        if "bird" in candidates or "bird" in caption_lower:
            targets.extend(BIRD_LABELS)
        # Check for dog
        if "dog" in candidates or "dog" in caption_lower:
            targets.extend(DOG_LABELS)
        # Check for cat
        if "cat" in candidates or "cat" in caption_lower:
            targets.extend(CAT_LABELS)
        # Check for flower
        if "flower" in candidates or "flower" in caption_lower:
            targets.extend(FLOWER_LABELS)
        # Check for tree
        if "tree" in candidates or "tree" in caption_lower:
            targets.extend(TREE_LABELS)

        # If no specific category found, use all species labels
        if not targets:
            targets = SPECIES_LABELS

        # Remove duplicates
        return list(set(targets))

    def _deduplicate_similar_labels(self, results: list[dict]) -> list[dict]:
        """Remove generic labels when a specific species label from the same category exists.

        For example: if results contain both 'dog' and 'golden retriever', only keep
        'golden retriever' since it is more specific.
        """
        if not results:
            return results

        result_labels = [r["label"] for r in results]
        generic_labels_to_remove = set()

        for result in results:
            label = result["label"].lower()
            if label in GENERIC_TO_SPECIFIC:
                specific_set = GENERIC_TO_SPECIFIC[label]
                for specific_label in specific_set:
                    if specific_label.lower() in result_labels:
                        generic_labels_to_remove.add(result["label"])
                        break

        deduplicated = [r for r in results if r["label"] not in generic_labels_to_remove]
        logger.debug("After label deduplication: removed generic labels %s, kept %s", generic_labels_to_remove, [r["label"] for r in deduplicated])
        return deduplicated

    def _validate_species_labels(self, image: Image.Image, species_labels: list[str]) -> list[dict]:
        """Validate species labels using zero-shot classification on the image.

        Args:
            image: PIL Image
            species_labels: List of species labels to validate

        Returns:
            List of dicts with label and score, sorted by score descending
        """
        if not self.validation_pipe:
            return []

        try:
            # Limit to top species to avoid too many queries
            top_species = species_labels[:50]
            results = self.validation_pipe(image, candidate_labels=top_species)
            return sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
        except Exception as exc:
            logger.warning("Species validation failed: %s", exc)
            return []

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