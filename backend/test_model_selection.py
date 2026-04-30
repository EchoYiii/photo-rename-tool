"""Test script for element recognition model selection.

This script verifies the supported element extraction model options can be selected
and that the recognition service honors the requested model name.

Run from the project root with:
    python -m backend.test_model_selection
"""

import os

from backend.app.core.config import settings
from backend.app.services.image_recognition import get_recognition_service


def test_supported_element_models():
    os.environ["FAKE_RECOGNITION"] = "true"

    print("=" * 70)
    print("Model Selection Test Suite")
    print("=" * 70)
    print()

    supported_models = list(settings.SUPPORTED_ELEMENT_RECOGNITION_MODELS.keys())
    if not supported_models:
        raise RuntimeError("No supported element recognition models were configured.")

    for model_name in supported_models:
        print(f"Testing model: {model_name}")
        service = get_recognition_service(
            model_name,
            settings.VALIDATION_MODEL_NAME,
            settings.DEVICE_PREFERENCE,
            use_fast=True,
        )
        result = service.recognize_image("dummy-image-path.jpg", confidence_threshold=0.1, max_labels=2)

        assert result["success"] is True, f"Recognition failed for {model_name}"
        assert result["model_used"] == model_name, f"Returned model mismatch for {model_name}"
        assert result["validation_model"] == settings.VALIDATION_MODEL_NAME
        assert len(result["labels"]) >= 1
        print(f"  ✓ OK: returned {len(result['labels'])} labels")

    print()
    print("All supported element recognition models can be selected and evaluated successfully.")


if __name__ == "__main__":
    test_supported_element_models()
