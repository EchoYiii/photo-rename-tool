import os
import sys
import tempfile
from PIL import Image

# Add backend to path
sys.path.insert(0, 'backend')

from app.services.image_recognition import get_recognition_service

def test_real_models():
    # Create a simple test image (solid color)
    img = Image.new('RGB', (224, 224), color=(255, 0, 0))
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        img.save(f.name)
        test_image_path = f.name

    models_to_test = [
        'florence-community/Florence-2-base-ft',
        'Salesforce/blip-image-captioning-large',
        'nlpconnect/vit-gpt2-image-captioning'
    ]

    print('Testing real models with a simple red square image...')
    for model in models_to_test:
        try:
            service = get_recognition_service(model, 'google/siglip2-base-patch16-224', 'cpu', True)
            result = service.recognize_image(test_image_path, confidence_threshold=0.1, max_labels=3)
            print(f'{model}: {len(result["labels"])} labels - {result["labels"][:2]}')
        except Exception as e:
            print(f'{model}: ERROR - {str(e)[:100]}')

    os.unlink(test_image_path)
    print('Test completed.')

if __name__ == '__main__':
    test_real_models()