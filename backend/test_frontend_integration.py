import requests
import json

def test_frontend_integration():
    """Test that frontend can load model options and descriptions."""
    try:
        # Test /info endpoint
        response = requests.get("http://localhost:8000/api/v1/info", timeout=10)
        if response.status_code != 200:
            print(f"❌ /info endpoint failed: {response.status_code}")
            return False

        data = response.json()
        element_models = data.get("element_models", [])

        if not element_models:
            print("❌ No element_models in /info response")
            return False

        print("✅ /info endpoint returns element_models:")
        for model in element_models:
            print(f"  - {model['name']}: {model['label']}")
            if 'desc' in model:
                print(f"    Description: {model['desc']}")

        # Test that all expected models are present
        expected_models = [
            "florence-community/Florence-2-base-ft",
            "Salesforce/blip-image-captioning-large",
            "nlpconnect/vit-gpt2-image-captioning"
        ]

        actual_names = [m['name'] for m in element_models]
        missing = [m for m in expected_models if m not in actual_names]

        if missing:
            print(f"❌ Missing models: {missing}")
            return False

        print("✅ All expected models present")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == '__main__':
    test_frontend_integration()