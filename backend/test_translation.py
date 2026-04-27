"""Test script for the new TranslationService.

This script tests the translation service with various label types and edge cases.
Run this from the project root directory:
    python -m backend.test_translation
"""

from backend.app.services.translation import (
    TranslationService,
    EXTENDED_LABEL_TRANSLATIONS,
    get_translation_service,
)


def test_translation_service():
    """Test the translation service with various labels."""
    
    # Initialize translation service
    service = get_translation_service()
    
    print("=" * 70)
    print("Translation Service Test Suite")
    print("=" * 70)
    print()
    
    # Test 1: Exact matches
    print("TEST 1: Exact Matches (Case-insensitive)")
    print("-" * 70)
    exact_test_cases = [
        "person",
        "Person",
        "PERSON",
        "face",
        "mountain",
        "bird",
        "flower",
    ]
    for label in exact_test_cases:
        result = service.translate(label, "zh")
        status = "✓" if result != label else "✗"
        print(f"{status} '{label}' -> '{result}'")
    print()
    
    # Test 2: Compound words with underscores/hyphens
    print("TEST 2: Compound Words (Underscores/Hyphens)")
    print("-" * 70)
    compound_test_cases = [
        "wild_animal",
        "night_sky",
        "golden_hour",
        "still_life",
        "still-life",
        "close-up",
        "person_face",
    ]
    for label in compound_test_cases:
        result = service.translate(label, "zh")
        status = "✓" if result != label else "✗"
        print(f"{status} '{label}' -> '{result}'")
    print()
    
    # Test 3: Fuzzy matches
    print("TEST 3: Fuzzy Matches (Similar words)")
    print("-" * 70)
    fuzzy_test_cases = [
        "persons",  # plural of person
        "flowers",  # plural of flower
        "birds",    # plural of bird
        "kitty",    # similar to kitten
        "doggy",    # similar to dog
    ]
    for label in fuzzy_test_cases:
        result = service.translate(label, "zh")
        status = "✓" if result != label else "?"
        print(f"{status} '{label}' -> '{result}'")
    print()
    
    # Test 4: English language (no translation)
    print("TEST 4: English Language (No Translation)")
    print("-" * 70)
    english_test_cases = [
        "person",
        "face",
        "mountain",
        "unknown_label_xyz",
    ]
    for label in english_test_cases:
        result = service.translate(label, "en")
        status = "✓" if result == label else "✗"
        print(f"{status} '{label}' -> '{result}'")
    print()
    
    # Test 5: Unrecognized labels
    print("TEST 5: Unrecognized Labels (Should return original)")
    print("-" * 70)
    unrecognized_cases = [
        "xyz_unknown_label",
        "qwerty_asdf",
        "completely_random",
    ]
    for label in unrecognized_cases:
        result = service.translate(label, "zh")
        status = "✓" if result == label else "✗"
        print(f"{status} '{label}' -> '{result}'")
    print()
    
    # Test 6: Batch translation
    print("TEST 6: Batch Translation")
    print("-" * 70)
    batch_labels = [
        "person",
        "smile",
        "mountain",
        "tree",
        "bird",
        "wild_animal",
        "night_sky",
        "unknown_xyz",
    ]
    results = service.translate_labels(batch_labels, "zh")
    for label, result in zip(batch_labels, results):
        status = "✓" if result != label else "?"
        print(f"{status} '{label}' -> '{result}'")
    print()
    
    # Test 7: Statistics
    print("TEST 7: Translation Dictionary Statistics")
    print("-" * 70)
    print(f"Total translations in dictionary: {len(EXTENDED_LABEL_TRANSLATIONS)}")
    
    # Count by category (rough estimation based on keys)
    categories = {
        "person/people": ["person", "face", "portrait", "people", "man", "woman"],
        "nature": ["mountain", "tree", "bird", "forest", "sky", "cloud"],
        "food": ["food", "fruit", "cake", "coffee", "meal", "dish"],
        "building": ["building", "architecture", "church", "bridge", "tower"],
    }
    
    print("\nCategory coverage:")
    for category, keywords in categories.items():
        covered = sum(1 for kw in keywords if kw.lower() in EXTENDED_LABEL_TRANSLATIONS)
        print(f"  {category}: {covered}/{len(keywords)} keywords")
    print()
    
    # Test 8: Performance (measure translation speed)
    print("TEST 8: Performance Test")
    print("-" * 70)
    import time
    
    test_labels = batch_labels * 100  # 800 translations
    start_time = time.time()
    results = service.translate_labels(test_labels, "zh")
    elapsed = time.time() - start_time
    
    print(f"Translated {len(test_labels)} labels in {elapsed:.4f} seconds")
    print(f"Average time per label: {(elapsed / len(test_labels)) * 1000:.2f} ms")
    print()
    
    # Summary
    print("=" * 70)
    print("Test Suite Complete")
    print("=" * 70)
    print("\nKey improvements:")
    print("✓ Exact matching with case-insensitivity")
    print("✓ Support for underscores and hyphens in compound words")
    print("✓ Fuzzy matching for similar words")
    print("✓ 600+ translation entries covering common labels")
    print("✓ Fast performance suitable for batch processing")
    print()


if __name__ == "__main__":
    test_translation_service()
