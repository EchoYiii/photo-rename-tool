# Photo Rename Tool

This project is a local web-based photo renaming tool:

- Scan a source directory for supported image formats in batch
- Use `florence-community/Florence-2-base-ft` to extract candidate visual elements
- Use `google/siglip2-base-patch16-224` to validate candidates with confidence scoring
- Keep only labels with confidence >= 60%, up to 3 labels
- Rename outputs using the `aa_bb_cc.ext` rule
- Present the workflow in a browser UI

## Start

```powershell
cd C:\Users\15316\Documents\New project\photo-rename-tool
python -m pip install -r backend\requirements.txt
python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
python backend\run.py
```

Open:

```text
http://127.0.0.1:8000
```

## Usage

1. Enter the source image directory in the page
2. Enter the output directory
3. Adjust the confidence threshold and label count if needed
4. Click the start button

The first run will download public model weights, so it will take longer than later runs.
