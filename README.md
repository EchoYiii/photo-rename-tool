# Photo Rename Tool

This project is a local web-based photo renaming tool with multi-model support:

- Scan a source directory for supported image formats in batch
- Use multiple AI models (Florence-2, BLIP, ViT-GPT2, CLIP, etc.) to extract candidate visual elements
- Use `google/siglip2-base-patch16-224` to validate candidates with confidence scoring
- Keep only labels with confidence >= 60%, up to 3 labels
- Rename outputs using the `aa_bb_cc.ext` rule
- Present the workflow in a browser UI with model selection

## Features

- **Multi-Model Support**: Choose from 3 different AI models for element recognition
- **Model Descriptions**: Each model includes performance characteristics and resource requirements
- **Flexible Configuration**: Adjust confidence thresholds, label counts, and output options
- **Batch Processing**: Process entire directories with progress tracking
- **Language Support**: Output labels in English or Chinese
- **Camera Detection**: Include camera manufacturer in filenames
- **Photo Type Classification**: Automatically classify photo types

## Supported Models

1. **Florence-2 base-ft** (Default, Accurate) - High accuracy, multi-task capable
2. **BLIP Image Captioning Large** (Fast, General) - Quick candidate generation
3. **ViT-GPT2 Image Captioning** (Lightweight, Stable) - Low resource requirements

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
3. Select your preferred element recognition model
4. Adjust the confidence threshold and label count if needed
5. Choose output options (include camera, photo type, elements)
6. Select label language (English/Chinese)
7. Click the start button

The first run will download public model weights, so it will take longer than later runs.
