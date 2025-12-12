# amplifier-app-vision

Image analysis and vision capabilities powered by OpenAI GPT-4 Vision.

## Overview

A complete image analysis application with both CLI and web interfaces. Analyze images, extract text, describe content, and answer questions about visual content.

## Features

- **Multi-source support**: Local files, URLs, or uploads
- **Multiple analysis modes**: Describe, extract text (OCR), custom prompts
- **Multi-image analysis**: Compare and analyze multiple images together
- **Beautiful CLI**: Rich terminal interface with progress and results
- **Web interface**: Streamlit-based browser UI
- **Token optimization**: Automatic image resizing and compression
- **Quality presets**: Quick, normal, detailed, or full quality

## Quick Start

### Run without installing (uvx)

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Launch web interface
uvx --from git+https://github.com/microsoft/amplifier-app-vision@main vision --web

# Or analyze from command line
uvx --from git+https://github.com/microsoft/amplifier-app-vision@main vision image.png
```

### Install and run

```bash
# Install
uv tool install git+https://github.com/microsoft/amplifier-app-vision@main

# Run
vision --web        # Web interface
vision image.png    # CLI analysis
```

### Local development

```bash
# Clone and install
git clone https://github.com/microsoft/amplifier-app-vision
cd amplifier-app-vision
uv sync --dev

# Run locally
uv run vision --web
```

## Usage Examples

### Web Interface (Easiest)

```bash
vision --web
```

Then:
1. Configure API key in sidebar (if not set via env)
2. Upload image or paste URL
3. Choose analysis type
4. Click "Analyze Image"

### CLI: Analyze Single Image

```bash
vision photo.jpg
```

### CLI: Custom Prompt

```bash
vision screenshot.png -p "What code is shown in this image?"
```

### CLI: Extract Text (OCR)

```bash
vision document.png --extract-text
```

### CLI: Detailed Description

```bash
vision artwork.jpg --describe
```

### CLI: Compare Multiple Images

```bash
vision before.png after.png -p "What changed between these images?"
```

### CLI: Save Results

```bash
vision image.png -o analysis.txt
```

## Configuration

### API Keys

Three ways to provide API keys:

**Option 1: Environment Variable**
```bash
export OPENAI_API_KEY=sk-...
```

**Option 2: .env File**
```
OPENAI_API_KEY=sk-...
```

**Option 3: Web UI Settings**
Enter key in sidebar (session-only, not saved)

### Quality Presets

| Preset | Max Size | Tokens | Use Case |
|--------|----------|--------|----------|
| quick | 512px | ~500 | Fast checks |
| normal | 1024px | ~1500 | General use |
| detailed | 1568px | ~4000 | Detailed analysis |
| full | Original | ~8000 | Maximum detail |

```bash
# Use detailed quality
vision image.png -q detailed
```

## Command Reference

```
Usage: vision [SOURCES...] [OPTIONS]

Arguments:
  SOURCES               Image paths or URLs

Options:
  --web                 Launch web interface in browser
  -p, --prompt TEXT     Analysis prompt (default: "What's in this image?")
  -d, --describe        Get detailed description
  -t, --extract-text    Extract text from image (OCR)
  -q, --quality         Image quality: quick, normal, detailed, full
  -m, --model TEXT      OpenAI model (default: gpt-4o)
  -o, --output PATH     Save analysis to file
  -v, --verbose         Enable verbose logging
  --help                Show this help
```

## Architecture

```
amplifier-app-vision/
├── src/amplifier_app_vision/
│   ├── __init__.py
│   ├── cli.py              # CLI entry point
│   ├── web.py              # Web launcher
│   ├── streamlit_app.py    # Streamlit UI
│   ├── analyzer.py         # Vision API integration
│   └── image_processor.py  # Image optimization
└── pyproject.toml
```

## Token Optimization

The app automatically optimizes images for token efficiency:

- **Auto-resize**: Images larger than preset max are resized
- **JPEG compression**: Non-transparent images converted to JPEG
- **Hash deduplication**: Same image not re-encoded
- **Metadata tracking**: Shows original vs optimized size

Example optimization:
```
Processed: screenshot.png (1,245,678 -> 156,432 bytes, 87% smaller)
```

## Requirements

- Python 3.11+
- OpenAI API key
- Pillow (image processing)
- Streamlit (web interface)

## Cost Information

GPT-4 Vision pricing varies by:
- Image size and detail level
- Response length

The app shows token usage after each analysis.

## Troubleshooting

### "API key not found"

```bash
export OPENAI_API_KEY="sk-..."
```

### Image format not supported

Supported formats: PNG, JPG, JPEG, GIF, WebP, BMP

### Web interface not loading

```bash
# Check Streamlit is installed
pip install streamlit

# Try running directly
python -m streamlit run src/amplifier_app_vision/streamlit_app.py
```

## License

MIT License
