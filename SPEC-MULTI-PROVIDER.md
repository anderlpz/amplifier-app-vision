# Specification: Multi-Provider Vision Analyzer

## Overview

Refactor `amplifier-app-vision` from OpenAI-only to a provider-agnostic architecture supporting OpenAI, Anthropic, and Google (Gemini) vision APIs while integrating with Amplifier's configuration system.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      VisionAnalyzer                              │
│  (Public API - backward compatible)                              │
│  - analyze(), describe(), extract_text(), answer_question()      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ProviderRouter                              │
│  - detect_provider(model) → "openai" | "anthropic" | "google"   │
│  - get_provider(model) → VisionProvider                          │
│  - resolve_api_key(provider) → str                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ OpenAIProvider  │ │AnthropicProvider│ │ GoogleProvider  │
│                 │ │                 │ │                 │
│ - format_msg()  │ │ - format_msg()  │ │ - format_msg()  │
│ - call_api()    │ │ - call_api()    │ │ - call_api()    │
│ - parse_resp()  │ │ - parse_resp()  │ │ - parse_resp()  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## Module Specifications

### Module: `providers/base.py`

**Purpose:** Abstract base class defining the provider contract.

**Contract:**
```python
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass

@dataclass
class VisionRequest:
    """Normalized request format."""
    images: list[dict]        # [{"data": b64, "media_type": str} | {"url": str}]
    prompt: str
    system_prompt: Optional[str]
    max_tokens: int

@dataclass  
class VisionResponse:
    """Normalized response format."""
    text: str
    model: str
    usage: dict               # {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    raw_response: Optional[dict] = None

class VisionProvider(ABC):
    """Base class for vision API providers."""
    
    provider_name: str        # "openai", "anthropic", "google"
    
    @abstractmethod
    def __init__(self, api_key: str, model: str): ...
    
    @abstractmethod
    def analyze(self, request: VisionRequest) -> VisionResponse:
        """Send vision request and return normalized response."""
        ...
    
    @property
    @abstractmethod
    def supported_models(self) -> list[str]:
        """List of supported model names."""
        ...
```

**Dependencies:** None (stdlib only)

---

### Module: `providers/openai.py`

**Purpose:** OpenAI GPT-4 Vision implementation.

**Contract:**
- Inputs: `VisionRequest` with base64 or URL images
- Outputs: `VisionResponse` with usage stats
- Side Effects: HTTP calls to OpenAI API

**Supported Models:**
- `gpt-4o` (default)
- `gpt-4o-mini`
- `gpt-4-turbo`

**API Format (OpenAI):**
```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        {"type": "text", "text": prompt}
    ]}
]
```

**Dependencies:** `openai>=1.0.0`

---

### Module: `providers/anthropic.py`

**Purpose:** Anthropic Claude Vision implementation.

**Contract:**
- Inputs: `VisionRequest` with base64 images (URLs not directly supported)
- Outputs: `VisionResponse` with usage stats
- Side Effects: HTTP calls to Anthropic API

**Supported Models:**
- `claude-sonnet-4-20250514`
- `claude-3.5-sonnet` / `claude-3-5-sonnet-latest`
- `claude-3-opus`
- `claude-3-haiku`

**API Format (Anthropic):**
```python
# Note: Anthropic uses "source" not "image_url", and requires media_type
messages = [
    {"role": "user", "content": [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "base64_data_here"
            }
        },
        {"type": "text", "text": prompt}
    ]}
]
# system_prompt is a top-level param, not in messages
response = client.messages.create(
    model=model,
    system=system_prompt,
    messages=messages,
    max_tokens=max_tokens
)
```

**Key Differences from OpenAI:**
1. System prompt is separate parameter, not in messages
2. Image format uses `source.type: "base64"` not `image_url`
3. Must include `media_type` in source
4. URLs must be fetched and converted to base64

**Dependencies:** `anthropic>=0.30.0`

---

### Module: `providers/google.py`

**Purpose:** Google Gemini Vision implementation.

**Contract:**
- Inputs: `VisionRequest` with base64 or URL images
- Outputs: `VisionResponse` with usage stats
- Side Effects: HTTP calls to Google Generative AI API

**Supported Models:**
- `gemini-1.5-pro`
- `gemini-1.5-flash`
- `gemini-2.0-flash-exp`

**API Format (Google):**
```python
import google.generativeai as genai

model = genai.GenerativeModel(model_name)
# Images as Part objects
content = [
    {"mime_type": "image/jpeg", "data": base64_bytes},  # or inline_data
    prompt_text
]
response = model.generate_content(content)
```

**Key Differences:**
1. Uses `google-generativeai` SDK (different from Vertex AI)
2. Images passed as Part objects with `mime_type` and `data`
3. System instruction via `GenerativeModel(system_instruction=...)`
4. Usage in `response.usage_metadata`

**Dependencies:** `google-generativeai>=0.5.0`

---

### Module: `providers/router.py`

**Purpose:** Provider detection, instantiation, and API key resolution.

**Contract:**
```python
class ProviderRouter:
    """Routes requests to appropriate provider based on model name."""
    
    # Model prefix patterns
    MODEL_PATTERNS = {
        "gpt-": "openai",
        "claude-": "anthropic", 
        "gemini-": "google",
    }
    
    # Default models per provider
    DEFAULT_MODELS = {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-20250514",
        "google": "gemini-1.5-flash",
    }
    
    # Environment variable names for API keys
    API_KEY_ENV_VARS = {
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    }
    
    def detect_provider(self, model: str) -> str:
        """Detect provider from model name prefix."""
        ...
    
    def resolve_api_key(self, provider: str, explicit_key: Optional[str] = None) -> str:
        """Resolve API key from: explicit param > Amplifier config > env var."""
        ...
    
    def get_provider(
        self, 
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> VisionProvider:
        """Get configured provider instance."""
        ...
```

**Key Resolution Order:**
1. Explicit `api_key` parameter
2. Amplifier config (`~/.amplifier/config.yaml` or session config)
3. Environment variables

**Dependencies:** `pyyaml>=6.0` (for config parsing)

---

### Module: `config.py`

**Purpose:** Amplifier configuration integration.

**Contract:**
```python
@dataclass
class AmplifierConfig:
    """Configuration from Amplifier."""
    model: Optional[str] = None
    api_keys: dict[str, str] = field(default_factory=dict)
    
def load_amplifier_config() -> AmplifierConfig:
    """Load config from Amplifier's config files.
    
    Searches (in order):
    1. .amplifier/config.yaml (project)
    2. ~/.amplifier/config.yaml (user)
    
    Expected format:
    ```yaml
    model: gpt-4o  # or claude-sonnet-4-20250514, gemini-1.5-pro
    api_keys:
      openai: sk-...
      anthropic: sk-ant-...
      google: AI...
    ```
    """
    ...

def get_model_from_amplifier_session() -> Optional[str]:
    """Get currently selected model from active Amplifier session.
    
    Future: Read from session state when Amplifier exposes this.
    For now: Return None (fall back to config/default).
    """
    return None
```

**Dependencies:** `pyyaml>=6.0`

---

### Module: `analyzer.py` (Refactored)

**Purpose:** Main public API - backward compatible with current interface.

**Contract (unchanged public interface):**
```python
class VisionAnalyzer:
    def __init__(
        self,
        api_key: Optional[str] = None,    # Explicit API key (any provider)
        model: str = "gpt-4o",            # Model name (auto-detects provider)
        quality: str = "normal",          # Image quality preset
        provider: Optional[str] = None,   # NEW: Force specific provider
    ): ...
    
    def analyze(
        self,
        image_source: str | Path,
        prompt: str = "What's in this image?",
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> dict: ...
    
    def analyze_multiple(
        self,
        image_sources: list[str | Path],
        prompt: str = "Describe these images.",
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> dict: ...
    
    def describe(self, image_source: str | Path) -> str: ...
    def extract_text(self, image_source: str | Path) -> str: ...
    def answer_question(self, image_source: str | Path, question: str) -> str: ...
```

**Backward Compatibility:**
- Default model remains `gpt-4o`
- Existing `api_key` param works (routes to detected provider)
- All existing method signatures preserved
- Return format unchanged

**Implementation Notes:**
```python
class VisionAnalyzer:
    def __init__(self, api_key=None, model="gpt-4o", quality="normal", provider=None):
        self.model = model
        self.processor = ImageProcessor(quality=quality)
        
        # Use router to get appropriate provider
        self._router = ProviderRouter()
        self._provider = self._router.get_provider(
            model=model,
            provider=provider,
            api_key=api_key,
        )
    
    def analyze(self, image_source, prompt, system_prompt=None, max_tokens=1024):
        # Process image (same as before)
        image_data = self._process_image(image_source)
        
        # Build normalized request
        request = VisionRequest(
            images=[image_data],
            prompt=prompt,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            max_tokens=max_tokens,
        )
        
        # Delegate to provider
        response = self._provider.analyze(request)
        
        # Return in existing format
        return {
            "text": response.text,
            "model": response.model,
            "usage": response.usage,
            "image_metadata": image_data.get("metadata", {}),
        }
```

---

## File Structure

```
src/amplifier_app_vision/
├── __init__.py              # Public exports
├── analyzer.py              # Refactored VisionAnalyzer
├── cli.py                   # CLI (unchanged)
├── config.py                # NEW: Amplifier config integration
├── image_processor.py       # Unchanged
└── providers/
    ├── __init__.py          # Provider exports
    ├── base.py              # VisionProvider ABC, VisionRequest/Response
    ├── router.py            # ProviderRouter
    ├── openai.py            # OpenAIProvider
    ├── anthropic.py         # AnthropicProvider
    └── google.py            # GoogleProvider
```

---

## Dependencies Update

```toml
# pyproject.toml additions
dependencies = [
    # Existing
    "click>=8.0",
    "rich>=13.0",
    "python-dotenv>=1.0.0",
    "openai>=1.0.0",
    "Pillow>=10.0.0",
    # New
    "anthropic>=0.30.0",
    "google-generativeai>=0.5.0",
    "pyyaml>=6.0",
]

# Optional: Make providers optional for minimal installs
[project.optional-dependencies]
openai = ["openai>=1.0.0"]
anthropic = ["anthropic>=0.30.0"]
google = ["google-generativeai>=0.5.0"]
all = ["openai>=1.0.0", "anthropic>=0.30.0", "google-generativeai>=0.5.0"]
```

---

## API Key Resolution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Explicit api_key param passed to VisionAnalyzer?             │
│    YES → Use it                                                  │
│    NO  ↓                                                         │
├─────────────────────────────────────────────────────────────────┤
│ 2. Check Amplifier config (~/.amplifier/config.yaml)            │
│    api_keys.{provider} exists?                                   │
│    YES → Use it                                                  │
│    NO  ↓                                                         │
├─────────────────────────────────────────────────────────────────┤
│ 3. Check environment variables                                   │
│    OpenAI:    OPENAI_API_KEY                                    │
│    Anthropic: ANTHROPIC_API_KEY                                 │
│    Google:    GOOGLE_API_KEY or GEMINI_API_KEY                  │
│    Found? → Use it                                               │
│    NO → Raise ConfigurationError                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Provider Detection Flow

```
Model: "claude-sonnet-4-20250514"
         │
         ▼
┌─────────────────────────────┐
│ Check prefix patterns:       │
│ - "gpt-" → openai           │
│ - "claude-" → anthropic  ✓  │
│ - "gemini-" → google        │
└─────────────────────────────┘
         │
         ▼
Provider: "anthropic"
         │
         ▼
┌─────────────────────────────┐
│ Instantiate AnthropicProvider│
│ with resolved API key        │
└─────────────────────────────┘
```

---

## Error Handling

```python
class VisionError(Exception):
    """Base exception for vision errors."""

class ProviderError(VisionError):
    """Provider-specific API error."""
    def __init__(self, provider: str, message: str, original: Optional[Exception] = None):
        self.provider = provider
        self.original = original
        super().__init__(f"[{provider}] {message}")

class ConfigurationError(VisionError):
    """Missing or invalid configuration."""

class UnsupportedModelError(VisionError):
    """Model not supported by any provider."""
```

---

## Future: Native Amplifier Integration

When Amplifier supports native image handling:

```python
# User drops image in Amplifier session
# Amplifier stores at: ~/.amplifier/sessions/{session_id}/images/image_001.png
# User types: "help me with [Image 1]"

# Amplifier parses reference, resolves path, calls:
from amplifier_app_vision import VisionAnalyzer

analyzer = VisionAnalyzer()  # Uses Amplifier's current model
result = analyzer.analyze(
    image_source="~/.amplifier/sessions/abc123/images/image_001.png",
    prompt="User asked: help me with this image. Describe what you see."
)

# Amplifier injects result into context
```

**Integration Points:**
1. `get_model_from_amplifier_session()` - read active model
2. Session image directory resolution
3. Context injection of vision results

---

## Test Strategy

### Unit Tests
- Provider detection for all model patterns
- API key resolution order
- Request/response normalization per provider
- Image processing (existing)

### Integration Tests  
- Each provider with real API calls (CI secrets)
- Fallback behavior when provider unavailable
- Config file loading

### Manual Testing
```bash
# Test each provider
vision analyze image.png --model gpt-4o
vision analyze image.png --model claude-sonnet-4-20250514
vision analyze image.png --model gemini-1.5-flash

# Test auto-detection
vision analyze image.png  # Uses default or Amplifier config
```

---

## Implementation Order

1. **Phase 1: Provider Abstraction** (No breaking changes)
   - Create `providers/base.py` with contracts
   - Create `providers/openai.py` (extract from current analyzer.py)
   - Create `providers/router.py` with detection logic
   - Refactor `analyzer.py` to use router (OpenAI only)
   - Tests pass, behavior unchanged

2. **Phase 2: Add Providers**
   - Implement `providers/anthropic.py`
   - Implement `providers/google.py`
   - Add dependencies
   - Integration tests

3. **Phase 3: Config Integration**
   - Implement `config.py`
   - Update router to use config
   - Document config file format

4. **Phase 4: CLI Updates**
   - Add `--provider` flag
   - Show detected provider in output
   - Config status command

---

## Success Criteria

1. All existing tests pass (backward compatibility)
2. Can analyze images with all three providers
3. Provider auto-detected from model name
4. API keys resolved from config or env vars
5. Clean error messages for missing config
6. No performance regression for single-provider use

---

## Design Decisions

### Why not use LiteLLM or similar?

While libraries like LiteLLM abstract provider differences, we choose custom implementation because:

1. **Minimal dependencies** - Only add what we need
2. **Vision-specific** - LiteLLM's vision support varies
3. **Full control** - Each provider has vision quirks
4. **Simpler debugging** - Direct API calls are traceable
5. **Amplifier integration** - Custom config/session hooks

### Why provider auto-detection vs explicit provider param?

Auto-detection from model name is the common case and reduces friction. Explicit `provider` param remains available for edge cases or testing.

### Why not lazy-load provider SDKs?

We could make providers optional and lazy-load, but:
1. Adds complexity for marginal benefit
2. Vision users likely want multiple providers
3. Clear error at import time > runtime surprise

If size becomes an issue, optional dependencies can be added later.
