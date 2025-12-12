"""Base classes and contracts for vision providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VisionRequest:
    """Normalized request format for all providers."""
    
    images: list[dict]  # [{"data": b64_str, "media_type": str} | {"url": str}]
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 1024


@dataclass
class VisionResponse:
    """Normalized response format from all providers."""
    
    text: str
    model: str
    usage: dict = field(default_factory=dict)  # prompt_tokens, completion_tokens, total_tokens
    raw_response: Optional[dict] = None


class VisionProvider(ABC):
    """Abstract base class for vision API providers.
    
    Each provider implementation handles the specifics of its API format
    while conforming to this common interface.
    """
    
    provider_name: str  # "openai", "anthropic", "google"
    
    @abstractmethod
    def __init__(self, api_key: str, model: str) -> None:
        """Initialize provider with API key and model.
        
        Args:
            api_key: Provider-specific API key
            model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514")
        """
        ...
    
    @abstractmethod
    def analyze(self, request: VisionRequest) -> VisionResponse:
        """Analyze images with the given prompt.
        
        Args:
            request: Normalized vision request
            
        Returns:
            Normalized vision response
        """
        ...
    
    @property
    @abstractmethod
    def supported_models(self) -> list[str]:
        """List of model identifiers this provider supports."""
        ...
