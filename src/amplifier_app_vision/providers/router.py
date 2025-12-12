"""Provider routing and detection."""

import os
import logging
from typing import Optional

from .base import VisionProvider

logger = logging.getLogger(__name__)


class ProviderRouter:
    """Routes requests to appropriate provider based on model name."""
    
    # Model prefix patterns -> provider name
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
        """Detect provider from model name prefix.
        
        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514")
            
        Returns:
            Provider name ("openai", "anthropic", "google")
            
        Raises:
            ValueError: If model pattern not recognized
        """
        model_lower = model.lower()
        
        for prefix, provider in self.MODEL_PATTERNS.items():
            if model_lower.startswith(prefix):
                return provider
        
        raise ValueError(
            f"Unknown model '{model}'. Expected prefix: {list(self.MODEL_PATTERNS.keys())}"
        )
    
    def resolve_api_key(
        self, 
        provider: str, 
        explicit_key: Optional[str] = None
    ) -> str:
        """Resolve API key from explicit param, Amplifier config, or env var.
        
        Resolution order:
        1. Explicit api_key parameter
        2. Amplifier config (~/.amplifier/keys.env)
        3. Environment variables
        
        Args:
            provider: Provider name
            explicit_key: Explicitly provided API key
            
        Returns:
            Resolved API key
            
        Raises:
            ValueError: If no API key found
        """
        # 1. Explicit key takes priority
        if explicit_key:
            return explicit_key
        
        # 2. Check Amplifier config (~/.amplifier/keys.env)
        try:
            from ..config import get_api_key_from_amplifier
            amplifier_key = get_api_key_from_amplifier(provider)
            if amplifier_key:
                logger.debug(f"Using API key from Amplifier config for {provider}")
                return amplifier_key
        except Exception as e:
            logger.debug(f"Could not load Amplifier config: {e}")
        
        # 3. Check environment variables
        env_vars = self.API_KEY_ENV_VARS.get(provider, [])
        for var in env_vars:
            key = os.environ.get(var)
            if key:
                return key
        
        raise ValueError(
            f"No API key found for {provider}. "
            f"Set {' or '.join(env_vars)} environment variable "
            f"or add to ~/.amplifier/keys.env"
        )
    
    def get_provider(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> VisionProvider:
        """Get configured provider instance.
        
        Args:
            model: Model name (auto-detects provider if not specified)
            provider: Force specific provider (overrides detection)
            api_key: Explicit API key
            
        Returns:
            Configured VisionProvider instance
        """
        # Determine provider
        if provider:
            provider_name = provider
            model = model or self.DEFAULT_MODELS.get(provider_name)
        elif model:
            provider_name = self.detect_provider(model)
        else:
            # Default to OpenAI
            provider_name = "openai"
            model = self.DEFAULT_MODELS["openai"]
        
        # Resolve API key
        resolved_key = self.resolve_api_key(provider_name, api_key)
        
        # Instantiate provider
        return self._create_provider(provider_name, resolved_key, model)
    
    def _create_provider(
        self, 
        provider_name: str, 
        api_key: str, 
        model: str
    ) -> VisionProvider:
        """Create provider instance.
        
        Args:
            provider_name: Provider identifier
            api_key: API key for provider
            model: Model to use
            
        Returns:
            Provider instance
        """
        if provider_name == "openai":
            from .openai import OpenAIProvider
            return OpenAIProvider(api_key=api_key, model=model)
        
        elif provider_name == "anthropic":
            from .anthropic import AnthropicProvider
            return AnthropicProvider(api_key=api_key, model=model)
        
        elif provider_name == "google":
            from .google import GoogleProvider
            return GoogleProvider(api_key=api_key, model=model)
        
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
