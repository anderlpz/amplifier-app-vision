"""Amplifier configuration integration."""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Amplifier config paths
AMPLIFIER_USER_DIR = Path.home() / ".amplifier"
AMPLIFIER_KEYS_FILE = AMPLIFIER_USER_DIR / "keys.env"
AMPLIFIER_SETTINGS_FILE = AMPLIFIER_USER_DIR / "settings.yaml"


@dataclass
class AmplifierConfig:
    """Configuration loaded from Amplifier."""
    
    model: Optional[str] = None
    api_keys: dict[str, str] = field(default_factory=dict)
    

def load_amplifier_keys() -> dict[str, str]:
    """Load API keys from Amplifier's keys.env file.
    
    Amplifier stores API keys in ~/.amplifier/keys.env in dotenv format.
    
    Returns:
        Dict mapping provider names to API keys
    """
    keys = {}
    
    if not AMPLIFIER_KEYS_FILE.exists():
        logger.debug(f"Amplifier keys file not found: {AMPLIFIER_KEYS_FILE}")
        return keys
    
    try:
        # Parse dotenv format manually (avoid adding python-dotenv as hard dep)
        content = AMPLIFIER_KEYS_FILE.read_text()
        
        for line in content.splitlines():
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            
            # Parse KEY=VALUE or KEY="VALUE"
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                
                # Map to provider name
                if key == "OPENAI_API_KEY":
                    keys["openai"] = value
                elif key == "ANTHROPIC_API_KEY":
                    keys["anthropic"] = value
                elif key in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
                    keys["google"] = value
                    
        logger.debug(f"Loaded {len(keys)} API keys from Amplifier config")
        
    except Exception as e:
        logger.warning(f"Failed to load Amplifier keys: {e}")
    
    return keys


def load_amplifier_config() -> AmplifierConfig:
    """Load full Amplifier configuration.
    
    Searches for config in:
    1. .amplifier/ (project-level, future)
    2. ~/.amplifier/ (user-level)
    
    Returns:
        AmplifierConfig with loaded settings
    """
    config = AmplifierConfig()
    
    # Load API keys
    config.api_keys = load_amplifier_keys()
    
    # TODO: Load model preference from settings.yaml if/when Amplifier adds that
    # For now, model is determined at runtime by what the user passes
    
    return config


def get_api_key_from_amplifier(provider: str) -> Optional[str]:
    """Get API key for a specific provider from Amplifier config.
    
    Args:
        provider: Provider name ("openai", "anthropic", "google")
        
    Returns:
        API key if found, None otherwise
    """
    keys = load_amplifier_keys()
    return keys.get(provider)
