"""Streamlit web interface for vision analysis."""

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def launch_web_ui(verbose: bool = False):
    """Launch the Streamlit web UI.
    
    Args:
        verbose: Enable verbose logging
    """
    # Get the path to the Streamlit app
    app_path = Path(__file__).parent / "streamlit_app.py"
    
    if not app_path.exists():
        raise FileNotFoundError(f"Streamlit app not found at {app_path}")
    
    # Build streamlit command
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    
    if verbose:
        cmd.append("--logger.level=debug")
    
    print(f"Launching Vision web interface...")
    print(f"Open http://localhost:8501 in your browser\n")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to launch Streamlit: {e}")
        raise
