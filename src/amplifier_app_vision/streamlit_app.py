"""Streamlit web interface for Vision app."""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Vision - Image Analysis",
    page_icon="👁️",
    layout="wide",
)

# Title
st.title("👁️ Vision")
st.markdown("*Image analysis powered by OpenAI GPT-4 Vision*")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key
    api_key = st.text_input(
        "OpenAI API Key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
        help="Your OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Model selection
    model = st.selectbox(
        "Model",
        ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        index=0,
        help="OpenAI model with vision capabilities",
    )
    
    # Quality preset
    quality = st.selectbox(
        "Image Quality",
        ["quick", "normal", "detailed", "full"],
        index=1,
        help="Image optimization preset (affects tokens and detail)",
    )
    
    st.divider()
    
    st.markdown("""
    **Quality Presets:**
    - **quick**: 512px, fast, ~500 tokens
    - **normal**: 1024px, balanced, ~1500 tokens  
    - **detailed**: 1568px, high detail, ~4000 tokens
    - **full**: Original size, max quality
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Image Input")
    
    # Input method tabs
    input_tab1, input_tab2 = st.tabs(["Upload File", "Enter URL"])
    
    with input_tab1:
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg", "gif", "webp"],
            help="Upload an image to analyze",
        )
        
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded image", use_container_width=True)
    
    with input_tab2:
        image_url = st.text_input(
            "Image URL",
            placeholder="https://example.com/image.png",
            help="URL to an image",
        )
        
        if image_url:
            try:
                st.image(image_url, caption="Image from URL", use_container_width=True)
            except Exception as e:
                st.error(f"Could not load image: {e}")

with col2:
    st.subheader("💬 Analysis")
    
    # Prompt input
    analysis_type = st.radio(
        "Analysis Type",
        ["Custom Prompt", "Describe", "Extract Text"],
        horizontal=True,
    )
    
    if analysis_type == "Custom Prompt":
        prompt = st.text_area(
            "Your prompt",
            value="What's in this image?",
            height=100,
            help="Ask anything about the image",
        )
    elif analysis_type == "Describe":
        prompt = "Describe this image in detail. Include what you see, any text, colors, and important elements."
        st.info("Will provide a detailed description of the image")
    else:  # Extract Text
        prompt = "Extract and transcribe all text visible in this image. Format it cleanly, preserving structure where possible."
        st.info("Will extract and transcribe all visible text (OCR)")
    
    # Analyze button
    analyze_clicked = st.button(
        "🔍 Analyze Image",
        type="primary",
        use_container_width=True,
        disabled=not (uploaded_file or image_url) or not api_key,
    )
    
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar")

# Results section
st.divider()

if analyze_clicked and (uploaded_file or image_url):
    try:
        from .analyzer import VisionAnalyzer
        
        analyzer = VisionAnalyzer(model=model, quality=quality)
        
        with st.spinner("Analyzing image..."):
            # Determine source
            if uploaded_file:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    suffix=Path(uploaded_file.name).suffix,
                    delete=False,
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    source = tmp.name
            else:
                source = image_url
            
            # Analyze
            result = analyzer.analyze(source, prompt=prompt)
            
            # Clean up temp file
            if uploaded_file and Path(source).exists():
                Path(source).unlink()
        
        # Display result
        st.subheader("📋 Result")
        st.markdown(result["text"])
        
        # Show stats
        with st.expander("📊 Details"):
            usage = result.get("usage", {})
            metadata = result.get("image_metadata", {})
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Total Tokens", f"{usage.get('total_tokens', 0):,}")
            
            with col_b:
                st.metric("Prompt Tokens", f"{usage.get('prompt_tokens', 0):,}")
            
            with col_c:
                st.metric("Completion Tokens", f"{usage.get('completion_tokens', 0):,}")
            
            if metadata:
                st.markdown("**Image Processing:**")
                st.json({
                    "filename": metadata.get("filename", "N/A"),
                    "original_size": f"{metadata.get('original_size', 0):,} bytes",
                    "optimized_size": f"{metadata.get('optimized_size', 0):,} bytes",
                    "compression": f"{metadata.get('compression_percent', 0):.1f}%",
                    "estimated_tokens": metadata.get("estimated_tokens", 0),
                })
        
        # Copy button
        st.download_button(
            "📥 Download Result",
            data=result["text"],
            file_name="vision_analysis.txt",
            mime="text/plain",
        )
        
    except Exception as e:
        st.error(f"Analysis failed: {e}")

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Vision - Part of the Amplifier ecosystem"
    "</div>",
    unsafe_allow_html=True,
)
