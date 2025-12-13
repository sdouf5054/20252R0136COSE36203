"""
YouTube Report Generator - Gradio Web App (Colab + GPU + LLM Edition)
======================================================================

This Gradio app is optimized for Google Colab with GPU runtime,
enabling full LLM-powered video analysis with Llama 3.1-8B.

================================================================================
COLAB USAGE (simplest method - automatic gradio.live URL):
================================================================================

# Cell 1: Install dependencies
!pip install gradio google-api-python-client isodate langdetect
!pip install torch transformers accelerate bitsandbytes

# Cell 2: Set YouTube API key
import os
os.environ['YOUTUBE_API_KEY'] = 'YOUR_YOUTUBE_API_KEY'

# Cell 3: Upload files (pipeline.py, app_gradio.py)
# Use Colab file browser or:
# from google.colab import files
# files.upload()

# Cell 4: Run the app
!python app_gradio.py

# A public URL (*.gradio.live) will be displayed automatically!
# Share this URL for demo access.

================================================================================
ALTERNATIVE: Direct import in Colab notebook
================================================================================

import gradio as gr
from pipeline import PipelineConfig, ModelManager, YouTubeAPIClient, ReportGenerator

# Initialize
config = PipelineConfig()
model_manager = ModelManager(config)
model_manager.load_model()  # Load LLM

# ... (create Gradio interface)
demo.launch(share=True)  # Creates gradio.live URL

================================================================================

Author: Based on merge_notebooks.py pipeline
Version: 2.0 (Colab LLM Edition)
"""

import gradio as gr
import os
from datetime import datetime
from typing import Optional, Dict, Tuple

# Import pipeline module
try:
    from pipeline import (
        PipelineConfig,
        YouTubeAPIClient,
        ReportGenerator,
        ModelManager,
        TextPreprocessor,
        check_dependencies,
        cleanup_model,
        MODEL_PRESETS,
        list_available_models,
        YOUTUBE_API_AVAILABLE,
        TORCH_AVAILABLE
    )
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    print(f"Failed to import pipeline module: {e}")


# =============================================================================
# Global State (Colab LLM Demo: Load model once)
# =============================================================================

# Global model manager - loaded once at startup
_model_manager: Optional[ModelManager] = None
_config: Optional[PipelineConfig] = None


def get_config() -> PipelineConfig:
    """
    Get or create pipeline config.
    
    Colab LLM Demo: 환경변수로 모델 선택 가능
        - MODEL_PRESET: 프리셋 이름 (예: 'mistral-7b', 'qwen2.5-7b')
        - MODEL_NAME: 직접 모델 경로 지정
    """
    global _config
    if _config is None:
        # 환경변수에서 모델 설정 읽기
        model_preset = os.environ.get('MODEL_PRESET', '')
        model_name = os.environ.get('MODEL_NAME', '')
        
        if model_preset and model_preset in MODEL_PRESETS:
            # 프리셋 사용
            print(f"📦 Using model preset from env: {model_preset}")
            _config = PipelineConfig.from_preset(model_preset)
        elif model_name:
            # 직접 모델 지정
            print(f"📦 Using custom model from env: {model_name}")
            _config = PipelineConfig(model_name=model_name)
        else:
            # 기본값 (Llama 3.1-8B)
            print("📦 Using default model: llama3.1-8b")
            _config = PipelineConfig.from_preset('llama3.1-8b')
        
        _config.enable_detailed_logging = True
        _config.log_token_counts = True
    return _config


def get_model_manager() -> ModelManager:
    """
    Get or create model manager.
    
    Colab LLM Demo: Loads Llama 3.1-8B on first call.
    """
    global _model_manager
    if _model_manager is None:
        config = get_config()
        _model_manager = ModelManager(config)
        print("\n" + "=" * 60)
        print("🚀 Loading LLM Model for Gradio App...")
        print("=" * 60)
        success = _model_manager.load_model()
        if success:
            print(f"✅ LLM ready! Mode: FULL ({_model_manager.device.upper()})")
        else:
            print(f"⚠️ LLM not loaded. Mode: FALLBACK")
    return _model_manager


# =============================================================================
# Helper Functions
# =============================================================================

def get_youtube_api_key() -> str:
    """Get YouTube API key from environment"""
    return os.environ.get('YOUTUBE_API_KEY', '')


def format_number(num: int) -> str:
    """Format large numbers with K, M, B suffixes"""
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(num)


def format_duration(seconds: int) -> str:
    """Format duration in seconds to HH:MM:SS or MM:SS"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


# =============================================================================
# Core Functions
# =============================================================================

def fetch_video_info(video_url: str) -> Tuple[str, str, Optional[str], Optional[Dict]]:
    """
    Fetch video information from YouTube URL.

    Returns:
        tuple: (status_message, video_info_html, thumbnail_url, video_data)
    """
    api_key = get_youtube_api_key()

    if not api_key:
        return (
            "❌ YouTube API key not configured. Set YOUTUBE_API_KEY environment variable.",
            "",
            None,
            None
        )

    if not YOUTUBE_API_AVAILABLE:
        return (
            "❌ YouTube API dependencies not installed.",
            "",
            None,
            None
        )

    if not video_url or not video_url.strip():
        return (
            "⚠️ Please enter a YouTube URL.",
            "",
            None,
            None
        )

    try:
        api_client = YouTubeAPIClient(api_key)
        video_id = api_client.extract_video_id(video_url.strip())

        if not video_id:
            return (
                "❌ Invalid YouTube URL. Please check and try again.",
                "",
                None,
                None
            )

        video_data = api_client.get_video_info(video_id)

        if not video_data:
            return (
                "❌ Could not fetch video information. Video may be private or deleted.",
                "",
                None,
                None
            )

        # Fetch comments
        config = get_config()
        comments = api_client.get_comments(video_id, max_results=config.max_comments_to_process)
        video_data['comments'] = comments

        video_info = video_data['video_info']

        # Format video info HTML
        info_html = f"""
        <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
            <h2 style="margin-top: 0; margin-bottom: 15px;">{video_info.get('title', 'Unknown')}</h2>
            <p style="margin: 5px 0;"><strong>📺 Channel:</strong> {video_info.get('channel_title', 'Unknown')}</p>
            <p style="margin: 5px 0;"><strong>📅 Published:</strong> {video_info.get('published_at', '')[:10] if video_info.get('published_at') else 'N/A'}</p>
            <p style="margin: 5px 0;"><strong>⏱️ Duration:</strong> {format_duration(video_info.get('duration', 0))}</p>
            <hr style="border-color: rgba(255,255,255,0.3); margin: 15px 0;">
            <div style="display: flex; justify-content: space-around; text-align: center;">
                <div>
                    <div style="font-size: 24px; font-weight: bold;">{format_number(video_info.get('view_count', 0))}</div>
                    <div style="font-size: 12px; opacity: 0.8;">Views</div>
                </div>
                <div>
                    <div style="font-size: 24px; font-weight: bold;">{format_number(video_info.get('like_count', 0))}</div>
                    <div style="font-size: 12px; opacity: 0.8;">Likes</div>
                </div>
                <div>
                    <div style="font-size: 24px; font-weight: bold;">{format_number(video_info.get('comment_count', 0))}</div>
                    <div style="font-size: 12px; opacity: 0.8;">Comments</div>
                </div>
            </div>
            <p style="margin-top: 15px; font-size: 14px; opacity: 0.9;">
                📝 Fetched {len(comments)} comments for AI analysis
            </p>
        </div>
        """

        thumbnail_url = video_info.get('thumbnail_url', '')

        return (
            f"✅ Video found: {video_info.get('title', 'Unknown')[:50]}...",
            info_html,
            thumbnail_url,
            video_data
        )

    except Exception as e:
        return (
            f"❌ Error: {str(e)}",
            "",
            None,
            None
        )


def generate_report(
    video_data: Optional[Dict],
    output_language: str,
    max_comments: int,
    enable_quality_gate: bool,
    min_summary_length: int,
    max_new_tokens: int,
    temperature: float
) -> Tuple[str, str]:
    """
    Generate full report from video data.

    Returns:
        tuple: (report_markdown, status_message)
    """
    if not video_data:
        return "", "⚠️ Please fetch video information first."

    if not PIPELINE_AVAILABLE:
        return "", "❌ Pipeline module not available."

    try:
        # Update config with UI settings
        config = get_config()
        config.output_language = output_language
        config.max_comments_to_process = max_comments
        config.enable_quality_gate = enable_quality_gate
        config.min_summary_length = min_summary_length
        config.max_new_tokens = max_new_tokens
        config.temperature = temperature

        # Get model manager
        model_manager = get_model_manager()

        # Generate report
        report_gen = ReportGenerator(config, model_manager)
        report = report_gen.generate_full_report(video_data)

        # Status message with LLM info
        status = model_manager.get_status()
        if status['loaded']:
            status_msg = f"✅ Report generated with FULL LLM ({status['device'].upper()}) | {status['total_calls']} LLM calls"
        else:
            status_msg = "⚠️ Report generated in FALLBACK mode (LLM not loaded)"

        return report, status_msg

    except Exception as e:
        return "", f"❌ Error: {str(e)}"


def get_llm_status() -> str:
    """Get current LLM status for display"""
    if not PIPELINE_AVAILABLE:
        return "❌ Pipeline not available"
    
    model_manager = get_model_manager()
    status = model_manager.get_status()
    
    if status['loaded']:
        return f"""🤖 **LLM Status: ACTIVE**
- Model: `{status['model_name'].split('/')[-1]}`
- Device: `{status['device'].upper()}`
- GPU Memory: `{status['gpu_memory']}`
- Total Calls: `{status['total_calls']}`"""
    else:
        return f"""⚠️ **LLM Status: FALLBACK**
- Error: `{status['error']}`
- Summaries will be placeholders"""


def get_token_stats() -> str:
    """Get token efficiency statistics"""
    if not PIPELINE_AVAILABLE:
        return "N/A"
    
    model_manager = get_model_manager()
    return model_manager.get_token_efficiency_report()


# =============================================================================
# Gradio Interface
# =============================================================================

def create_interface() -> gr.Blocks:
    """
    Create Gradio interface.
    
    Colab LLM Demo: Full-featured UI with LLM controls.
    """

    # Check dependencies
    api_key = get_youtube_api_key()
    api_status = "✅ YouTube API Key configured" if api_key else "❌ YouTube API Key NOT configured"
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .llm-status {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        margin-bottom: 15px;
    }
    """

    with gr.Blocks(
        title="YouTube Report Generator (LLM)",
        css=custom_css,
        theme=gr.themes.Soft()
    ) as demo:

        # Header
        gr.Markdown("""
        # 🤖 YouTube Report Generator (Colab LLM Edition)
        
        > **Full AI Analysis**: Generate comprehensive video reports using Llama 3.1-8B.
        > 
        > Running on Colab? Use `launch(share=True)` for a public gradio.live URL!
        """)

        # Status row
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(f"**{api_status}**")
            with gr.Column(scale=2):
                llm_status_display = gr.Markdown(get_llm_status())

        # Main layout
        with gr.Row():
            # Left column - Input & Settings
            with gr.Column(scale=1):
                gr.Markdown("### 🔗 YouTube URL")
                
                url_input = gr.Textbox(
                    label="Enter URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    lines=1
                )
                
                fetch_btn = gr.Button("📥 Fetch Video Info", variant="primary", size="lg")
                
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1
                )
                
                # Settings accordion
                with gr.Accordion("⚙️ Settings", open=False):
                    output_lang = gr.Dropdown(
                        label="Report Language",
                        choices=["English", "Korean", "Japanese", "Chinese", "Spanish", "French", "German"],
                        value="English"
                    )
                    
                    max_comments_slider = gr.Slider(
                        label="Max Comments",
                        minimum=10,
                        maximum=200,
                        value=100,
                        step=10
                    )
                
                # LLM Parameters accordion
                with gr.Accordion("🔬 LLM Parameters", open=False):
                    qgate_checkbox = gr.Checkbox(
                        label="Enable Quality Gate",
                        value=True,
                        info="Validate and regenerate low-quality outputs"
                    )
                    
                    min_len_slider = gr.Slider(
                        label="Min Summary Length",
                        minimum=50,
                        maximum=400,
                        value=100,
                        step=50
                    )
                    
                    max_tokens_slider = gr.Slider(
                        label="Max New Tokens",
                        minimum=128,
                        maximum=1024,
                        value=512,
                        step=64
                    )
                    
                    temp_slider = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=1.5,
                        value=0.7,
                        step=0.1
                    )

            # Right column - Preview & Results
            with gr.Column(scale=2):
                gr.Markdown("### 📺 Video Preview")
                
                thumbnail_output = gr.Image(
                    label="Thumbnail",
                    type="filepath",
                    height=250
                )
                
                video_info_output = gr.HTML()

        # Hidden state
        video_data_state = gr.State(value=None)

        # Analyze section
        gr.Markdown("---")
        
        with gr.Row():
            analyze_btn = gr.Button("🚀 Generate AI Report", variant="primary", size="lg", scale=2)
            clear_btn = gr.Button("🔄 Clear", scale=1)

        # Output section
        gr.Markdown("### 📋 Generated Report")
        
        report_status = gr.Textbox(label="Generation Status", interactive=False, lines=1)
        
        report_output = gr.Markdown(label="Report")

        # Token stats
        with gr.Accordion("📊 Token Statistics", open=False):
            token_stats_output = gr.Textbox(
                label="Token Efficiency Report",
                interactive=False,
                lines=8
            )
            refresh_stats_btn = gr.Button("🔄 Refresh Stats")

        # Event handlers
        fetch_btn.click(
            fn=fetch_video_info,
            inputs=[url_input],
            outputs=[status_output, video_info_output, thumbnail_output, video_data_state]
        )

        analyze_btn.click(
            fn=generate_report,
            inputs=[
                video_data_state,
                output_lang,
                max_comments_slider,
                qgate_checkbox,
                min_len_slider,
                max_tokens_slider,
                temp_slider
            ],
            outputs=[report_output, report_status]
        )

        def clear_all():
            return "", "", "", None, None, "", ""

        clear_btn.click(
            fn=clear_all,
            outputs=[url_input, status_output, video_info_output, thumbnail_output, video_data_state, report_output, report_status]
        )

        refresh_stats_btn.click(
            fn=get_token_stats,
            outputs=[token_stats_output]
        )

        # Footer
        gr.Markdown("""
        ---
        <div style="text-align: center; color: gray; font-size: 0.9em;">
            <strong>YouTube Report Generator (Colab LLM Edition) v2.0</strong><br>
            🤖 Powered by Llama 3.1-8B | Based on merge_notebooks.py pipeline<br>
            <br>
            <strong>Colab Tip:</strong> Run with <code>demo.launch(share=True)</code> to get a public gradio.live URL!
        </div>
        """)

    return demo


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting YouTube Report Generator (Colab LLM Edition)")
    print("=" * 60)
    
    # Pre-load model manager
    if PIPELINE_AVAILABLE:
        print("\n📦 Pre-loading LLM model...")
        _ = get_model_manager()
    
    # Create and launch interface
    demo = create_interface()
    
    # Launch settings
    # Colab LLM Demo: share=True creates a public gradio.live URL
    print("\n🌐 Launching Gradio interface...")
    print("   share=True will create a public URL (gradio.live)")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Colab LLM Demo: Creates public URL automatically
        show_error=True
    )
