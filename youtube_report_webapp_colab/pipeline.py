"""
YouTube Report Generator Pipeline Module (Colab + GPU + LLM Edition)
====================================================================

This module is optimized for Google Colab with GPU runtime,
enabling full LLM-powered video analysis with Llama 3.1-8B.

MODES:
- FULL LLM Mode: GPU available, model loads successfully → Real AI summaries
- FALLBACK Mode: No GPU or load fails → Placeholder summaries

USAGE IN COLAB:
    !pip install -r requirements.txt
    from pipeline import (
        PipelineConfig,
        YouTubeAPIClient,
        ModelManager,
        ReportGenerator,
        generate_report_from_url,
    )

    config = PipelineConfig()
    model_manager = ModelManager(config)
    model_manager.load_model()  # Actually loads Llama 3.1-8B

Original: merge_notebooks.ipynb / merge_notebooks.py
Version: 2.0 (Colab LLM Edition)
"""

import json
import re
import os
import gc
import warnings
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# Dependency Checks
# =============================================================================

TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
LANGDETECT_AVAILABLE = False
YOUTUBE_API_AVAILABLE = False

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    np = None
    print("[pipeline.py] Warning: PyTorch not found. LLM features disabled.")

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        BitsAndBytesConfig,
        pipeline as hf_pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("[pipeline.py] Warning: Transformers not found. LLM features disabled.")

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    detect = None
    LangDetectException = Exception

try:
    import isodate
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    isodate = None
    build = None
    HttpError = Exception


# =============================================================================
# Configuration
# =============================================================================

# =============================================================================
# Model Presets (No Approval Required)
# =============================================================================

# 승인 없이 바로 사용 가능한 모델들 (Colab에서 테스트 완료)
MODEL_PRESETS = {
    # -------------------------------------------------------------------------
    # 추천: 승인 불필요, 성능 좋음
    # -------------------------------------------------------------------------
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "description": "Mistral 7B Instruct v0.3 - 빠르고 품질 좋음 (추천)",
        "use_4bit": True,
        "vram_gb": 5.0
    },
    "qwen2.5-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen 2.5 7B - 다국어 지원 우수 (한국어 추천)",
        "use_4bit": True,
        "vram_gb": 5.0
    },
    "gemma2-9b": {
        "name": "google/gemma-2-9b-it",
        "description": "Google Gemma 2 9B - 고품질, 약간 느림",
        "use_4bit": True,
        "vram_gb": 6.0
    },
    "phi3-mini": {
        "name": "microsoft/Phi-3-mini-4k-instruct",
        "description": "Microsoft Phi-3 Mini 3.8B - 가볍고 빠름",
        "use_4bit": True,
        "vram_gb": 3.0
    },
    "phi3.5-mini": {
        "name": "microsoft/Phi-3.5-mini-instruct", 
        "description": "Microsoft Phi-3.5 Mini - 최신, 성능 개선",
        "use_4bit": True,
        "vram_gb": 3.5
    },
    
    # -------------------------------------------------------------------------
    # 경량 모델 (T4에서도 빠름)
    # -------------------------------------------------------------------------
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "TinyLlama 1.1B - 매우 가벼움, 품질 낮음",
        "use_4bit": False,  # 이미 작아서 양자화 불필요
        "vram_gb": 2.5
    },
    "stablelm-2": {
        "name": "stabilityai/stablelm-2-1_6b-chat",
        "description": "StableLM 2 1.6B - 가볍고 안정적",
        "use_4bit": False,
        "vram_gb": 3.0
    },
    
    # -------------------------------------------------------------------------
    # 승인 필요 (참고용)
    # -------------------------------------------------------------------------
    "llama3.1-8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "description": "Llama 3.1 8B - 최고 품질 (HF 승인 필요)",
        "use_4bit": True,
        "vram_gb": 5.5,
        "requires_approval": True
    },
    "llama3.2-3b": {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "description": "Llama 3.2 3B - 가벼운 Llama (HF 승인 필요)",
        "use_4bit": True,
        "vram_gb": 3.0,
        "requires_approval": True
    },
}

def list_available_models():
    """사용 가능한 모델 목록 출력"""
    print("=" * 70)
    print("📦 Available Model Presets")
    print("=" * 70)
    print("\n🟢 No Approval Required (바로 사용 가능):\n")
    
    for key, info in MODEL_PRESETS.items():
        if not info.get("requires_approval", False):
            print(f"  '{key}'")
            print(f"      {info['description']}")
            print(f"      VRAM: ~{info['vram_gb']}GB | 4-bit: {info['use_4bit']}")
            print()
    
    print("🟡 Requires HF Approval (승인 필요):\n")
    for key, info in MODEL_PRESETS.items():
        if info.get("requires_approval", False):
            print(f"  '{key}'")
            print(f"      {info['description']}")
            print()
    
    print("=" * 70)
    print("Usage: config = PipelineConfig.from_preset('mistral-7b')")
    print("   or: config = PipelineConfig(model_name='your/custom-model')")
    print("=" * 70)


@dataclass
class PipelineConfig:
    """
    Full pipeline configuration.
    
    Colab LLM Demo: All settings can be adjusted via UI sliders.
    
    Usage:
        # 프리셋 사용 (추천)
        config = PipelineConfig.from_preset('mistral-7b')
        config = PipelineConfig.from_preset('qwen2.5-7b')  # 한국어 추천
        
        # 커스텀 모델
        config = PipelineConfig(model_name='your/custom-model')
        
        # 모델 목록 보기
        list_available_models()
    """

    # -------------------------------------------------------------------------
    # LLM Model Settings (Colab GPU optimized)
    # -------------------------------------------------------------------------
    # 기본값: Llama 3.1-8B (HF 승인 필요)
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    use_4bit: bool = True  # 4-bit quantization for Colab T4/V100
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> 'PipelineConfig':
        """
        프리셋으로 Config 생성.
        
        Args:
            preset_name: 프리셋 이름 ('mistral-7b', 'qwen2.5-7b', 등)
            **overrides: 추가로 덮어쓸 설정
            
        Returns:
            PipelineConfig instance
            
        Example:
            config = PipelineConfig.from_preset('mistral-7b')
            config = PipelineConfig.from_preset('qwen2.5-7b', max_new_tokens=256)
        """
        if preset_name not in MODEL_PRESETS:
            available = list(MODEL_PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        
        preset = MODEL_PRESETS[preset_name]
        
        if preset.get("requires_approval", False):
            print(f"⚠️  Warning: '{preset_name}' requires HuggingFace approval.")
            print(f"    Apply at: https://huggingface.co/{preset['name']}")
        
        config = cls(
            model_name=preset["name"],
            use_4bit=preset["use_4bit"],
            **overrides
        )
        
        print(f"✅ Config created with preset '{preset_name}'")
        print(f"   Model: {preset['name']}")
        print(f"   VRAM: ~{preset['vram_gb']}GB")
        
        return config

    # -------------------------------------------------------------------------
    # YouTube API Settings
    # -------------------------------------------------------------------------
    youtube_api_key: Optional[str] = None

    # -------------------------------------------------------------------------
    # Processing Settings
    # -------------------------------------------------------------------------
    max_description_length: int = 2000
    max_comments_to_process: int = 100
    min_comment_length: int = 10
    remove_urls: bool = True
    detect_language: bool = True

    # -------------------------------------------------------------------------
    # Output Settings
    # -------------------------------------------------------------------------
    output_language: str = "English"
    multilingual_understanding: bool = True

    # -------------------------------------------------------------------------
    # Token Efficiency Settings
    # -------------------------------------------------------------------------
    enable_dynamic_tokens: bool = True
    token_efficiency_mode: str = "adaptive"  # conservative / adaptive / aggressive

    # -------------------------------------------------------------------------
    # Quality Gate Settings (adjustable in UI)
    # -------------------------------------------------------------------------
    enable_quality_gate: bool = True
    min_summary_length: int = 100
    min_keyword_diversity: int = 10
    max_regeneration_attempts: int = 2
    quality_gate_temperature: float = 0.5

    # -------------------------------------------------------------------------
    # Category/Sentiment Models (disabled for web demo)
    # -------------------------------------------------------------------------
    use_category_model: bool = False
    category_model_path: Optional[str] = None
    enable_llm_category_correction: bool = False
    use_sentiment_model: bool = False
    sentiment_model_path: Optional[str] = None

    # -------------------------------------------------------------------------
    # Logging (enable for Colab debugging)
    # -------------------------------------------------------------------------
    enable_detailed_logging: bool = True
    log_token_counts: bool = True

    def __post_init__(self):
        if self.youtube_api_key is None:
            self.youtube_api_key = os.environ.get("YOUTUBE_API_KEY")


# =============================================================================
# Text Preprocessing
# =============================================================================

class TextPreprocessor:
    """Text preprocessing utilities"""

    LANG_NAMES = {
        'ko': 'Korean', 'en': 'English', 'ja': 'Japanese',
        'zh-cn': 'Chinese (Simplified)', 'zh-tw': 'Chinese (Traditional)',
        'de': 'German', 'fr': 'French', 'es': 'Spanish',
        'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian',
        'ar': 'Arabic', 'hi': 'Hindi', 'th': 'Thai', 'vi': 'Vietnamese',
        'nl': 'Dutch', 'pl': 'Polish', 'tr': 'Turkish',
        'sv': 'Swedish', 'da': 'Danish', 'no': 'Norwegian', 'fi': 'Finnish',
    }

    @staticmethod
    def remove_urls(text: str) -> str:
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)

    @staticmethod
    def clean_text(text: str, remove_urls: bool = True) -> str:
        if not text:
            return ""
        if remove_urls:
            text = TextPreprocessor.remove_urls(text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def detect_language(text: str) -> str:
        if not LANGDETECT_AVAILABLE:
            return "Unknown"
        try:
            if not text or len(text.strip()) < 10:
                return "Unknown"
            lang = detect(text)
            return TextPreprocessor.LANG_NAMES.get(lang, lang.title())
        except:
            return "Unknown"

    @staticmethod
    def get_language_distribution(texts: List[str]) -> Dict[str, int]:
        lang_dist = {}
        for text in texts:
            lang = TextPreprocessor.detect_language(text)
            lang_dist[lang] = lang_dist.get(lang, 0) + 1
        return lang_dist


# =============================================================================
# Prompt Templates
# =============================================================================

class PromptTemplates:
    """Prompt templates for video analysis (LLM mode)"""

    @staticmethod
    def get_video_summary_prompt(title: str, description: str, category_id: str,
                                  duration: str, output_lang: str = "English") -> str:
        return f"""You are analyzing a YouTube video. Based on the information provided, create a comprehensive summary in {output_lang}.

Video Information:
- Title: {title}
- Description: {description}
- Category ID: {category_id}
- Duration: {duration}

Instructions:
1. First, provide a KEY POINT (1-2 sentences) that captures the video's essence
2. Then, provide a DETAILED SUMMARY (3-5 sentences) with more context
3. Write ONLY in {output_lang}, regardless of the input language
4. Focus on what viewers will learn or experience
5. Be concise and informative

Format your response EXACTLY as:
KEY POINT:
[Your 1-2 sentence key point here]

DETAILED SUMMARY:
[Your 3-5 sentence detailed summary here]
"""

    @staticmethod
    def get_reaction_summary_prompt(comments: List[str], output_lang: str = "English") -> str:
        comments_text = "\n".join([f"- {c}" for c in comments[:100]])
        return f"""You are analyzing audience reactions to a YouTube video. Based on the comments provided, create a comprehensive reaction summary in {output_lang}.

Comments:
{comments_text}

Instructions:
1. First, provide a KEY POINT (1-2 sentences) about the overall reaction
2. Then, provide a DETAILED SUMMARY (3-5 sentences) with:
   - Majority opinion (what most viewers think)
   - Notable minority views (outliers, different perspectives)
   - Language patterns (if mixed Korean/English/Japanese, note code-switching)
3. After the summary, provide a SENTIMENT BREAKDOWN estimate:
   Positive: X%  Negative: Y%  Neutral: Z%
4. Write ONLY in {output_lang}, regardless of comment languages
5. Ignore spam, absurd, or irrelevant comments (<5% outliers)

Format your response EXACTLY as:
KEY POINT:
[Your 1-2 sentence key point here]

DETAILED SUMMARY:
[Your 3-5 sentence detailed analysis here]

SENTIMENT BREAKDOWN:
Positive: X%  Negative: Y%  Neutral: Z%
"""


# =============================================================================
# Quality Gate
# =============================================================================

class QualityGate:
    """Quality validation system for generated summaries"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.validation_stats = {
            "total_validations": 0,
            "passed_first_time": 0,
            "regenerated": 0,
            "failed_final": 0
        }

    def validate_summary(self, summary: str, summary_type: str = "video") -> Tuple[bool, List[str]]:
        if not self.config.enable_quality_gate:
            return True, []

        self.validation_stats["total_validations"] += 1
        failure_reasons = []

        if len(summary) < self.config.min_summary_length:
            failure_reasons.append(f"Too short ({len(summary)} < {self.config.min_summary_length})")

        words = summary.lower().split()
        unique_words = set(words)
        if len(unique_words) < self.config.min_keyword_diversity:
            failure_reasons.append(f"Low diversity ({len(unique_words)} < {self.config.min_keyword_diversity})")

        if "KEY POINT" not in summary:
            failure_reasons.append("Missing 'KEY POINT' section")

        is_valid = len(failure_reasons) == 0
        if is_valid:
            self.validation_stats["passed_first_time"] += 1

        return is_valid, failure_reasons

    def get_quality_report(self) -> str:
        stats = self.validation_stats
        total = stats["total_validations"]

        if total == 0:
            return "Quality Gate: No validations performed"

        pass_rate = (stats["passed_first_time"] / total * 100) if total > 0 else 0
        regen = stats["regenerated"]
        failed = stats["failed_final"]

        return f"""Quality Gate Report:
  Validations: {total}
  Passed (1st try): {stats['passed_first_time']} ({pass_rate:.1f}%)
  Regenerated: {regen}
  Failed: {failed}"""


# =============================================================================
# Model Manager (Colab GPU + LLM Edition)
# =============================================================================

class ModelManager:
    """
    LLM Model Manager with full GPU support for Colab.
    
    MODES:
    - _loaded = True  → FULL LLM mode (real generation)
    - _loaded = False → FALLBACK mode (placeholder text)
    
    Colab LLM Demo: Call load_model() explicitly to load Llama 3.1-8B.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.token_stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_calls": 0,
            "tokens_saved": 0
        }
        self._loaded = False
        self._load_error = None

    def load_model(self) -> bool:
        """
        Load the LLM model (Llama 3.1-8B with 4-bit quantization).
        
        Returns:
            bool: True if loaded successfully, False otherwise
        
        Colab LLM Demo: This actually loads the 8B model onto GPU.
        """
        print("=" * 60)
        print("🚀 Loading LLM Model")
        print("=" * 60)

        # Check dependencies
        if not TORCH_AVAILABLE:
            self._load_error = "PyTorch not installed"
            print(f"❌ {self._load_error}")
            return False

        if not TRANSFORMERS_AVAILABLE:
            self._load_error = "Transformers not installed"
            print(f"❌ {self._load_error}")
            return False

        # Check GPU
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            self.device = "cpu"
            print("⚠️  No GPU detected. Running on CPU (slow, may fail for 8B model).")

        print(f"📦 Model: {self.config.model_name}")
        print(f"⚙️  4-bit Quantization: {self.config.use_4bit}")

        try:
            # Load tokenizer
            print("📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✅ Tokenizer loaded")

            # Load model with 4-bit quantization
            print("📥 Loading model (this may take a few minutes)...")

            if self.config.use_4bit and self.device == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # Full precision or CPU
                dtype = torch.float16 if self.device == "cuda" else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=dtype,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)

            self._loaded = True

            # Report memory usage
            if self.device == "cuda":
                mem_used = torch.cuda.memory_allocated(0) / 1e9
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"✅ Model loaded! GPU Memory: {mem_used:.2f} / {mem_total:.1f} GB")
            else:
                print(f"✅ Model loaded on {self.device}")

            print("=" * 60)
            return True

        except Exception as e:
            self._load_error = str(e)
            print(f"❌ Failed to load model: {self._load_error}")
            print("=" * 60)
            return False

    def dynamic_max_tokens(self, input_length: int) -> int:
        """Calculate dynamic token limit based on input length"""
        if not self.config.enable_dynamic_tokens:
            return self.config.max_new_tokens

        mode = self.config.token_efficiency_mode
        if mode == "conservative":
            if input_length < 500:
                return 320
            elif input_length < 1500:
                return 448
            return 512
        elif mode == "aggressive":
            if input_length < 500:
                return 192
            elif input_length < 1500:
                return 320
            return 448
        else:  # adaptive (default)
            if input_length < 500:
                return 256
            elif input_length < 1500:
                return 384
            return 512

    def generate(self, prompt: str, temperature: Optional[float] = None,
                 force_max_tokens: Optional[int] = None) -> str:
        """
        Generate text using the LLM.
        
        Returns:
            str: Generated text (real if model loaded, fallback otherwise)
        """
        # Fallback path
        if not self._loaded or self.model is None or self.tokenizer is None:
            if self.config.enable_detailed_logging:
                print("    [FALLBACK MODE - LLM not loaded]")
            return self._generate_fallback(prompt)

        # Full LLM path
        try:
            input_tokens = len(self.tokenizer.encode(prompt))
            max_tokens = force_max_tokens or self.dynamic_max_tokens(input_tokens)
            tokens_saved = self.config.max_new_tokens - max_tokens

            if self.config.enable_dynamic_tokens and tokens_saved > 0:
                self.token_stats["tokens_saved"] += tokens_saved

            if self.config.log_token_counts:
                print(f"    [LLM] Input: {input_tokens} tokens, Max output: {max_tokens}")

            # Format as chat
            messages = [{"role": "user", "content": prompt}]
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            temp = temperature if temperature is not None else self.config.temperature

            # Generate
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            if prompt_text in response:
                response = response.split(prompt_text)[-1].strip()
            else:
                # Try to find assistant response marker
                markers = ["assistant\n", "[/INST]", "<|assistant|>"]
                for marker in markers:
                    if marker in response:
                        response = response.split(marker)[-1].strip()
                        break

            output_tokens = len(self.tokenizer.encode(response))

            if self.config.log_token_counts:
                print(f"    [LLM] Output: {output_tokens} tokens")

            # Update stats
            self.token_stats["total_input_tokens"] += input_tokens
            self.token_stats["total_output_tokens"] += output_tokens
            self.token_stats["total_calls"] += 1

            return response

        except Exception as e:
            print(f"    [LLM ERROR] {e}")
            return self._generate_fallback(prompt)

    def _generate_fallback(self, prompt: str) -> str:
        """Fallback generation when model is not available"""
        if "video summary" in prompt.lower() or "Video Information" in prompt:
            return """KEY POINT:
This video analysis requires the LLM model which is not currently loaded.

DETAILED SUMMARY:
The full video summary requires GPU resources with the Llama 3.1-8B model loaded. To enable full analysis, run this on Google Colab with GPU runtime and call model_manager.load_model(). Basic video metadata and engagement metrics are still available."""

        elif "reaction" in prompt.lower() or "Comments:" in prompt:
            return """KEY POINT:
Comment analysis requires the LLM model which is currently not available.

DETAILED SUMMARY:
The detailed audience reaction analysis requires GPU resources for the language model. Basic engagement metrics and top comments are still available. For full AI-powered sentiment analysis, deploy on Colab with GPU.

SENTIMENT BREAKDOWN:
Positive: N/A%  Negative: N/A%  Neutral: N/A%"""

        return "[LLM not loaded - showing placeholder]"

    def get_token_efficiency_report(self) -> str:
        """Generate token efficiency report"""
        stats = self.token_stats
        total_calls = stats["total_calls"]

        if total_calls == 0:
            return "Token Stats: No LLM calls made yet"

        total_tokens = stats["total_input_tokens"] + stats["total_output_tokens"]
        avg_per_call = total_tokens / total_calls
        efficiency = (stats['tokens_saved'] / total_tokens * 100) if total_tokens > 0 else 0

        return f"""Token Efficiency Report:
  Total Calls: {total_calls}
  Input Tokens: {stats['total_input_tokens']:,}
  Output Tokens: {stats['total_output_tokens']:,}
  Tokens Saved: {stats['tokens_saved']:,}
  Avg Tokens/Call: {avg_per_call:.0f}
  Efficiency Gain: {efficiency:.1f}%"""

    def get_status(self) -> Dict:
        """Get current model status for UI display"""
        return {
            "loaded": self._loaded,
            "model_name": self.config.model_name,
            "device": self.device,
            "error": self._load_error,
            "mode": "FULL LLM" if self._loaded else "FALLBACK",
            "gpu_memory": (
                f"{torch.cuda.memory_allocated(0)/1e9:.2f} GB"
                if self._loaded and self.device == "cuda"
                else "N/A"
            ),
            "total_calls": self.token_stats["total_calls"]
        }


def cleanup_model(model_manager: ModelManager):
    """Release model from GPU memory"""
    if model_manager.model is not None:
        del model_manager.model
        model_manager.model = None
    if model_manager.tokenizer is not None:
        del model_manager.tokenizer
        model_manager.tokenizer = None
    model_manager._loaded = False

    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ GPU memory released")


# =============================================================================
# YouTube API Client
# =============================================================================

class YouTubeAPIClient:
    """YouTube API client for fetching video data"""

    def __init__(self, api_key: str):
        if not YOUTUBE_API_AVAILABLE:
            raise RuntimeError("google-api-python-client not installed")
        if not api_key:
            raise ValueError("YouTube API key is required")
        self.youtube = build('youtube', 'v3', developerKey=api_key)

    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/|v\/|youtu.be\/)([0-9A-Za-z_-]{11})',
            r'^([0-9A-Za-z_-]{11})$'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def get_video_info(self, video_id: str) -> Optional[Dict]:
        try:
            request = self.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=video_id
            )
            response = request.execute()

            if not response['items']:
                return None

            item = response['items'][0]
            snippet = item['snippet']
            statistics = item['statistics']
            content_details = item['contentDetails']

            duration = isodate.parse_duration(content_details['duration']).total_seconds()

            thumbnails = snippet.get('thumbnails', {})
            thumbnail_url = (
                thumbnails.get('maxres', {}).get('url') or
                thumbnails.get('standard', {}).get('url') or
                thumbnails.get('high', {}).get('url') or
                thumbnails.get('medium', {}).get('url') or
                thumbnails.get('default', {}).get('url', '')
            )

            return {
                'video_info': {
                    'video_id': video_id,
                    'title': snippet['title'],
                    'description': snippet['description'],
                    'channel_title': snippet['channelTitle'],
                    'published_at': snippet['publishedAt'],
                    'category_id': snippet['categoryId'],
                    'view_count': int(statistics.get('viewCount', 0)),
                    'like_count': int(statistics.get('likeCount', 0)),
                    'comment_count': int(statistics.get('commentCount', 0)),
                    'duration': int(duration),
                    'thumbnail_url': thumbnail_url
                },
                'comments': []
            }

        except HttpError as e:
            print(f"API Error: {e}")
            return None

    def get_comments(self, video_id: str, max_results: int = 100) -> List[Dict]:
        try:
            comments = []
            request = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(max_results, 100),
                order='relevance'
            )

            while request and len(comments) < max_results:
                response = request.execute()

                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    comments.append({
                        'author': comment['authorDisplayName'],
                        'text': comment['textDisplay'],
                        'like_count': comment['likeCount'],
                        'published_at': comment['publishedAt']
                    })

                if 'nextPageToken' in response and len(comments) < max_results:
                    request = self.youtube.commentThreads().list(
                        part='snippet',
                        videoId=video_id,
                        pageToken=response['nextPageToken'],
                        maxResults=min(max_results - len(comments), 100),
                        order='relevance'
                    )
                else:
                    break

            return comments

        except HttpError as e:
            error_content = e.content.decode() if hasattr(e, 'content') else str(e)
            if 'commentsDisabled' in error_content:
                print("Comments are disabled for this video")
            else:
                print(f"Comments API Error: {e}")
            return []


# =============================================================================
# Report Generator
# =============================================================================

class ReportGenerator:
    """
    Report generator with Quality Gate and LLM integration.
    
    Colab LLM Demo: Generates real AI summaries when model is loaded.
    """

    CATEGORY_MAPPING = {
        '1': 'Film & Animation', '2': 'Autos & Vehicles',
        '10': 'Music', '15': 'Pets & Animals', '17': 'Sports',
        '19': 'Travel & Events', '20': 'Gaming', '22': 'People & Blogs',
        '23': 'Comedy', '24': 'Entertainment', '25': 'News & Politics',
        '26': 'HowTo & Style', '27': 'Education', '28': 'Science & Tech',
    }

    def __init__(self, config: PipelineConfig, model_manager: Optional[ModelManager] = None):
        self.config = config
        self.model = model_manager
        self.quality_gate = QualityGate(config)

    def generate_video_summary(self, video_data: Dict) -> Tuple[str, str]:
        """Generate video summary with quality gate"""
        if self.config.enable_detailed_logging:
            print("  📝 Generating video summary...")

        video_info = video_data.get('video_info', {})
        title = video_info.get('title', 'N/A')
        description = video_info.get('description', 'N/A')
        category_id = video_info.get('category_id', 'Unknown')
        duration = video_info.get('duration', 'Unknown')

        if self.config.remove_urls:
            description = TextPreprocessor.remove_urls(description)
        if len(description) > self.config.max_description_length:
            description = description[:self.config.max_description_length] + "..."

        detected_lang = TextPreprocessor.detect_language(f"{title} {description}")

        if self.model:
            prompt = PromptTemplates.get_video_summary_prompt(
                title, description, str(category_id), str(duration),
                self.config.output_language
            )

            # Quality gate with regeneration
            summary = None
            attempts = 0
            max_attempts = self.config.max_regeneration_attempts + 1

            while attempts < max_attempts:
                temp = self.config.quality_gate_temperature if attempts > 0 else self.config.temperature
                summary = self.model.generate(prompt, temperature=temp)

                is_valid, failure_reasons = self.quality_gate.validate_summary(summary, "video")

                if is_valid:
                    if attempts > 0:
                        self.quality_gate.validation_stats["regenerated"] += 1
                        if self.config.enable_detailed_logging:
                            print(f"    ✅ Quality passed after {attempts} regeneration(s)")
                    break
                else:
                    attempts += 1
                    if attempts < max_attempts:
                        if self.config.enable_detailed_logging:
                            print(f"    ⚠️  Quality check failed: {', '.join(failure_reasons)}")
                    else:
                        self.quality_gate.validation_stats["failed_final"] += 1
        else:
            summary = f"""KEY POINT:
"{title}" - Video analysis requires LLM support.

DETAILED SUMMARY:
This video is from the {self.CATEGORY_MAPPING.get(str(category_id), 'Unknown')} category. Full content analysis requires the LLM model to be loaded."""

        return summary, detected_lang

    def generate_reaction_summary(self, comments: List[str]) -> Tuple[str, Dict[str, int]]:
        """Generate reaction summary with quality gate"""
        if self.config.enable_detailed_logging:
            print("  💬 Generating reaction summary...")

        if not comments:
            return "No comments available for analysis.", {}

        processed_comments = []
        for comment in comments[:self.config.max_comments_to_process]:
            cleaned = TextPreprocessor.clean_text(comment, remove_urls=self.config.remove_urls)
            if len(cleaned) >= self.config.min_comment_length:
                processed_comments.append(cleaned)

        if not processed_comments:
            return "No valid comments after filtering.", {}

        lang_dist = TextPreprocessor.get_language_distribution(processed_comments)

        if self.model:
            prompt = PromptTemplates.get_reaction_summary_prompt(
                processed_comments,
                self.config.output_language
            )

            # Quality gate with regeneration
            summary = None
            attempts = 0
            max_attempts = self.config.max_regeneration_attempts + 1

            while attempts < max_attempts:
                temp = self.config.quality_gate_temperature if attempts > 0 else self.config.temperature
                summary = self.model.generate(prompt, temperature=temp)

                is_valid, failure_reasons = self.quality_gate.validate_summary(summary, "reaction")

                if is_valid:
                    if attempts > 0:
                        self.quality_gate.validation_stats["regenerated"] += 1
                    break
                else:
                    attempts += 1
                    if attempts >= max_attempts:
                        self.quality_gate.validation_stats["failed_final"] += 1
        else:
            # Simple estimation without LLM
            pos_words = ['love', 'great', 'amazing', 'good', 'best', 'awesome', '좋아', '최고', '대박']
            neg_words = ['bad', 'hate', 'worst', 'terrible', 'boring', '싫어', '별로', '최악']

            pos_count = sum(1 for c in processed_comments if any(w in c.lower() for w in pos_words))
            neg_count = sum(1 for c in processed_comments if any(w in c.lower() for w in neg_words))
            total = len(processed_comments)
            neu_count = total - pos_count - neg_count

            summary = f"""KEY POINT:
Based on {total} comments, audience shows mixed reactions.

DETAILED SUMMARY:
Basic sentiment analysis without LLM. {len(lang_dist)} language(s) detected.

SENTIMENT BREAKDOWN:
Positive: {pos_count/total*100:.0f}%  Negative: {neg_count/total*100:.0f}%  Neutral: {neu_count/total*100:.0f}%"""

        return summary, lang_dist

    def calculate_metrics(self, video_data: Dict) -> Dict[str, float]:
        video_info = video_data.get('video_info', {})
        views = video_info.get('view_count', 0)
        likes = video_info.get('like_count', 0)
        comments = video_info.get('comment_count', 0)

        return {
            'engagement_rate': ((likes + comments) / views * 100) if views > 0 else 0,
            'like_rate': (likes / views * 100) if views > 0 else 0,
            'comment_rate': (comments / views * 100) if views > 0 else 0
        }

    def format_report(self, video_data: Dict, video_summary: str,
                      reaction_summary: str, metrics: Dict,
                      video_lang: str, comment_langs: Dict,
                      llm_status: str = "Unknown") -> str:
        """Format the final markdown report"""
        video_info = video_data.get('video_info', {})

        category_id = str(video_info.get('category_id', 'Unknown'))
        category_name = self.CATEGORY_MAPPING.get(category_id, f'Unknown ({category_id})')

        top_comments = []
        for i, comment in enumerate(video_data.get('comments', [])[:5], 1):
            author = comment.get('author', 'Unknown')
            text = comment.get('text', '')
            likes = comment.get('like_count', 0)
            text_preview = text[:150] + "..." if len(text) > 150 else text
            top_comments.append(f"{i}. **@{author}** ({likes} likes): {text_preview}")

        top_comments_section = "\n\n".join(top_comments) if top_comments else "No comments available."

        if comment_langs:
            sorted_langs = sorted(comment_langs.items(), key=lambda x: x[1], reverse=True)
            comment_lang_str = ", ".join([f"{lang} ({count})" for lang, count in sorted_langs])
        else:
            comment_lang_str = "No comments analyzed"

        total_comments = video_info.get('comment_count', 0)
        analyzed_comments = len(video_data.get('comments', []))
        analysis_rate = (analyzed_comments / total_comments * 100) if total_comments > 0 else 0

        duration_sec = video_info.get('duration', 0)
        duration_str = f"{duration_sec // 60}:{duration_sec % 60:02d}" if duration_sec else "N/A"

        report = f"""# YouTube Video Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**LLM Status**: {llm_status}
**Pipeline Version**: 2.0 (Colab LLM Edition)

---

## Video Information

- **Title**: {video_info.get('title', 'N/A')}
- **Channel**: {video_info.get('channel_title', 'N/A')}
- **Category**: {category_name}
- **Published**: {video_info.get('published_at', 'N/A')[:10] if video_info.get('published_at') else 'N/A'}
- **Duration**: {duration_str}
- **Video ID**: `{video_info.get('video_id', 'N/A')}`
- **URL**: https://www.youtube.com/watch?v={video_info.get('video_id', 'N/A')}

### Detected Languages

- **Video content**: {video_lang}
- **Comments**: {comment_lang_str}
- **Report language**: {self.config.output_language}

---

## Engagement Metrics

| Metric | Value |
|--------|-------|
| Views | {video_info.get('view_count', 0):,} |
| Likes | {video_info.get('like_count', 0):,} |
| Comments | {video_info.get('comment_count', 0):,} |
| Engagement Rate | {metrics.get('engagement_rate', 0):.3f}% |
| Like Rate | {metrics.get('like_rate', 0):.3f}% |
| Comment Rate | {metrics.get('comment_rate', 0):.4f}% |

---

## Video Summary

{video_summary}

---

## Audience Reaction Summary

**Comments Analyzed**: {analyzed_comments} / {total_comments} ({analysis_rate:.1f}%)

{reaction_summary}

---

## Top Comments

{top_comments_section}

---

## Technical Notes

- LLM Status: {llm_status}
- Quality Gate: {'Enabled' if self.config.enable_quality_gate else 'Disabled'}
- Output Language: {self.config.output_language}

---

*Generated by YouTube Report Generator - Colab LLM Edition*
"""

        return report

    def generate_full_report(self, video_data: Dict) -> str:
        """Generate complete report"""
        video_info = video_data.get('video_info', {})
        
        if self.config.enable_detailed_logging:
            print(f"\n📊 Processing: {video_info.get('title', 'Unknown')[:50]}...")

        # Determine LLM status
        if self.model and self.model._loaded:
            llm_status = f"✅ FULL ({self.model.config.model_name} on {self.model.device.upper()})"
        else:
            llm_status = "⚠️ FALLBACK (LLM not loaded)"

        video_summary, video_lang = self.generate_video_summary(video_data)

        comments = [c.get('text', '') for c in video_data.get('comments', [])]
        reaction_summary, comment_langs = self.generate_reaction_summary(comments)

        metrics = self.calculate_metrics(video_data)

        report = self.format_report(
            video_data,
            video_summary,
            reaction_summary,
            metrics,
            video_lang,
            comment_langs,
            llm_status
        )

        if self.config.enable_detailed_logging:
            print("  ✅ Report generated!")

        return report


# =============================================================================
# High-Level API Functions
# =============================================================================

def generate_report_from_url(video_url: str, api_key: str,
                              config: Optional[PipelineConfig] = None,
                              model_manager: Optional[ModelManager] = None,
                              include_comments: bool = True) -> str:
    """
    Generate full report from YouTube URL.
    
    Colab LLM Demo: Pass a loaded ModelManager for full AI analysis.
    """
    if not YOUTUBE_API_AVAILABLE:
        raise RuntimeError("YouTube API dependencies not installed")

    if config is None:
        config = PipelineConfig()

    api_client = YouTubeAPIClient(api_key)
    video_id = api_client.extract_video_id(video_url)

    if not video_id:
        raise ValueError("Invalid YouTube URL")

    video_data = api_client.get_video_info(video_id)
    if not video_data:
        raise ValueError("Could not fetch video information")

    if include_comments:
        comments = api_client.get_comments(video_id, config.max_comments_to_process)
        video_data['comments'] = comments

    report_gen = ReportGenerator(config, model_manager)
    report = report_gen.generate_full_report(video_data)

    return report


# =============================================================================
# Dependency Check
# =============================================================================

def check_dependencies() -> Dict[str, bool]:
    """Check availability of all dependencies"""
    gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
    
    return {
        'torch': TORCH_AVAILABLE,
        'transformers': TRANSFORMERS_AVAILABLE,
        'langdetect': LANGDETECT_AVAILABLE,
        'youtube_api': YOUTUBE_API_AVAILABLE,
        'gpu_available': gpu_available,
        'gpu_name': torch.cuda.get_device_name(0) if gpu_available else "N/A"
    }


# =============================================================================
# Module Info
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("YouTube Report Generator Pipeline (Colab LLM Edition)")
    print("=" * 60)
    
    deps = check_dependencies()
    for dep, available in deps.items():
        if dep == 'gpu_name':
            print(f"  GPU: {available}")
        else:
            status = "✅" if available else "❌"
            print(f"  {status} {dep}")
    
    print("\nUsage:")
    print("  from pipeline import PipelineConfig, ModelManager, ReportGenerator")
    print("  config = PipelineConfig()")
    print("  mm = ModelManager(config)")
    print("  mm.load_model()  # Load Llama 3.1-8B")
    print("=" * 60)
