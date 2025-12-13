# 🤖 YouTube Report Generator (Colab + GPU + LLM Edition)

A web application that generates **AI-powered** comprehensive reports from YouTube videos using open-source LLMs.

> **This version is optimized for Google Colab with GPU runtime.**
> 
> **🆕 NEW: 승인 없이 바로 사용 가능한 모델 프리셋 지원!**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)
![LLM](https://img.shields.io/badge/LLM-Open_Source-green.svg)

## ✨ Features

- 🤖 **Real AI Summaries**: Uses open-source LLMs for video content analysis
- 🔓 **No Approval Required**: Mistral, Qwen, Gemma, Phi 등 바로 사용 가능
- 🔗 **URL Input**: Paste any YouTube URL to analyze
- 📺 **Video Preview**: See thumbnail, title, and metadata
- 📊 **Engagement Metrics**: Views, likes, comments, engagement rates
- 📝 **AI Video Summary**: LLM-generated summary with KEY POINT + DETAILED SUMMARY
- 💬 **AI Reaction Analysis**: LLM-analyzed audience sentiment and themes
- 🌐 **Public Demo URL**: Share via ngrok (Streamlit) or gradio.live (Gradio)
- ⚙️ **Tunable Parameters**: Adjust temperature, tokens, quality gate in UI

---

## 📦 Available Model Presets

### 🟢 No Approval Required (바로 사용 가능)

| Preset | Model | VRAM | 특징 |
|--------|-------|------|------|
| `mistral-7b` | Mistral-7B-Instruct-v0.3 | ~5GB | **추천** - 빠르고 품질 좋음 |
| `qwen2.5-7b` | Qwen2.5-7B-Instruct | ~5GB | **한국어 추천** - 다국어 우수 |
| `gemma2-9b` | Gemma-2-9B-it | ~6GB | 고품질, 약간 느림 |
| `phi3-mini` | Phi-3-mini-4k-instruct | ~3GB | 가볍고 빠름 |
| `phi3.5-mini` | Phi-3.5-mini-instruct | ~3.5GB | 최신, 성능 개선 |
| `tinyllama` | TinyLlama-1.1B-Chat | ~2.5GB | 매우 가벼움 |
| `stablelm-2` | StableLM-2-1.6B-Chat | ~3GB | 가볍고 안정적 |

### 🟡 Requires HF Approval (승인 필요)

| Preset | Model | 신청 링크 |
|--------|-------|----------|
| `llama3.1-8b` | Llama-3.1-8B-Instruct | [신청](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| `llama3.2-3b` | Llama-3.2-3B-Instruct | [신청](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |

---

## 🚀 Quick Start (Google Colab)

### Option 1: Gradio (Easiest - Built-in Public URL)

```python
# Cell 1: Install dependencies
!pip install gradio google-api-python-client isodate langdetect
!pip install transformers accelerate bitsandbytes

# Cell 2: Set API key
import os
os.environ['YOUTUBE_API_KEY'] = 'YOUR_YOUTUBE_API_KEY'

# Cell 3: Upload files
# Upload pipeline.py and app_gradio.py using Colab file browser

# Cell 4: (선택) 모델 확인
from pipeline import list_available_models
list_available_models()

# Cell 5: Run (automatic gradio.live URL!)
!python app_gradio.py
```

### 🔧 모델 변경하기 (Colab 셀에서)

```python
# Cell: 모델 변경 후 앱 실행
from pipeline import PipelineConfig, MODEL_PRESETS, list_available_models

# 사용 가능한 모델 목록 보기
list_available_models()

# 방법 1: 프리셋 사용 (추천)
config = PipelineConfig.from_preset('qwen2.5-7b')  # 한국어 추천
config = PipelineConfig.from_preset('mistral-7b')  # 영어 추천
config = PipelineConfig.from_preset('phi3-mini')   # 가벼운 모델

# 방법 2: 직접 모델 지정
config = PipelineConfig(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    use_4bit=True,
    max_new_tokens=512
)

# 방법 3: 프리셋 + 커스텀 설정
config = PipelineConfig.from_preset('qwen2.5-7b', max_new_tokens=256, temperature=0.5)

# 이후 ModelManager에 전달
from pipeline import ModelManager
model_manager = ModelManager(config)
model_manager.load_model()
```

### Gradio 앱에서 모델 변경

```python
# app_gradio.py 실행 전에 환경변수로 모델 지정
import os
os.environ['YOUTUBE_API_KEY'] = 'YOUR_KEY'
os.environ['MODEL_PRESET'] = 'qwen2.5-7b'  # 프리셋 이름
# 또는
os.environ['MODEL_NAME'] = 'microsoft/Phi-3-mini-4k-instruct'  # 직접 지정

!python app_gradio.py
```

---

### Option 2: Streamlit (with ngrok tunnel)

```python
# Cell 1: Install dependencies
!pip install streamlit google-api-python-client isodate langdetect pyngrok
!pip install transformers accelerate bitsandbytes

# Cell 2: Set API keys
import os
os.environ['YOUTUBE_API_KEY'] = 'YOUR_YOUTUBE_API_KEY'
os.environ['NGROK_AUTH_TOKEN'] = 'YOUR_NGROK_TOKEN'  # Get free token from ngrok.com

# Cell 3: Upload files
# Upload pipeline.py and app_streamlit.py using Colab file browser

# Cell 4: Run Streamlit with ngrok
import threading
from pyngrok import ngrok

ngrok.set_auth_token(os.environ.get('NGROK_AUTH_TOKEN', ''))

def run_streamlit():
    import os
    os.system('streamlit run app_streamlit.py --server.port 8501 --server.headless true')

thread = threading.Thread(target=run_streamlit)
thread.start()

import time
time.sleep(10)  # Wait for Streamlit to start

public_url = ngrok.connect(8501)
print(f"\n🌐 PUBLIC URL: {public_url}")
print("Share this URL for demo access!")
```

---

## 📁 Project Structure

```
youtube_report_webapp_colab/
├── app_streamlit.py    # Streamlit web app (use with ngrok)
├── app_gradio.py       # Gradio web app (built-in share=True)
├── pipeline.py         # Core pipeline with LLM support
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

---

## 🔧 LLM Modes

The application has two modes:

| Mode | Condition | Behavior |
|------|-----------|----------|
| **FULL LLM** | GPU available + model loads | Real AI-generated summaries using Llama 3.1-8B |
| **FALLBACK** | No GPU or load fails | Placeholder summaries with basic stats |

The UI clearly shows which mode is active:
- ✅ `LLM Status: FULL (Llama 3.1-8B on CUDA)` - Real AI analysis
- ⚠️ `LLM Status: FALLBACK` - Placeholder mode

---

## ⚙️ Configurable Parameters

These can be adjusted in the sidebar (Streamlit) or accordion (Gradio):

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Report Language | English | 7 options | Output language for summaries |
| Max Comments | 100 | 10-200 | Comments to analyze |
| Quality Gate | On | Toggle | Validate and regenerate low-quality outputs |
| Min Summary Length | 100 | 50-400 | Minimum chars for valid summary |
| Max New Tokens | 512 | 128-1024 | Maximum LLM output tokens |
| Temperature | 0.7 | 0.1-1.5 | LLM creativity (lower=focused) |

---

## 🔑 Getting API Keys

### YouTube Data API Key
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create project → Enable YouTube Data API v3
3. Create Credentials → API Key
4. Copy and set: `os.environ['YOUTUBE_API_KEY'] = 'your-key'`

### ngrok Auth Token (for Streamlit only)
1. Sign up at [ngrok.com](https://ngrok.com/) (free)
2. Go to Dashboard → Your Authtoken
3. Copy and set: `os.environ['NGROK_AUTH_TOKEN'] = 'your-token'`

---

## 📊 Example Output

The generated report includes:

```markdown
# YouTube Video Report

**Generated**: 2024-01-15 14:30:00
**LLM Status**: ✅ FULL (meta-llama/Llama-3.1-8B-Instruct on CUDA)

## Video Information
- Title: Amazing Video Title
- Channel: Cool Channel
- Duration: 12:34
...

## Video Summary

KEY POINT:
This video explores the fascinating world of...

DETAILED SUMMARY:
The creator takes viewers through an in-depth...

## Audience Reaction Summary

KEY POINT:
Viewers overwhelmingly praised the video's...

DETAILED SUMMARY:
The majority of comments (estimated 75%) express...

SENTIMENT BREAKDOWN:
Positive: 75%  Negative: 5%  Neutral: 20%
```

---

## 🛠️ Troubleshooting

### "CUDA out of memory"
- Use 4-bit quantization (default: `use_4bit=True`)
- Reduce `max_new_tokens` to 256-384
- Restart Colab runtime to free GPU memory

### "Model loading takes forever"
- First load downloads ~4GB model files
- Subsequent runs are faster due to caching
- Ensure GPU runtime is selected (Runtime → Change runtime type → GPU)

### "LLM Status: FALLBACK"
- Check GPU: `!nvidia-smi`
- Check CUDA: `import torch; print(torch.cuda.is_available())`
- Install bitsandbytes: `!pip install bitsandbytes`

### "YouTube API Error"
- Verify API key is set correctly
- Check API quota (default: 10,000 units/day)
- Some videos have comments disabled

---

## 📝 Technical Notes

- **Model**: `meta-llama/Llama-3.1-8B-Instruct` with 4-bit quantization
- **GPU Memory**: ~5GB with 4-bit quantization
- **Inference Time**: ~10-30 seconds per summary (depends on GPU)
- **Quality Gate**: Auto-regenerates outputs that fail validation

---

## 🆚 Comparison: Colab LLM vs. Web Edition

| Feature | Colab LLM Edition | Web Edition |
|---------|-------------------|-------------|
| LLM Support | ✅ Full (Llama 3.1-8B) | ⚠️ Fallback only |
| GPU Required | ✅ Yes | ❌ No |
| Deployment | Temporary (ngrok/gradio.live) | Permanent (Streamlit Cloud) |
| Best For | Demos, research | Portfolio, always-on |

---

## 📄 License

MIT License

## 🙏 Acknowledgments

- Original pipeline: `merge_notebooks.ipynb` / `merge_notebooks.py`
- [Streamlit](https://streamlit.io/) & [Gradio](https://gradio.app/)
- [Hugging Face Transformers](https://huggingface.co/transformers)
- [Meta Llama](https://ai.meta.com/llama/)

---

**🤖 Powered by Llama 3.1-8B | Made for Colab GPU Demos**
