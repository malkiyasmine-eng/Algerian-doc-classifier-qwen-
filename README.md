# Algerian Document Classifier (4-bit Quantized)

Zero-shot classification of Algerian corporate documents using **Qwen2.5-VL-7B-Instruct**
loaded in **4-bit NF4 quantization** (bitsandbytes) — runs on a single Colab T4 GPU (15 GB VRAM).
Exposed via a FastAPI REST API, tunneled publicly with **ngrok**.

---

## Table of Contents

- [Supported Document Classes](#supported-document-classes)
- [Architecture Overview](#architecture-overview)
- [Feasibility on Google Colab](#feasibility-on-google-colab)
- [Google Colab Setup (Step-by-Step)](#google-colab-setup-step-by-step)
- [Local Installation](#local-installation)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Running the Test Suite](#running-the-test-suite)
- [GPU Memory Guide](#gpu-memory-guide)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

---

## Supported Document Classes

| # | Label | Description |
|---|-------|-------------|
| 1 | `NIF_certificate` | Attestation d'immatriculation fiscale |
| 2 | `NIS_certificate` | Numéro d'identification statistique |
| 3 | `certificat_existence` | Certificat d'existence (with diagonal strokes) |
| 4 | `tax_declaration_form` | Déclaration d'existence form |
| 5 | `residence_certificate` | بطاقة إقامة — Arabic residence document |
| 6 | `legal_contract_front` | Notarial contract cover page |
| 7 | `legal_contract_inside` | Notarial contract body pages |
| 8 | `balance_sheet` | Financial statements (bilan, compte de résultat, etc.) |
| 9 | `RC_front` | Registre de commerce — front page |
| 10 | `RC_inside_activities` | Registre de commerce — activities page |
| 11 | `RC_inside_2` | Registre de commerce — penalties page |
| 12 | `driving_license_front` | Driving license — front side only |
| 13 | `driving_license_back` | Driving license — back side only |
| 14 | `driving_license_frontback` | Driving license — both sides in one image |

---

## Architecture Overview

```
Client (HTTP / curl / Python)
    │
    ▼
app.py  (FastAPI)
    │  lifespan → load_model() on startup
    │  GET  /health   → model status + device info
    │  POST /classify → input validation → inference
    │
    ▼
inference.py
    │  _get_bnb_config()         → 4-bit NF4 BitsAndBytesConfig
    │  load_model()              → loads quantized weights + processor
    │  classify_image(bytes)
    │      └─ _classify_document(PIL image)
    │              └─ processor → model.generate() → decode
    │      └─ _match_label(raw_output)
    │              └─ exact / case-insensitive / fuzzy match
    └─ returns {"label": str, "confidence": float}
```

**4-bit quantization reduces VRAM from ~14 GB → ~5 GB**, making the 7B model
comfortably fit on a Colab T4 (15 GB) with ~9 GB headroom for activations.

---

## Feasibility on Google Colab

| Colab Tier | GPU | VRAM | 7B 4-bit | Notes |
|---|---|---|---|---|
| **Free** | T4 | 15 GB | ✅ | ~5 GB model load, ~8 GB peak inference |
| **Colab Pro** | T4 / V100 | 15–16 GB | ✅ | Same — more session time |
| **Colab Pro+** | A100 | 40 GB | ✅✅ | Ideal — loads fast, no OOM risk |

**Limitations to be aware of:**
- Free Colab sessions disconnect after ~12 hours (90 min idle).
- The ngrok public URL changes each session — update your clients.
- First run downloads ~15 GB of model weights. Subsequent sessions are faster if you mount Google Drive for caching.
- Inference speed: ~3–8 seconds per image on T4 (4-bit is slower than fp16 due to dequantization overhead, but uses much less memory).
- Free ngrok has a 1 req/s rate limit and 40 connections/min. Upgrade to ngrok Pro for production use.

---

## Google Colab Setup (Step-by-Step)

### Step 0 — Open Colab with a GPU

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Create a new notebook
3. **Runtime → Change runtime type → T4 GPU → Save**
4. Verify GPU: run `!nvidia-smi` — you should see a T4 or similar.

---

### Step 1 — Install dependencies

Paste this into a Colab cell and run it:

```python
# Cell 1 — Install all dependencies
!pip install -q fastapi uvicorn[standard] python-multipart Pillow requests
!pip install -q bitsandbytes>=0.43.0 accelerate
!pip install -q qwen-vl-utils
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install -q pyngrok

# Verify torch + CUDA
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

Expected output:
```
PyTorch: 2.x.x+cu121
CUDA available: True
GPU: Tesla T4
VRAM: 15.0 GB
```

---

### Step 2 — Upload project files

**Option A — Upload manually (simplest):**

```python
# Cell 2 — Upload files via Colab UI
from google.colab import files

print("Upload app.py, inference.py when prompted")
uploaded = files.upload()
# Upload app.py, then inference.py
```

**Option B — Write files directly (no upload needed):**

Paste the full contents of `app.py` and `inference.py` into two separate cells
using `%%writefile`:

```python
# Cell 2a
%%writefile app.py
# <paste the entire content of app.py here>
```

```python
# Cell 2b
%%writefile inference.py
# <paste the entire content of inference.py here>
```

---

### Step 3 — Set up ngrok

1. Sign up for a free ngrok account at [https://ngrok.com](https://ngrok.com)
2. Go to **Dashboard → Your Authtoken** and copy your token

```python
# Cell 3 — Configure ngrok
from pyngrok import ngrok, conf

NGROK_TOKEN = "YOUR_NGROK_AUTHTOKEN_HERE"   # ← paste your token
conf.get_default().auth_token = NGROK_TOKEN
ngrok.set_auth_token(NGROK_TOKEN)
print("✅ ngrok configured")
```

---

### Step 4 — Start the FastAPI server (background)

```python
# Cell 4 — Launch uvicorn in background
import subprocess, time, threading

def run_server():
    subprocess.run(
        ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

thread = threading.Thread(target=run_server, daemon=True)
thread.start()
print("⏳ Server starting — waiting for model to load (this takes 1–3 min on first run)...")
time.sleep(90)   # give the model time to download and load
print("✅ Server should be up. Proceeding to ngrok tunnel.")
```

> **Note:** On first run the model downloads ~15 GB from HuggingFace. Increase
> `time.sleep(90)` to `time.sleep(180)` if you see 502 errors from ngrok.

---

### Step 5 — Expose API via ngrok

```python
# Cell 5 — Open ngrok tunnel
from pyngrok import ngrok

public_url = ngrok.connect(8000).public_url
print(f"\n🌍 Public API URL: {public_url}")
print(f"   Swagger UI    : {public_url}/docs")
print(f"   Health check  : {public_url}/health")
print(f"\nShare this URL to call the API from anywhere.")
```

Expected output:
```
🌍 Public API URL: https://abc123.ngrok-free.app
   Swagger UI    : https://abc123.ngrok-free.app/docs
   Health check  : https://abc123.ngrok-free.app/health
```

---

### Step 6 — Test the endpoint

**Health check (confirms model is loaded):**

```python
# Cell 6a — Health check
import requests

BASE_URL = public_url   # from Cell 5

resp = requests.get(f"{BASE_URL}/health")
print(resp.json())
```

Expected:
```json
{
  "status": "ok",
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "quantization": "4-bit NF4 (bitsandbytes)",
  "device": "cuda (1 GPU)",
  "classes": ["NIF_certificate", ...],
  "n_classes": 14
}
```

**Classify a document image:**

```python
# Cell 6b — Classify a document
import requests

BASE_URL = public_url   # from Cell 5

image_path = "document.jpg"   # path to your test image in Colab

with open(image_path, "rb") as f:
    resp = requests.post(
        f"{BASE_URL}/classify",
        files={"file": (image_path, f, "image/jpeg")},
    )

print(resp.json())
# {"label": "NIF_certificate", "confidence": 1.0}
```

**From your local machine (using curl):**

```bash
curl -X POST https://abc123.ngrok-free.app/classify \
  -F "file=@/path/to/document.jpg" \
  | python -m json.tool
```

---

### Optional: Cache model weights in Google Drive

Speeds up subsequent sessions by avoiding re-download:

```python
# Cell 0 (run BEFORE Cell 1) — Mount Drive and set HF cache
from google.colab import drive
drive.mount("/content/drive")

import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
print("✅ HuggingFace cache pointed to Google Drive")
```

---

## Local Installation

### Prerequisites
- Python 3.10+
- CUDA GPU with ≥ 8 GB VRAM (4-bit 7B peaks at ~8 GB)
- CUDA toolkit 11.8, 12.1, or 12.4

### Install

```bash
git clone <your-repo-url>
cd algerian-doc-classifier
python -m venv .venv
source .venv/bin/activate
# Edit requirements.txt: replace cu121 with your CUDA version
pip install -r requirements.txt
```

### Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Configuration

Open `inference.py` to adjust these constants near the top:

```python
# Pixel budget — reduce if you hit OOM on T4
MIN_PIXELS = 128 * 28 * 28   # ~100 K pixels
MAX_PIXELS = 256 * 28 * 28   # ~200 K pixels  ← safe for T4

# Increase for A100 / high-res documents:
# MAX_PIXELS = 512 * 28 * 28   # ~400 K pixels
```

---

## API Reference

### `GET /health`

Returns server status and model metadata.

**Response:**
```json
{
  "status": "ok",
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "quantization": "4-bit NF4 (bitsandbytes)",
  "device": "cuda (1 GPU)",
  "classes": ["NIF_certificate", "..."],
  "n_classes": 14
}
```

---

### `POST /classify`

Classifies a document image.

**Request:** `multipart/form-data`, field `file`, max 10 MB (JPEG / PNG / WEBP / TIFF)

**Success response (HTTP 200):**
```json
{
  "label": "NIF_certificate",
  "confidence": 1.0
}
```

**Confidence score:**

| Score | Meaning |
|---|---|
| `1.0` | Exact match |
| `0.95` | Case-insensitive match |
| `0.4–0.94` | Fuzzy match |
| `0.0` | No match found |

**Error responses:**

| HTTP | Cause |
|---|---|
| `400` | Uploaded file is empty |
| `413` | File exceeds 10 MB |
| `415` | Unsupported MIME type |
| `422` | File is not a valid image |
| `500` | Model inference error / GPU OOM |

---

## GPU Memory Guide

| Setup | VRAM usage | Notes |
|---|---|---|
| 7B float16 (original) | ~14 GB | Doesn't fit on T4 |
| 7B bfloat16 (original) | ~14 GB | Doesn't fit on T4 |
| **7B 4-bit NF4 (this version)** | **~5 GB** | **Fits T4 comfortably** |
| 7B 4-bit + double quant | ~4.5 GB | Enabled by default |

Peak VRAM during inference (image tokenization + generation): ~8–10 GB on T4.

---

## Performance Optimization

### Increase pixel budget (for high-res documents on A100)

```python
# inference.py
MAX_PIXELS = 512 * 28 * 28   # ~400 K pixels
```

### Flash Attention 2 (Ampere+ GPUs — A100, RTX 30xx/40xx)

```bash
pip install flash-attn --no-build-isolation
```

Then in `load_model()`:
```python
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    attn_implementation="flash_attention_2",
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)
```

> Flash Attention 2 is **not compatible** with 4-bit quantization on T4 (no bf16 support).
> Only enable it on A100 or RTX 30xx/40xx.

### Torch compile (PyTorch 2.x, optional)

```python
# After model.eval() in load_model()
# NOTE: may be incompatible with bitsandbytes — test before enabling
# model = torch.compile(model)
```

---

## Troubleshooting

**`RuntimeError: No CUDA GPU detected`**
→ You forgot to enable GPU in Colab. Go to Runtime → Change runtime type → T4 GPU.

**`ModuleNotFoundError: bitsandbytes`**
→ Run `!pip install bitsandbytes>=0.43.0` and restart the runtime.

**`torch.cuda.OutOfMemoryError` (HTTP 500)**
→ Reduce `MAX_PIXELS` in `inference.py` to `128*28*28` and restart the server.
→ Or upload a smaller/lower-resolution image.

**ngrok `502 Bad Gateway` right after Cell 5**
→ The model is still loading. Wait another 60–90 seconds and retry.

**`HuggingFace download timeout`**
→ Set a mirror: `import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"` before loading.

**`Connection refused` when running test_api.py**
→ The server is not running. Make sure Cell 4 completed and uvicorn is up.

**Server seems up but `/health` returns `"status": "loading"`**
→ Model is still initializing. Weight loading takes 1–3 min on first run. Wait and retry.

---

## Project Structure

```
algerian-doc-classifier/
├── app.py            # FastAPI application — routes, validation, lifespan
├── inference.py      # Model loading (4-bit), inference pipeline, label matching
├── test_api.py       # API test suite (8 tests, no external fixtures needed)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```
