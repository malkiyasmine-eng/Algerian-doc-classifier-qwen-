# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Algerian Document Classifier — FastAPI app
#
# Run  : uvicorn app:app --host 0.0.0.0 --port 8000
# Docs : http://127.0.0.1:8000/docs
# ─────────────────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import UnidentifiedImageError
import inference

# ── Allowed image MIME types ──────────────────────────────────────────────────
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/tiff",
}
MAX_FILE_SIZE_MB    = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# ─────────────────────────────────────────────────────────────────────────────
# lifespan : loads the model ONCE when the server starts.
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("🚀 Starting Algerian Document Classifier API")
    print(f"   Model : {inference.MODEL_ID}  (4-bit quantized)")
    print("   Loading model — please wait...")
    print("=" * 60)
    try:
        inference.load_model()
    except RuntimeError as e:
        print(f"❌ FATAL: Model failed to load — {e}")
        raise SystemExit(1)
    print("=" * 60)
    print("✅ Model ready. API is live.")
    print("   Swagger UI : http://127.0.0.1:8000/docs")
    print("=" * 60)
    yield
    print("🛑 Server shutting down. Goodbye!")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app instance
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Algerian Document Classifier",
    description=(
        "Zero-shot classification of Algerian corporate documents "
        "using Qwen2.5-VL-7B-Instruct (4-bit quantized) on GPU.\n\n"
        "**Categories:** NIF_certificate, NIS_certificate, certificat_existence, "
        "tax_declaration_form, residence_certificate, legal_contract_front, "
        "legal_contract_inside, balance_sheet, RC_front, RC_inside_activities, "
        "RC_inside_2, driving_license_front, driving_license_back, driving_license_frontback"
    ),
    version="3.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", summary="Health check")
def health_check():
    """
    Returns server status and model information.
    - `status: "ok"` — model is loaded and ready for inference.
    - `status: "loading"` — model is still initializing.
    """
    model_ready = inference.model is not None

    if torch.cuda.is_available():
        device_str = f"cuda ({torch.cuda.device_count()} GPU)"
    else:
        device_str = "cpu"

    return {
        "status"      : "ok" if model_ready else "loading",
        "model"       : inference.MODEL_ID,
        "quantization": "4-bit NF4 (bitsandbytes)",
        "device"      : device_str,
        "classes"     : inference.CLASS_NAMES,
        "n_classes"   : len(inference.CLASS_NAMES),
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /classify
# ─────────────────────────────────────────────────────────────────────────────
@app.post(
    "/classify",
    summary="Classify a document image",
    response_description="Predicted label and confidence score",
)
async def classify(file: UploadFile = File(..., description="Document image (JPEG, PNG, WEBP, TIFF)")):
    """
    Upload a scanned document image and receive its predicted category.

    **Input:** multipart image file (JPEG, PNG, WEBP or TIFF), max 10 MB

    **Output:**
    ```json
    {
      "label": "NIF_certificate",
      "confidence": 1.0
    }
    ```

    **Confidence score:**
    - `1.0`  → exact match
    - `0.95` → case-insensitive match
    - `0.4–0.94` → fuzzy match
    - `0.0`  → no match found

    **Error codes:**
    - `400` → empty file
    - `413` → file too large (> 10 MB)
    - `415` → unsupported file type
    - `422` → file is not a valid image
    - `500` → model inference error (includes GPU OOM)
    """

    # ── Guard 1: MIME type ────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: '{file.content_type}'. "
                f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}"
            ),
        )

    image_bytes = await file.read()

    # ── Guard 2: empty file ───────────────────────────────────────────────────
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Guard 3: file size ────────────────────────────────────────────────────
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({len(image_bytes) / 1e6:.1f} MB). "
                f"Max allowed: {MAX_FILE_SIZE_MB} MB."
            ),
        )

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        result = inference.classify_image(image_bytes)

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=422,
            detail=(
                "File could not be decoded as an image. "
                "Ensure it is a valid JPEG, PNG, WEBP, or TIFF file."
            ),
        )

    except MemoryError:
        raise HTTPException(
            status_code=500,
            detail="Out of memory during inference. Try a smaller image.",
        )

    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference error: {str(e)}",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}",
        )

    return JSONResponse(content=result)
