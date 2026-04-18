# inference.py
# ─────────────────────────────────────────────────────────────────────────────
# Algerian Document Classifier — GPU inference (4-bit quantized)
# Model : Qwen/Qwen2.5-VL-7B-Instruct  (4-bit via bitsandbytes)
#
# Run   : uvicorn app:app --host 0.0.0.0 --port 8000
# ─────────────────────────────────────────────────────────────────────────────

import io
import difflib
import torch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_SIZE = "7B"
MODEL_ID   = "Qwen/Qwen2.5-VL-7B-Instruct"

# ── Pixel budget for the vision processor ─────────────────────────────────────
# Lower max_pixels = less VRAM per image = safer on Colab (15 GB T4).
# 256 * 28 * 28 = ~200 K pixels ≈ ~450×450 effective resolution — more than
# enough for document classification.  Raise to 512*28*28 if you have A100.
MIN_PIXELS = 128 * 28 * 28   # ~100 K pixels
MAX_PIXELS = 256 * 28 * 28   # ~200 K pixels  ← conservative for T4 / Colab

# ── Module-level globals (populated by load_model()) ─────────────────────────
model     = None
processor = None

# ── Class names ───────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "NIF_certificate",
    "NIS_certificate",
    "certificat_existence",
    "tax_declaration_form",
    "residence_certificate",
    "legal_contract_front",
    "legal_contract_inside",
    "balance_sheet",
    "RC_front",
    "RC_inside_activities",
    "RC_inside_2",
    "driving_license_front",
    "driving_license_back",
    "driving_license_frontback",
]

# ── Classification prompt ─────────────────────────────────────────────────────
CLASSIFICATION_PROMPT = """You are a document classifier for Algerian official documents.
Classify the image into EXACTLY ONE category from this list:
RC_front | RC_inside_2 | RC_inside_activities | NIS_certificate | NIF_certificate |
certificat_existence | tax_declaration_form | balance_sheet | residence_certificate |
legal_contract_front | legal_contract_inside | driving_license_front |
driving_license_back | driving_license_frontback

INSTRUCTIONS: Work through the checks below IN ORDER. Stop at the FIRST match. Output ONLY the category name — no explanation, no punctuation, no extra words.

━━━ STAGE 1 — UNIQUE VISUAL ANCHORS (highest confidence, check first) ━━━

CHECK 1 — RC_front
  • Large pill/oval rectangle in center with Arabic company name
  • CNRC logo + QR code in corner
  • Light green or beige background
  → RC_front

CHECK 2 — driving_license_frontback   ⚠ ALWAYS CHECK THIS FIRST
  PRIMARY signal (definitive): TWO separate credit-card rectangles visible
  in the same image, stacked vertically or side-by-side, with a clear gap
  or shadow between them.
  • Top/first card shows a FACE PHOTO (front side)
  • Bottom/second card shows a CHIP + MRZ lines (back side)
  • Orientation may be portrait OR landscape — do not rely on orientation alone
  → driving_license_frontback

CHECK 3 — driving_license_front   (only if exactly ONE card is visible)
  PRIMARY signal (definitive): FACE PHOTO present on the card
  • Single card, wider than tall (landscape)
  • Text "DZ", "DRIVING LICENSE", "رخصة القيادة" visible at top
  • Face photo occupies LEFT half; numbered data fields on RIGHT half
  • NO golden chip anywhere on the card
  → driving_license_front

CHECK 4 — driving_license_back   (only if exactly ONE card is visible)
  PRIMARY signal (definitive):  CHIP present
  • Single card, wider than tall (landscape)
  • Chip on LEFT side of card
  • Vehicle category codes (A, A1, B, C, D, BE, CE) in a table
  • Three lines of MRZ text at bottom, beginning with "DLDZAA"
  → driving_license_back

CHECK 5 — RC_inside_2
  • Bold Arabic header: العقوبات التي يتعرض لها الخاضع للقيد
  • 9-digit serial number الرقم التسلسلي visible (e.g. 900116729)
  • Large circular CNRC official stamp  — this is NOT a diagonal stroke
  • Authentication box with handwritten date + branch name + signature
  → RC_inside_2

CHECK 6 — RC_inside_activities
  • Page split into TOP and BOTTOM sections by a horizontal line
  • TOP: bold headers عنوان الشركة / الشكل القانوني / مبلغ رأسمال الشركة + dotted lines
         + 5-column table الممثل أو الممثلون الشرعيون with birth dates and nationalities
  • BOTTOM: 6-digit activity codes (e.g. 428301) + Arabic descriptions ending in ***...***
  • Paper may be yellow-green or white; may be rotated 90°
  → RC_inside_activities

CHECK 7 — NIS_certificate  ⚠ ALL FOUR signals must be present
  • Bordered box containing: الفهرس الوطني للأعوان الإقتصاديين و الإجتماعيين
  • Second bordered box: إشعار بالتعريف
  • Bilingual layout: French fields left (NOM, SIGLE, ADRESSE) / Arabic fields right
  • Long NIS number format: 0 021 XXXX XXXXX XX
  • O.N.S circular logo top-right + round ONS stamp bottom center
  → NIS_certificate
  ✗ Missing any one of these → NOT NIS_certificate

CHECK 8 — NIF_certificate  ⚠ Check AFTER certificat_existence
  • Header: building icon + DIRECTION GENERALE DES IMPOTS
  • Two stacked bordered boxes: ATTESTATION D'IMMATRICULATION FISCALE / NIF
  • Large blank white lower half — NO diagonal strokes
  → NIF_certificate

CHECK 9 — certificat_existence  ⚠ Must have diagonal strokes
  • CERTIFICAT bold upper-right + Série C n° reference code
  • Diagonal pen strokes (~45°) slashing across blank lower half — REQUIRED
  • Row of empty NIF digit boxes + circular DGI stamp
  ✗ Page has CNRC stamp (circular, not DGI) → NOT certificat_existence → go to CHECK 5
  ✗ Page contains Arabic text headers or العقوبات → NOT certificat_existence → go to CHECK 5
  → certificat_existence
  ✗ No diagonal strokes → NOT certificat_existence → go to CHECK 8

CHECK 10 — tax_declaration_form
  • Bold DECLARATION D'EXISTENCE at very top
  • Two rows of digit boxes labeled NIS and NIF
  • FORME JURIDIQUE DE L'ENTREPRISE checkbox list
  • Circular stamp bottom-right; NO diagonal slashes; NO 6-digit activity codes
  → tax_declaration_form

CHECK 11 — balance_sheet
  REQUIRED booklet header (present on most pages):
    • "IMPRIMÉ DESTINÉ AU CONTRIBUABLE" or "IMPRIME DESTINE A L'ADMINISTRATION"
    • NIF number displayed in a row of individual digit boxes at top-right
    • Company fields: Désignation de l'entreprise / Activité / Adresse
    • Fiscal period line: "Exercice du ... au ..." or "Exercice clos le ..."

  SUB-PATTERN A — BILAN ACTIF
    • Bold centered title: BILAN (ACTIF) or BILAN ACTIF
    • 4 or 5-column accounting table
    • Bold section groups: ACTIFS NON COURANTS / ACTIF COURANT
    → balance_sheet

  SUB-PATTERN B — BILAN PASSIF
    • Bold centered title: BILAN (PASSIF) or BILAN PASSIF
    • Section headers: CAPITAUX PROPRES / PASSIFS NON-COURANTS / PASSIFS COURANTS
    → balance_sheet

  SUB-PATTERN C — COMPTE DE RESULTAT
    • Bold centered title: COMPTE DE RESULTAT
    • 4-column table: DEBIT / CREDIT for N and N-1
    → balance_sheet

  SUB-PATTERN D — ANNEXE NUMBERED SCHEDULE
    • Numbered section titles: "N/ Tableau des ..." or "N/ Relevé des ..."
    • Table may have large printed block-letter "NEANT" stamp
    → balance_sheet

  SUB-PATTERN E — ASSET REGISTER
    • Dense multi-column table: year | asset code | description | amount
    → balance_sheet

  SUB-PATTERN F — BORDEREAU AVIS DE VERSEMENT
    • Contains header: "Impôts sur les Bénéfices des Sociétés"
    • Bold title: BORDEREAU AVIS DE VERSEMENT
    → balance_sheet

  STOP CONDITIONS:
    ✗ Arabic text inside main data table cells → NOT balance_sheet
    ✗ DECLARATION D'EXISTENCE with NIS/NIF digit input rows → tax_declaration_form
    ✗ ***...*** asterisk strings + 6-digit codes → RC_inside_activities

  → balance_sheet

CHECK 12 — residence_certificate
  • Large bold Arabic title بطاقة إقامة at very top
  • Sparse Arabic rows on dotted lines; phrase نشهد بأن in body
  • Two circular stamps (one at bottom-left, one mid-right)
  • NO CNRC logo; NO NIF/NIS boxes; NO driving license chip
  → residence_certificate

CHECK 13 — legal_contract_front
  • CENTER of page is BLANK or nearly blank — no dense text block
  • Ornamental or decorative border around the page edge
  • Republic of Algeria header: الجمهورية الجزائرية الديمقراطية الشعبية
  • Notary office header at top (وثيقة رسمية / Acte Authentique)
  • Single large circular stamp or embossed seal
  → legal_contract_front

CHECK 14 — legal_contract_inside
  • DENSE paragraph body text in Arabic (right-to-left prose)
  • Contains القانون or legal clause references
  • Numbered article/paragraph markers (أولاً, ثانياً, المادة X)
  • Page number at bottom; may have small circular stamps in margins
  → legal_contract_inside

━━━ DISAMBIGUATION TABLE (when two classes look similar) ━━━
NIF_certificate vs certificat_existence → diagonal strokes = certificat_existence, always.
NIS_certificate vs NIF_certificate → ONS logo + NIS long number = NIS_certificate.
RC_inside_2 vs RC_inside_activities → العقوبات header = RC_inside_2, always.
RC_inside_activities vs tax_declaration_form → ***...*** asterisk strings + 6-digit codes = RC_inside_activities, always.
driving_license_frontback vs front/back → TWO stacked cards = frontback, always.
legal_contract_front vs legal_contract_inside → Blank center + ornaments = front. Dense prose + القانون title = inside.
balance_sheet vs tax_declaration_form → BORDEREAU AVIS DE VERSEMENT inside a liasse = balance_sheet.
balance_sheet vs certificat_existence → "NEANT" block-letter stamp = balance_sheet. Diagonal handwritten strokes = certificat_existence.

⚠ BONUS RULES (added for robustness):
If the image is blurry, rotated, or partially cut off, still pick the CLOSEST match from the list.
Never output anything other than one of the 14 category names.
EXAMPLE: If you see a document with a QR code and green background → output: RC_front

━━━ OUTPUT ━━━
Reply with EXACTLY ONE of these strings and nothing else:
RC_front | RC_inside_2 | RC_inside_activities | NIS_certificate | NIF_certificate |
certificat_existence | tax_declaration_form | balance_sheet | residence_certificate |
legal_contract_front | legal_contract_inside | driving_license_front |
driving_license_back | driving_license_frontback

Category:"""


# ─────────────────────────────────────────────────────────────────────────────
# _get_bnb_config()
# Returns a BitsAndBytesConfig for 4-bit NF4 quantization.
# This keeps the 7B model under ~5 GB VRAM — safe on Colab T4 (15 GB).
# ─────────────────────────────────────────────────────────────────────────────
def _get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NF4 is better than fp4 for LLMs
        bnb_4bit_use_double_quant=True,      # nested quantization → saves ~0.4 GB extra
        bnb_4bit_compute_dtype=torch.bfloat16  # compute in bf16 for speed + stability
        if torch.cuda.is_bf16_supported()
        else torch.float16,
    )


# ─────────────────────────────────────────────────────────────────────────────
# load_model()
# Called ONCE at FastAPI startup via lifespan() in app.py.
# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    global model, processor

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. "
            "Qwen2.5-VL-7B-Instruct (4-bit) still requires a GPU. "
            "Attach a GPU in Colab: Runtime → Change runtime type → T4 GPU."
        )

    bnb_config = _get_bnb_config()
    compute_dtype = bnb_config.bnb_4bit_compute_dtype

    print(f"⏳ Loading {MODEL_ID} in 4-bit quantization")
    print(f"   compute_dtype={compute_dtype}  device_map=auto")
    print(f"   GPUs detected: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1e9
        print(f"   GPU {i}: {props.name}  {total:.1f} GB VRAM")
    print("   (First run downloads ~15 GB of model weights — please wait...)")

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{MODEL_ID}': {e}") from e

    model.eval()

    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load processor for '{MODEL_ID}': {e}") from e

    # Report GPU memory after loading
    for i in range(torch.cuda.device_count()):
        used  = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"   GPU {i} after load: {used:.1f} GB / {total:.1f} GB used")

    print(f"✅ Model ready (4-bit): {MODEL_ID}")


# ─────────────────────────────────────────────────────────────────────────────
# _classify_document()
# Single forward pass. Clears GPU cache on OOM and re-raises as RuntimeError.
# ─────────────────────────────────────────────────────────────────────────────
def _classify_document(pil_image: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": CLASSIFICATION_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move all tensors to the model's device.
    # With device_map="auto" the model may span multiple devices;
    # model.device refers to the first shard — use it for inputs.
    target_device = next(model.parameters()).device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=20,      # class names are ≤ 30 chars — 20 tokens is enough
                do_sample=False,        # greedy decoding: faster + deterministic
                temperature=None,       # not needed for greedy
                top_p=None,             # not needed for greedy
                repetition_penalty=1.0, # no penalty needed — output is a single label
            )
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "GPU out of memory during inference. "
            "Try uploading a lower-resolution image (max 2000×2000 px)."
        ) from e

    input_length = inputs["input_ids"].shape[1]
    trimmed_ids  = generated_ids[0][input_length:]
    return processor.decode(trimmed_ids, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# _match_label()
# Maps raw model output → canonical class name + confidence.
# ─────────────────────────────────────────────────────────────────────────────
def _match_label(raw_output: str) -> tuple:
    normalized = raw_output.strip().rstrip(".,;:")
    lower_map  = {c.lower(): c for c in CLASS_NAMES}

    if normalized in CLASS_NAMES:
        return normalized, 1.0

    if normalized.lower() in lower_map:
        return lower_map[normalized.lower()], 0.95

    close = difflib.get_close_matches(
        normalized.lower(),
        [c.lower() for c in CLASS_NAMES],
        n=1,
        cutoff=0.4,
    )
    if close:
        matched    = lower_map[close[0]]
        confidence = difflib.SequenceMatcher(None, normalized.lower(), close[0]).ratio()
        return matched, round(confidence, 4)

    return "unknown", 0.0


# ─────────────────────────────────────────────────────────────────────────────
# classify_image()
# Public entry point called by the FastAPI /classify endpoint.
# ─────────────────────────────────────────────────────────────────────────────
def classify_image(image_bytes: bytes) -> dict:
    if model is None or processor is None:
        raise RuntimeError("Model is not loaded. Server startup may have failed.")

    pil_image  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    raw_output = _classify_document(pil_image)
    label, confidence = _match_label(raw_output)
    return {"label": label, "confidence": confidence}
