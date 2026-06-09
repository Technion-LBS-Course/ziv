"""Helipad Intelligence Engine (HIE) — visual detection module.

Imported by scripts/compare_zero_shot.py, scripts/validate_helipad_cascade.py,
and app.py.  All detection functions are pure (no I/O side-effects) and return a
unified result dict so callers can switch detectors without changing logic.

NAIP chip geometry
------------------
  640 × 640 px  |  100 m × 100 m window  |  0.15625 m/px
  The FAA coordinate is always at the chip centre (cx = cy = 320).

Result dict schema
------------------
  {
    "detected":    bool,
    "bbox_px":     [x1, y1, x2, y2] | None,   # pixel coords in 640×640 chip
    "cx":          int | None,                  # bbox centre x (pixels)
    "cy":          int | None,                  # bbox centre y (pixels)
    "confidence":  float,
    "method":      str,
    "latency_s":   float,
  }
"""

import logging
import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# ── NAIP chip constants ──────────────────────────────────────────────────────
IMG_PX: int = 640
NAIP_WINDOW_M: float = 100.0
GSD_M: float = NAIP_WINDOW_M / IMG_PX  # 0.15625 m/px

# ── Paths ────────────────────────────────────────────────────────────────────
_PROJ_ROOT = Path(__file__).resolve().parents[1]
YOLO_MODEL_PATH = _PROJ_ROOT / "models" / "helipad_yolov8s.pt"


# ────────────────────────────────────────────────────────────────────────────
# Chip I/O
# ────────────────────────────────────────────────────────────────────────────

def load_chip(chip_path: Path) -> Image.Image:
    """Load a 640×640 NAIP chip from disk as an RGB PIL Image.

    Args:
        chip_path: Path to a JPEG chip produced by build_yolo_dataset.py.

    Returns:
        PIL Image in RGB mode.

    Raises:
        FileNotFoundError: If chip_path does not exist.
    """
    if not chip_path.exists():
        raise FileNotFoundError(chip_path)
    return Image.open(chip_path).convert("RGB")


# ────────────────────────────────────────────────────────────────────────────
# Tier 1 — Classical CV  (H-template matching)
# ────────────────────────────────────────────────────────────────────────────

def _make_h_template(size_px: int, rotated: bool = False) -> np.ndarray:
    """Build a binary H-shape template as a float32 numpy array.

    Args:
        size_px: Template side length in pixels.
        rotated: If True, rotate 90° (legs top/bottom, crossbar vertical).

    Returns:
        float32 array in [0, 255], H shape white on black.
    """
    s = max(2, size_px // 5)          # stroke width
    tmpl = np.zeros((size_px, size_px), dtype=np.float32)
    # Left vertical bar
    tmpl[:, :s] = 255.0
    # Right vertical bar
    tmpl[:, size_px - s:] = 255.0
    # Horizontal crossbar
    mid = size_px // 2
    half = max(1, s // 2)
    tmpl[mid - half: mid + half, :] = 255.0
    if rotated:
        tmpl = cv2.rotate(tmpl, cv2.ROTATE_90_CLOCKWISE)
    return tmpl


# Template bank: (size_px, rotated, inverted_bg)
_TEMPLATE_SPECS = [
    (30,  False), (30,  True),
    (60,  False), (60,  True),
    (100, False), (100, True),
]
_TEMPLATES: Optional[list] = None   # built lazily


def _get_templates() -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (white-on-dark, dark-on-white) template pairs."""
    global _TEMPLATES
    if _TEMPLATES is not None:
        return _TEMPLATES
    _TEMPLATES = []
    for size, rotated in _TEMPLATE_SPECS:
        t = _make_h_template(size, rotated)
        _TEMPLATES.append((t, 255.0 - t))
    return _TEMPLATES


def detect_classical(image: Image.Image) -> dict:
    """Tier 1: OpenCV normalised cross-correlation against H-shape templates.

    Matches white-H-on-dark and dark-H-on-light at three scales and two
    orientations (0° and 90°).  Only detects H-marked pads — hospital
    rooftops fall through to Tier 2.

    Args:
        image: PIL RGB Image, expected 640×640 px.

    Returns:
        Unified detection result dict (see module docstring).
    """
    t0 = time.perf_counter()

    gray = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    best_score: float = 0.0
    best_loc = (IMG_PX // 2, IMG_PX // 2)
    best_size = 60

    for variants in _get_templates():
        for tmpl in variants:
            if tmpl.shape[0] > gray.shape[0] or tmpl.shape[1] > gray.shape[1]:
                continue
            result = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                th, tw = tmpl.shape[:2]
                # max_loc is top-left corner; shift to centre of template
                best_loc = (max_loc[0] + tw // 2, max_loc[1] + th // 2)
                best_size = tw

    threshold = 0.72
    detected = best_score >= threshold
    cx, cy = best_loc
    half = best_size // 2
    bbox = [cx - half, cy - half, cx + half, cy + half] if detected else None

    return {
        "detected":   detected,
        "bbox_px":    bbox,
        "cx":         cx if detected else None,
        "cy":         cy if detected else None,
        "confidence": float(best_score),
        "method":     "classical",
        "latency_s":  time.perf_counter() - t0,
    }


# ────────────────────────────────────────────────────────────────────────────
# Tier 2 — Fine-tuned YOLOv8s
# ────────────────────────────────────────────────────────────────────────────

def detect_yolo(image: Image.Image, model) -> dict:
    """Tier 2: YOLOv8s fine-tuned on NAIP helipad chips.

    Args:
        image: PIL RGB Image, 640×640 px.
        model: Loaded ultralytics YOLO model (helipad_yolov8s.pt).

    Returns:
        Unified detection result dict.
    """
    t0 = time.perf_counter()
    results = model.predict(np.asarray(image), conf=0.25, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        return {"detected": False, "bbox_px": None, "cx": None, "cy": None,
                "confidence": 0.0, "method": "yolo_finetuned",
                "latency_s": time.perf_counter() - t0}

    # Pick highest-confidence box
    idx = int(boxes.conf.argmax())
    conf = float(boxes.conf[idx])
    x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[idx].tolist())
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    return {
        "detected":   True,
        "bbox_px":    [x1, y1, x2, y2],
        "cx":         cx,
        "cy":         cy,
        "confidence": conf,
        "method":     "yolo_finetuned",
        "latency_s":  time.perf_counter() - t0,
    }


def load_yolo_model(path: Path = YOLO_MODEL_PATH):
    """Load a trained ultralytics YOLO model from disk.

    Args:
        path: Path to .pt weights file (default: models/helipad_yolov8s.pt).

    Returns:
        Loaded YOLO model.

    Raises:
        FileNotFoundError: If weights file does not exist.
    """
    from ultralytics import YOLO
    if not path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {path}")
    return YOLO(str(path))


# ────────────────────────────────────────────────────────────────────────────
# Zero-shot comparison — YOLO-World small
# ────────────────────────────────────────────────────────────────────────────

_YOLO_WORLD_CLASSES = ["helipad", "landing pad", "H marking"]

_yolo_world_model = None  # module-level singleton


def load_yolo_world_model():
    """Load and configure YOLO-World small (yolov8s-worldv2.pt).

    Downloads ~14 MB weights to ~/.config/Ultralytics/ on first use.

    Returns:
        Configured YOLO-World model.
    """
    global _yolo_world_model
    if _yolo_world_model is None:
        from ultralytics import YOLO
        _yolo_world_model = YOLO("yolov8s-worldv2.pt")
        _yolo_world_model.set_classes(_YOLO_WORLD_CLASSES)
        log.info("YOLO-World loaded, classes: %s", _YOLO_WORLD_CLASSES)
    return _yolo_world_model


def detect_yolo_world(image: Image.Image, model=None, conf: float = 0.05) -> dict:
    """Zero-shot comparison: YOLO-World small open-vocabulary detection.

    No fine-tuning — runs purely on the text classes "helipad", "landing pad",
    "H marking".  Loads the model on first call if not provided.

    Args:
        image: PIL RGB Image, 640×640 px.
        model: Pre-loaded YOLO-World model (optional; loaded automatically).
        conf: Confidence threshold (low default to capture weak detections).

    Returns:
        Unified detection result dict.
    """
    t0 = time.perf_counter()

    if model is None:
        model = load_yolo_world_model()

    results = model.predict(np.asarray(image), conf=conf, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        return {"detected": False, "bbox_px": None, "cx": None, "cy": None,
                "confidence": 0.0, "method": "yolo_world",
                "latency_s": time.perf_counter() - t0}

    idx = int(boxes.conf.argmax())
    conf_val = float(boxes.conf[idx])
    x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[idx].tolist())
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    return {
        "detected":   True,
        "bbox_px":    [x1, y1, x2, y2],
        "cx":         cx,
        "cy":         cy,
        "confidence": conf_val,
        "method":     "yolo_world",
        "latency_s":  time.perf_counter() - t0,
    }


# ────────────────────────────────────────────────────────────────────────────
# Zero-shot comparison — Florence-2-base
# ────────────────────────────────────────────────────────────────────────────

_FL2_MODEL_ID = "microsoft/Florence-2-base"
_FL2_TASK = "<OPEN_VOCABULARY_DETECTION>"
_FL2_PROMPT = "helipad"

_fl2_model = None
_fl2_processor = None


def _patch_fl2_config(model) -> None:
    """Set forced_bos_token_id=None on every Florence-2 sub-config that lacks it.

    Florence2LanguageConfig does not define forced_bos_token_id, which
    transformers >= 4.49 requires in GenerationMixin.generate().
    Patch all candidate config objects so generate() never hits AttributeError.
    """
    candidates = [model.config]
    for attr in ("language_model", "text_model"):
        sub = getattr(model, attr, None)
        if sub is not None:
            candidates.append(sub.config)
    for attr in ("text_config", "vision_config"):
        sub = getattr(model.config, attr, None)
        if sub is not None:
            candidates.append(sub)

    for cfg in candidates:
        if cfg is not None and not hasattr(cfg, "forced_bos_token_id"):
            cfg.forced_bos_token_id = None


def load_florence2_model(model_id: str = _FL2_MODEL_ID):
    """Load Florence-2-base from HuggingFace (~460 MB, cached after first download).

    COMPATIBILITY NOTE: Florence-2's custom model code is incompatible with
    transformers >= 4.49 (missing forced_bos_token_id / _supports_sdpa attributes).
    This function will raise an ImportError if transformers >= 4.49 is installed.
    To use Florence-2, install: pip install transformers>=4.44.0,<4.49.0

    Args:
        model_id: HuggingFace model ID.

    Returns:
        (model, processor) tuple.

    Raises:
        ImportError: If transformers >= 4.49 is detected.
    """
    global _fl2_model, _fl2_processor
    if _fl2_model is None:
        import torch
        import transformers
        from packaging.version import Version

        tv = Version(transformers.__version__)
        if tv >= Version("4.49"):
            raise ImportError(
                f"Florence-2 requires transformers < 4.49 (installed: {transformers.__version__}). "
                "Install a compatible version: pip install 'transformers>=4.44.0,<4.49.0'"
            )

        from transformers import AutoModelForCausalLM, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        log.info("Loading Florence-2 (%s) on %s …", model_id, device)
        _fl2_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        _fl2_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        _patch_fl2_config(_fl2_model)
        log.info("Florence-2 loaded (forced_bos_token_id patched).")
    return _fl2_model, _fl2_processor


def detect_florence2(
    image: Image.Image,
    model=None,
    processor=None,
    conf: float = 0.05,
) -> dict:
    """Zero-shot comparison: Florence-2-base open-vocabulary detection.

    Uses the <OPEN_VOCABULARY_DETECTION> task with prompt "helipad".
    Downloads ~460 MB weights on first call if model not provided.

    Args:
        image: PIL RGB Image, 640×640 px.
        model: Pre-loaded Florence-2 model (optional; loaded automatically).
        processor: Pre-loaded Florence-2 processor (optional).
        conf: Minimum confidence to accept a detection (post-processing only
              — Florence-2 does not return per-box scores, so this is unused
              and kept for API uniformity; confidence is reported as 1.0).

    Returns:
        Unified detection result dict.
    """
    t0 = time.perf_counter()

    if model is None or processor is None:
        model, processor = load_florence2_model()

    import torch

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    prompt_text = _FL2_TASK + _FL2_PROMPT
    inputs = processor(text=prompt_text, images=image, return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=dtype) if v.dtype.is_floating_point else v.to(device=device)
              for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    try:
        parsed = processor.post_process_generation(
            generated_text,
            task=_FL2_TASK,
            image_size=(image.width, image.height),
        )
        bboxes = parsed[_FL2_TASK].get("bboxes", [])
    except Exception as exc:
        log.warning("Florence-2 post-process failed: %s", exc)
        bboxes = []

    if not bboxes:
        return {"detected": False, "bbox_px": None, "cx": None, "cy": None,
                "confidence": 0.0, "method": "florence2",
                "latency_s": time.perf_counter() - t0}

    # Pick the box closest to image centre (most likely to be the main subject)
    best_bbox = min(
        bboxes,
        key=lambda b: (((b[0] + b[2]) / 2 - IMG_PX / 2) ** 2 +
                       ((b[1] + b[3]) / 2 - IMG_PX / 2) ** 2),
    )
    x1, y1, x2, y2 = (int(round(v)) for v in best_bbox)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    return {
        "detected":   True,
        "bbox_px":    [x1, y1, x2, y2],
        "cx":         cx,
        "cy":         cy,
        "confidence": 1.0,   # Florence-2 does not return per-box scores
        "method":     "florence2",
        "latency_s":  time.perf_counter() - t0,
    }


# ────────────────────────────────────────────────────────────────────────────
# Zero-shot comparison (optional) — Grounding DINO tiny
# ────────────────────────────────────────────────────────────────────────────

_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
_DINO_LABELS = [["helipad", "H marking", "circular landing pad"]]

_dino_model = None
_dino_processor = None


def load_dino_model(model_id: str = _DINO_MODEL_ID):
    """Load Grounding DINO tiny (~661 MB, cached after first download).

    Args:
        model_id: HuggingFace model ID.

    Returns:
        (model, processor) tuple.
    """
    global _dino_model, _dino_processor
    if _dino_model is None:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        log.info("Loading Grounding DINO (%s) …", model_id)
        _dino_processor = AutoProcessor.from_pretrained(model_id)
        _dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        log.info("Grounding DINO loaded.")
    return _dino_model, _dino_processor


def detect_dino(
    image: Image.Image,
    model=None,
    processor=None,
    conf: float = 0.10,
) -> dict:
    """Zero-shot comparison (optional): Grounding DINO tiny.

    Note: published benchmarks show poor zero-shot performance on nadir
    aerial imagery without domain adaptation (arxiv 2601.22164).  Run for
    academic completeness only.

    Args:
        image: PIL RGB Image, 640×640 px.
        model: Pre-loaded GDINO model (optional; loaded automatically).
        processor: Pre-loaded GDINO processor (optional).
        conf: Confidence threshold for detections.

    Returns:
        Unified detection result dict.
    """
    t0 = time.perf_counter()

    if model is None or processor is None:
        model, processor = load_dino_model()

    import torch

    inputs = processor(images=image, text=_DINO_LABELS, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        text_threshold=conf,
        target_sizes=[image.size[::-1]],
    )[0]

    boxes = results["boxes"]
    scores = results["scores"]

    if len(boxes) == 0:
        return {"detected": False, "bbox_px": None, "cx": None, "cy": None,
                "confidence": 0.0, "method": "dino",
                "latency_s": time.perf_counter() - t0}

    idx = int(scores.argmax())
    conf_val = float(scores[idx])
    x1, y1, x2, y2 = (int(round(v)) for v in boxes[idx].tolist())
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    return {
        "detected":   True,
        "bbox_px":    [x1, y1, x2, y2],
        "cx":         cx,
        "cy":         cy,
        "confidence": conf_val,
        "method":     "dino",
        "latency_s":  time.perf_counter() - t0,
    }


# ────────────────────────────────────────────────────────────────────────────
# Production cascade — Tier 1 → Tier 2
# ────────────────────────────────────────────────────────────────────────────

def detect_helipad_cascade(
    image: Image.Image,
    yolo_model,
    classical_threshold: float = 0.75,
) -> dict:
    """Production Tier 1 → Tier 2 cascade.

    Runs classical H-template matching first.  If confidence ≥ threshold,
    returns that result immediately.  Otherwise falls through to YOLOv8s.

    Args:
        image: PIL RGB Image, 640×640 px.
        yolo_model: Loaded ultralytics YOLO model (helipad_yolov8s.pt).
        classical_threshold: Classical CV confidence cutoff (default 0.75).

    Returns:
        Unified detection result dict with method='classical' or 'yolo_finetuned'.
    """
    t0 = time.perf_counter()
    classical = detect_classical(image)
    if classical["confidence"] >= classical_threshold:
        classical["latency_s"] = time.perf_counter() - t0
        return classical
    result = detect_yolo(image, yolo_model)
    result["latency_s"] = time.perf_counter() - t0
    return result


# ────────────────────────────────────────────────────────────────────────────
# Coordinate utilities
# ────────────────────────────────────────────────────────────────────────────

def bbox_px_to_latlon(
    bbox_px: list[int],
    chip_lat: float,
    chip_lon: float,
    window_m: float = NAIP_WINDOW_M,
    img_px: int = IMG_PX,
) -> tuple[float, float]:
    """Convert a pixel-space bbox centre to geographic coordinates.

    Assumes the chip is centred on (chip_lat, chip_lon) and covers a square
    window of window_m × window_m metres.

    Args:
        bbox_px: [x1, y1, x2, y2] in pixel coordinates.
        chip_lat: Latitude of the chip centre (the FAA registered coordinate).
        chip_lon: Longitude of the chip centre.
        window_m: Side length of the chip in metres (default 100 m).
        img_px: Chip side length in pixels (default 640).

    Returns:
        (lat, lon) of the detected bbox centre.
    """
    cx_px = (bbox_px[0] + bbox_px[2]) / 2
    cy_px = (bbox_px[1] + bbox_px[3]) / 2
    gsd = window_m / img_px

    # Image y-axis points down; north is up
    dx_m = (cx_px - img_px / 2) * gsd    # positive = east
    dy_m = (img_px / 2 - cy_px) * gsd    # positive = north

    lat = chip_lat + dy_m / 111_320
    lon = chip_lon + dx_m / (111_320 * math.cos(math.radians(chip_lat)))
    return lat, lon


def compute_offset_m(
    ref_lat: float,
    ref_lon: float,
    det_lat: float,
    det_lon: float,
) -> float:
    """Haversine distance in metres between a registry coordinate and a detection.

    Args:
        ref_lat: Registry latitude.
        ref_lon: Registry longitude.
        det_lat: Detected centroid latitude.
        det_lon: Detected centroid longitude.

    Returns:
        Distance in metres.
    """
    R = 6_371_000
    lat1, lat2 = math.radians(ref_lat), math.radians(det_lat)
    dlat = math.radians(det_lat - ref_lat)
    dlon = math.radians(det_lon - ref_lon)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))
