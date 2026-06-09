"""Helipad dataset annotation review tool.

Run with:
    streamlit run scripts/annotate_dataset.py

For each chip you can:
  ✓  Approve     — bbox is correct, keep as-is
  ✗  Disqualify  — no visible helipad, remove label
  ✏  Adjust      — drag sliders to correct the bbox, then Save

All decisions are written to data/yolo_dataset/review_decisions.csv
and kept until you click "Apply decisions → write YOLO labels".
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR = ROOT / "data" / "yolo_dataset"
DECISIONS_CSV = YOLO_DIR / "review_decisions.csv"
IMG_PX = 640

st.set_page_config(
    page_title="Helipad Annotation Review",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── data helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_manifest() -> pd.DataFrame:
    """Scan all chips and note which have a bbox in their label file."""
    rows = []
    for split in ("train", "val", "test"):
        img_dir = YOLO_DIR / "images" / split
        lbl_dir = YOLO_DIR / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            has_bbox = False
            is_synthetic = False
            if lbl_path.exists():
                try:
                    txt = lbl_path.read_text(encoding="utf-8", errors="ignore")
                    has_bbox = any(ln.startswith("0 ") for ln in txt.splitlines())
                    is_synthetic = "SYNTHETIC" in txt
                except OSError:
                    pass
            rows.append({
                "chip_stem": img_path.stem,
                "split": split,
                "img_path": str(img_path),
                "lbl_path": str(lbl_path),
                "has_bbox": has_bbox,
                "is_synthetic": is_synthetic,
            })
    return pd.DataFrame(rows)


def load_decisions() -> pd.DataFrame:
    """Load review_decisions.csv, creating empty frame if absent."""
    cols = ["chip_stem", "action", "new_x1", "new_y1", "new_x2", "new_y2", "reviewed_at"]
    if DECISIONS_CSV.exists():
        try:
            df = pd.read_csv(DECISIONS_CSV)
            for c in cols:
                if c not in df.columns:
                    df[c] = None
            return df[cols]
        except Exception:
            pass
    return pd.DataFrame(columns=cols)


def save_decision(chip_stem: str, action: str, bbox: tuple | None) -> None:
    """Write (or overwrite) one decision row and flush to disk."""
    df = load_decisions()
    df = df[df.chip_stem != chip_stem].copy()
    df = pd.concat([df, pd.DataFrame([{
        "chip_stem": chip_stem,
        "action": action,
        "new_x1": bbox[0] if bbox else None,
        "new_y1": bbox[1] if bbox else None,
        "new_x2": bbox[2] if bbox else None,
        "new_y2": bbox[3] if bbox else None,
        "reviewed_at": datetime.now().isoformat(timespec="seconds"),
    }])], ignore_index=True)
    DECISIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DECISIONS_CSV, index=False)
    load_manifest.clear()


def parse_label_bbox(lbl_path: str) -> tuple[int, int, int, int] | None:
    """Return (x1, y1, x2, y2) pixels from first YOLO line, or None."""
    try:
        for line in Path(lbl_path).read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("0 ") and len(line.split()) >= 5:
                _, cx_n, cy_n, w_n, h_n = line.split()[:5]
                cx_n, cy_n, w_n, h_n = map(float, (cx_n, cy_n, w_n, h_n))
                x1 = max(0, int((cx_n - w_n / 2) * IMG_PX))
                y1 = max(0, int((cy_n - h_n / 2) * IMG_PX))
                x2 = min(IMG_PX, int((cx_n + w_n / 2) * IMG_PX))
                y2 = min(IMG_PX, int((cy_n + h_n / 2) * IMG_PX))
                return x1, y1, x2, y2
    except Exception:
        pass
    return None


def render_chip(img_path: str, bbox: tuple | None, color: str = "red") -> Image.Image:
    """Load chip and overlay bbox rectangle + centre crosshair."""
    img = Image.open(img_path).convert("RGB")
    if bbox and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
        x1, y1, x2, y2 = bbox
        d = ImageDraw.Draw(img)
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        d.line([cx - 12, cy, cx + 12, cy], fill=color, width=2)
        d.line([cx, cy - 12, cx, cy + 12], fill=color, width=2)
    return img


def apply_all_decisions() -> dict:
    """Write YOLO label files for all 'disqualified' and 'adjusted' chips."""
    manifest = load_manifest()
    decisions = load_decisions()
    dec = {r.chip_stem: r for _, r in decisions.iterrows()}
    stats = {"disqualified": 0, "adjusted": 0}

    for _, chip in manifest.iterrows():
        d = dec.get(chip.chip_stem)
        if d is None:
            continue
        lbl = Path(chip.lbl_path)
        lbl.parent.mkdir(parents=True, exist_ok=True)

        if d.action == "disqualified":
            lbl.write_text("", encoding="utf-8")
            stats["disqualified"] += 1

        elif d.action == "adjusted" and all(
            pd.notna(d[k]) for k in ["new_x1", "new_y1", "new_x2", "new_y2"]
        ):
            x1, y1, x2, y2 = int(d.new_x1), int(d.new_y1), int(d.new_x2), int(d.new_y2)
            if x2 > x1 and y2 > y1:
                cx_n = ((x1 + x2) / 2) / IMG_PX
                cy_n = ((y1 + y2) / 2) / IMG_PX
                lbl.write_text(
                    f"0 {cx_n:.6f} {cy_n:.6f} {(x2-x1)/IMG_PX:.6f} {(y2-y1)/IMG_PX:.6f}\n",
                    encoding="utf-8",
                )
                stats["adjusted"] += 1

    load_manifest.clear()
    return stats


# ── main UI ────────────────────────────────────────────────────────────────────

def main() -> None:  # noqa: C901
    manifest = load_manifest()
    decisions = load_decisions()

    if manifest.empty:
        st.error(f"No chips found under {YOLO_DIR / 'images'}. Run the dataset build script first.")
        return

    dec_set = set(decisions.chip_stem)
    dec_action = {r.chip_stem: r.action for _, r in decisions.iterrows()}

    # ── sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🚁 Annotation Review")
        st.caption("Approve · Disqualify · Adjust bbox")

        # Progress
        n_total = len(manifest)
        n_done = len(dec_set & set(manifest.chip_stem))
        n_approved = sum(1 for s, a in dec_action.items() if a == "approved" and s in set(manifest.chip_stem))
        n_disq = sum(1 for s, a in dec_action.items() if a == "disqualified" and s in set(manifest.chip_stem))
        n_adj = sum(1 for s, a in dec_action.items() if a == "adjusted" and s in set(manifest.chip_stem))

        st.progress(n_done / n_total if n_total else 0,
                    text=f"{n_done} / {n_total} reviewed")
        st.markdown(
            f"✅ {n_approved} approved &nbsp; "
            f"✗ {n_disq} disqualified &nbsp; "
            f"✏️ {n_adj} adjusted"
        )
        st.divider()

        # Filter
        filter_opt = st.radio(
            "Show chips",
            ["Unreviewed with bbox", "Test set", "All with bbox", "All unreviewed", "All chips"],
            index=0,
        )

        def _apply_filter(df: pd.DataFrame) -> pd.DataFrame:
            if filter_opt == "Unreviewed with bbox":
                return df[df.has_bbox & ~df.chip_stem.isin(dec_set)].reset_index(drop=True)
            if filter_opt == "Test set":
                return df[df.split == "test"].reset_index(drop=True)
            if filter_opt == "All with bbox":
                return df[df.has_bbox].reset_index(drop=True)
            if filter_opt == "All unreviewed":
                return df[~df.chip_stem.isin(dec_set)].reset_index(drop=True)
            return df.reset_index(drop=True)

        filtered = _apply_filter(manifest)

        st.caption(f"{len(filtered)} chips in current filter")
        st.divider()

        # Apply decisions
        st.subheader("Apply decisions")
        st.caption("Writes YOLO label files for all disqualified / adjusted chips.")
        if st.button("✅ Apply all decisions", type="primary", use_container_width=True):
            stats = apply_all_decisions()
            st.success(
                f"Done — {stats['disqualified']} disqualified, {stats['adjusted']} adjusted."
            )
            load_manifest.clear()

        # Export decisions CSV
        if not decisions.empty:
            st.download_button(
                "⬇ Download review_decisions.csv",
                data=decisions.to_csv(index=False).encode(),
                file_name="review_decisions.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ── no chips ───────────────────────────────────────────────────────────────
    if filtered.empty:
        st.success("🎉 All chips in this filter have been reviewed!")
        return

    # ── chip navigation ────────────────────────────────────────────────────────
    if "chip_idx" not in st.session_state:
        st.session_state.chip_idx = 0
    st.session_state.chip_idx = min(st.session_state.chip_idx, len(filtered) - 1)

    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 2, 1, 2])
    with nav_col1:
        if st.button("◀ Prev", use_container_width=True):
            st.session_state.chip_idx = max(0, st.session_state.chip_idx - 1)
            st.rerun()
    with nav_col2:
        idx = st.number_input(
            "Chip index", 0, len(filtered) - 1,
            value=st.session_state.chip_idx, step=1, label_visibility="collapsed",
        )
        if idx != st.session_state.chip_idx:
            st.session_state.chip_idx = int(idx)
            st.rerun()
    with nav_col3:
        if st.button("Next ▶", use_container_width=True):
            st.session_state.chip_idx = min(len(filtered) - 1, st.session_state.chip_idx + 1)
            st.rerun()
    with nav_col4:
        if st.button("⏭ Jump to first unreviewed", use_container_width=True):
            for i, row in filtered.iterrows():
                if row.chip_stem not in dec_set:
                    st.session_state.chip_idx = i
                    break
            st.rerun()

    chip = filtered.iloc[st.session_state.chip_idx]
    chip_stem = chip.chip_stem
    current_decision = dec_action.get(chip_stem, "unreviewed")

    # ── reset sliders when chip changes ───────────────────────────────────────
    if st.session_state.get("_slider_chip") != chip_stem:
        # Check if there's an existing adjusted decision to restore
        adj_row = decisions[decisions.chip_stem == chip_stem]
        if not adj_row.empty and adj_row.iloc[0].action == "adjusted":
            r = adj_row.iloc[0]
            x1_init, y1_init = int(r.new_x1), int(r.new_y1)
            x2_init, y2_init = int(r.new_x2), int(r.new_y2)
        else:
            bbox = parse_label_bbox(chip.lbl_path)
            if bbox:
                x1_init, y1_init, x2_init, y2_init = bbox
            else:
                x1_init, y1_init, x2_init, y2_init = 160, 160, 480, 480  # centre default
        st.session_state["sl_x1"] = x1_init
        st.session_state["sl_y1"] = y1_init
        st.session_state["sl_x2"] = x2_init
        st.session_state["sl_y2"] = y2_init
        st.session_state["_slider_chip"] = chip_stem

    # ── main display ───────────────────────────────────────────────────────────
    img_col, ctrl_col = st.columns([1, 1], gap="large")

    with ctrl_col:
        # Chip info
        status_icon = {"approved": "✅", "disqualified": "✗", "adjusted": "✏️"}.get(
            current_decision, "⬜"
        )
        st.markdown(
            f"**{chip_stem}** &nbsp; `{chip.split}` &nbsp; {status_icon} *{current_decision}*"
        )
        if chip.is_synthetic:
            st.warning("⚠️ Synthetic bbox — please verify and adjust.", icon="⚠️")

        orig_bbox = parse_label_bbox(chip.lbl_path)
        if orig_bbox:
            x1_o, y1_o, x2_o, y2_o = orig_bbox
            st.caption(
                f"Current label: x1={x1_o} y1={y1_o} x2={x2_o} y2={y2_o} "
                f"({x2_o-x1_o}×{y2_o-y1_o} px)"
            )
        else:
            st.caption("No bbox in label (negative chip)")

        st.divider()

        # Bbox sliders
        st.markdown("**Adjust bounding box** *(live preview on left)*")
        x1 = st.slider("Left  (x1)", 0, IMG_PX - 1, key="sl_x1", step=2)
        y1 = st.slider("Top   (y1)", 0, IMG_PX - 1, key="sl_y1", step=2)
        x2 = st.slider("Right (x2)", 1, IMG_PX, key="sl_x2", step=2)
        y2 = st.slider("Bottom (y2)", 1, IMG_PX, key="sl_y2", step=2)

        # Guard: x2 must be > x1, y2 > y1
        x2 = max(x2, x1 + 10)
        y2 = max(y2, y1 + 10)

        adjusted_bbox = (x1, y1, x2, y2)
        w, h = x2 - x1, y2 - y1
        st.caption(f"Adjusted: {w}×{h} px  ({w/IMG_PX*100:.1f}% × {h/IMG_PX*100:.1f}%)")

        st.divider()

        # Action buttons
        # Approve always saves current slider values.
        # If sliders were not moved it records "approved"; if moved it records "adjusted".
        bbox_was_changed = orig_bbox is not None and adjusted_bbox != orig_bbox

        if bbox_was_changed:
            st.info("Sliders moved — Approve will save the adjusted bbox.")

        btn1, btn2 = st.columns(2)
        with btn1:
            label = "✅ Approve + save bbox" if bbox_was_changed else "✅ Approve"
            if st.button(label, use_container_width=True, type="primary"):
                action = "adjusted" if bbox_was_changed else "approved"
                save_decision(chip_stem, action, adjusted_bbox)
                st.session_state.chip_idx = min(len(filtered) - 1, st.session_state.chip_idx + 1)
                st.session_state["_slider_chip"] = None
                st.rerun()
        with btn2:
            if st.button("✗ Disqualify", use_container_width=True):
                save_decision(chip_stem, "disqualified", None)
                st.session_state.chip_idx = min(len(filtered) - 1, st.session_state.chip_idx + 1)
                st.session_state["_slider_chip"] = None
                st.rerun()

    with img_col:
        # Show slider bbox in green, original label bbox in red (faint)
        try:
            img = render_chip(chip.img_path, adjusted_bbox, color="#00FF00")
            # Also draw original bbox faintly in red for comparison
            if orig_bbox and orig_bbox != adjusted_bbox:
                d = ImageDraw.Draw(img)
                d.rectangle(list(orig_bbox), outline="#FF4444", width=1)
        except Exception as exc:
            st.error(f"Could not load image: {exc}")
            return
        st.image(img, use_column_width=True)
        st.caption(
            "🟢 green = current slider bbox  ·  🔴 red outline = original label bbox"
        )


if __name__ == "__main__":
    main()
