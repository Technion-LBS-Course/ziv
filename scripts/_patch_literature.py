#!/usr/bin/env python3
"""Replace the Literature tab content with 4 curated papers + new layout."""
from pathlib import Path

APP = Path(__file__).parent.parent / "app.py"
lines = APP.read_text(encoding="utf-8").splitlines(keepends=True)

# Find splice boundaries (0-indexed)
START = next(i for i, l in enumerate(lines) if "    _PAPERS = [" in l)
END   = next(i for i, l in enumerate(lines) if "# ── TAB 3 · Market Survey" in l)

print(f"Replacing lines {START+1}–{END} with new Literature tab content.")

NEW = '''\
    _PAPERS = [
        {
            "num": 1,
            "authors": "O\'Reilly, P.E., Rahimi, R.A., Marques, J.L.R., & Babadopulos, M.A.F.A.L.",
            "year": 2024,
            "title": "Vertiport ventures: assessing operational feasibility for eVTOL integration in São Paulo\'s helipad and heliport infrastructure",
            "journal": "Journal of Marketing Analytics",
            "vol_issue": "Vol. 12, pp. 873–884",
            "doi": "10.1057/s41270-024-00323-0",
            "doi_url": "https://doi.org/10.1057/s41270-024-00323-0",
            "field": "eVTOL Infrastructure & Site Scoring",
            "summary": (
                "Evaluates whether eVTOL aircraft (4+ passengers, 50 km+ range) can be integrated into "
                "São Paulo\'s extensive helicopter infrastructure — the world\'s busiest helicopter city. "
                "Analyses site suitability of existing helipads and heliports across the metropolitan region, "
                "finding that the dense rooftop pad network provides a structurally advantageous starting point. "
                "Identifies key gaps in infrastructure dimensions, regulatory alignment, and charging logistics "
                "that must be resolved before commercial eVTOL service can launch."
            ),
            "skyroute_benefit": (
                "São Paulo\'s helipad-reuse playbook validates SkyRoute\'s HIE approach of scoring existing FAA "
                "helipads as vertiport candidates — and quantifies the operational upgrade gaps that feed "
                "the scoring model."
            ),
        },
        {
            "num": 2,
            "authors": "Zhang, Y., Yang, C., Xi, H., Peng, S., Yang, J., Gan, M., Liu, X., & Ai, R.",
            "year": 2026,
            "title": "Air-ground multimodal transport planning for joint passenger mobility and parcel delivery: integration of drones, aircraft, and ground vehicles",
            "journal": "Transportation Research Part E: Logistics and Transportation Review",
            "vol_issue": "Vol. 210, Article 104825",
            "doi": "10.1016/j.tre.2026.104825",
            "doi_url": "https://doi.org/10.1016/j.tre.2026.104825",
            "field": "Multi-Modal AAM Routing & Optimisation",
            "summary": (
                "Formulates a joint optimisation model for multimodal transport that simultaneously routes passengers "
                "and parcels using drones, fixed-wing/rotary aircraft, and ground vehicles. "
                "The model integrates air-ground transfer nodes (analogous to vertiports/helipads) with last-mile "
                "ground connections, optimising vehicle types, transfer schedules, and fleet allocation in a unified "
                "mathematical programme. "
                "Demonstrates that coordinated multi-fleet planning significantly reduces total transport time and cost "
                "compared to single-mode or sequentially planned operations, with explicit treatment of passenger "
                "access/egress legs and cargo hand-off at intermodal nodes."
            ),
            "skyroute_benefit": (
                "The joint passenger-parcel multimodal optimisation framework directly parallels SkyRoute\'s routing "
                "architecture — replacing drones+fixed-wing with eVTOL+helicopter and cargo transfers with "
                "rideshare/subway connections. Provides rigorous mathematical grounding for the transfer-node-based "
                "routing that HIE-validated helipads feed into, and supports the multimodal comparison table "
                "already built into the app."
            ),
        },
        {
            "num": 3,
            "authors": "Singh, R., Puhl, R.B., Dhakal, K., & Sornapudi, S.",
            "year": 2025,
            "title": "Few-Shot Adaptation of Grounding DINO for Agricultural Domain",
            "journal": "arXiv preprint",
            "vol_issue": "arXiv:2504.07252",
            "doi": "10.48550/arXiv.2504.07252",
            "doi_url": "https://doi.org/10.48550/arXiv.2504.07252",
            "field": "ML: Promptable Object Detection in Aerial Imagery",
            "summary": (
                "Adapts Grounding DINO — an open-set, text-prompted object detector — for aerial and agricultural "
                "remote-sensing imagery using few-shot learning. "
                "Removes the BERT text encoder and replaces it with a lightweight trainable text embedding, "
                "substantially reducing the model\'s parameter count and adaptation cost. "
                "The resulting few-shot variant achieves up to 24% higher mAP than fully fine-tuned YOLO baselines "
                "on agricultural datasets, and outperforms prior state-of-the-art by ~10% on remote-sensing "
                "object-detection benchmarks under low-data conditions — demonstrating Grounding DINO\'s practical "
                "viability for promptable detection in overhead imagery without large labelled datasets."
            ),
            "skyroute_benefit": (
                "Directly underpins HIE Phase 1: confirms that Grounding DINO generalises to overhead/satellite "
                "imagery via text-prompt detection without helipad-specific training data. "
                "The few-shot mechanism means a small number of verified helipad examples is sufficient to "
                "guide the detector — critical given the scarcity of labelled helipad chips in public datasets."
            ),
        },
        {
            "num": 4,
            "authors": "Eyinade, J.A., & Ademusire, A.J.",
            "year": 2025,
            "title": "GeoLLMs in action: A systematic review of multimodal models for satellite image captioning and geospatial understanding",
            "journal": "Open Access Research Journal of Science and Technology",
            "vol_issue": "Vol. 14, Issue 2, pp. 049–064",
            "doi": "10.53022/oarjst.2025.14.2.0093",
            "doi_url": "https://doi.org/10.53022/oarjst.2025.14.2.0093",
            "field": "ML: LLM for Geospatial & Satellite Understanding",
            "summary": (
                "Systematic review of 42 peer-reviewed studies (2020–2025) on multimodal large language models "
                "applied to geospatial tasks: semantic segmentation, satellite image captioning, and spatial "
                "question answering. "
                "Identifies three dominant architectural patterns: frozen vision encoders with language adapters, "
                "end-to-end fine-tuned transformers, and retrieval-augmented hybrid systems. "
                "Surfaces persistent challenges — geographic generalisation, temporal reasoning, lack of standardised "
                "benchmarks, underrepresentation of non-Western regions — and recommends geometry-aware embeddings "
                "and multilingual fine-tuning as priority future directions."
            ),
            "skyroute_benefit": (
                "Provides the academic umbrella for HIE Phase 2: querying an LLM (Gemini/GPT-4o with search "
                "grounding) about a helipad\'s operational status from its name and location. "
                "The retrieval-augmented hybrid pattern identified in the review is exactly the architecture "
                "used — OSM name + coordinates as retrieval keys, LLM as the reasoning layer — and the review\'s "
                "benchmarks offer a framework for evaluating HIE Phase 2 classification accuracy."
            ),
        },
    ]

    # ── quick-reference table ─────────────────────────────────────────────────
    st.markdown("#### Quick Reference")
    _tbl_md = (
        "| # | Article | Field of Relevance | DOI |\\n"
        "|---|---------|-------------------|-----|\\n"
    )
    for _p in _PAPERS:
        _a = _p["authors"].split(",")[0] + (" et al." if "," in _p["authors"] else "")
        _t = _p["title"][:62] + "…" if len(_p["title"]) > 62 else _p["title"]
        _tbl_md += f"| {_p[\'num\']} | **{_a} ({_p[\'year\']})** — {_t} | {_p[\'field\']} | [🔗 DOI]({_p[\'doi_url\']}) |\\n"
    st.markdown(_tbl_md)

    st.divider()
    st.markdown("#### Paper Summaries")

    for p in _PAPERS:
        short_auth = p["authors"].split(",")[0] + (" et al." if "," in p["authors"] else "")
        _ttl = p["title"]
        label = f"[{p[\'num\']}]  {short_auth} ({p[\'year\']}) — {_ttl[:65]}{'…' if len(_ttl)>65 else ''}"
        with st.expander(label):
            st.markdown(f"**{p[\'authors\']} ({p[\'year\']})**")
            st.markdown(f"*{p[\'title\']}*")
            st.markdown(f"📚 **Source:** *{p[\'journal\']}*, {p[\'vol_issue\']}")
            st.markdown(f"🔗 **DOI:** [{p[\'doi\']}]({p[\'doi_url\']})")
            st.divider()
            st.markdown("**Abstract summary**")
            st.markdown(p["summary"])
            st.markdown(
                f"<div style=\'background:#071a2e;border-left:4px solid #22c55e;"
                f"border-radius:6px;padding:9px 14px;margin-top:10px;font-size:13px\'>"
                f"<span style=\'color:#22c55e;font-weight:700\'>✈ SkyRoute benefit:&nbsp;</span>"
                f"<span style=\'color:#22c55e\'>{p[\'skyroute_benefit\']}</span></div>",
                unsafe_allow_html=True,
            )


'''

new_lines = lines[:START] + [NEW] + lines[END:]
APP.write_text("".join(new_lines), encoding="utf-8")
print(f"Done. {len(lines)} -> {len(new_lines)} lines.")
