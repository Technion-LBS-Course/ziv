#!/usr/bin/env python3
"""Remove the single-tab wrapper around the KPI section so st_folium renders immediately.

The Leaflet map inside st.tabs() measures its container at 0×0 while the tab is
hidden (even if it is the only tab), causing the heatmap not to appear until a
resize event fires.  Removing the wrapper renders the content directly inside
with tab_eda: — no hidden container, map shows on first load.
"""
from pathlib import Path

APP = Path(__file__).parent.parent / "app.py"
lines = APP.read_text(encoding="utf-8").splitlines(keepends=True)

# ── locate key line indices (0-based) ─────────────────────────────────────────
tab_def  = next(i for i, l in enumerate(lines) if "[tab_density] = st.tabs" in l)
with_ln  = next(i for i, l in enumerate(lines) if l.strip() == "with tab_density:")
block_end = next(i for i, l in enumerate(lines) if "# ── multi-modal routing simulator" in l)

print(f"tab_def={tab_def+1}, with_ln={with_ln+1}, block_end={block_end+1}")

new_lines = []
skip = {tab_def, with_ln}          # lines to drop entirely
# also drop the blank line immediately before the [tab_density] = ... line if it exists
if tab_def > 0 and lines[tab_def - 1].strip() == "":
    skip.add(tab_def - 1)

for i, line in enumerate(lines):
    if i in skip:
        continue

    if with_ln < i < block_end:
        # strip 4 leading spaces from the formerly-double-indented content
        if line.startswith("    "):        # 4+ spaces: remove first 4
            new_lines.append(line[4:])
        else:                              # blank line (just \n) — keep as-is
            new_lines.append(line)
    else:
        new_lines.append(line)

APP.write_text("".join(new_lines), encoding="utf-8")
print(f"Done. {len(lines)} -> {len(new_lines)} lines.")
