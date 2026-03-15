#!/usr/bin/env python3
"""
Visualize beam search trajectory to illustrate how samples evolve during the search.

Creates an interactive HTML visualization showing:
1. Intermediate states (decoded at checkpoint - what we SELECT and keep)
2. Look-ahead samples (originate from intermediate, used for reward evaluation)
3. Search tree: intermediate -> look-ahead (children); selection keeps intermediate

Usage:
    # Visualize a saved trajectory (from beam search with trajectory_dir set)
    python script_utils/plot/visualize_beam_search_trajectory.py --trajectory_dir path/to/trajectory --output trajectory_viz.html

    # Create demo visualization from a single PDB (for illustration without a full run)
    python script_utils/plot/visualize_beam_search_trajectory.py --demo --pdb path/to/protein.pdb --output demo_trajectory.html

    # Demo using bundled asset
    python script_utils/plot/visualize_beam_search_trajectory.py --demo --output demo_trajectory.html
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Add project root for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize beam search trajectory evolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default=None,
        help="Directory containing trajectory data (step_0/, step_1/, ..., trajectory_manifest.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="beam_search_trajectory.html",
        help="Output HTML file path",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Create demo visualization from a single PDB (simulates trajectory with small perturbations)",
    )
    parser.add_argument(
        "--pdb",
        type=str,
        default=None,
        help="PDB file for demo mode (default: use bundled 1ww1.cif from assets)",
    )
    parser.add_argument(
        "--max_samples_per_step",
        type=int,
        default=4,
        help="Max number of samples to show per step in overlay (default: 4)",
    )
    parser.add_argument(
        "--show_reward_chart",
        action="store_true",
        default=True,
        help="Include reward progression chart (default: True)",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Show what would be done without writing files",
    )
    return parser.parse_args()


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_trajectory(trajectory_dir: str) -> tuple[dict, list[dict]]:
    """Load trajectory from directory. Returns (step_data, all_nodes).

    step_data: (step, sample_type) -> list of sample entries for flat step-based view.
    all_nodes: flat list of nodes with step, sample_type, sample_id, parent_step, parent_sample_id,
        branch, beam, orig_sample, and entry (pdb_path, etc) for tree structure.
    """
    manifest_path = os.path.join(trajectory_dir, "trajectory_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    step_data = {}
    all_nodes = []

    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        step = int(row["step"])
        sample_type = row["sample_type"]
        pdb_path = os.path.join(trajectory_dir, row["pdb_path"])
        reward = row["total_reward"]
        try:
            reward = float(reward) if reward else None
        except ValueError:
            reward = None
        selected = row["selected"] == "1"
        kept = row["kept"] == "1"
        parent_id = int(row["parent_sample_id"])
        ois = row["origin_intermediate_sample_id"]
        origin_intermediate_id = int(ois) if ois else None

        entry = {
            "pdb_path": pdb_path,
            "total_reward": reward,
            "selected": selected,
            "kept": kept,
            "sample_type": sample_type,
            "branch": int(row["branch"]),
            "beam": int(row["beam"]),
            "orig_sample": int(row["orig_sample"]),
        }
        if origin_intermediate_id is not None:
            entry["origin_intermediate_sample_id"] = origin_intermediate_id

        key = (step, sample_type)
        if key not in step_data:
            step_data[key] = []
        local_idx = len(step_data[key])
        step_data[key].append(entry)
        manifest_sample_id = int(row.get("sample_id", local_idx))

        node = {
            "step": step,
            "sample_type": sample_type,
            "sample_id": manifest_sample_id,
            "parent_sample_id": parent_id,
            "origin_intermediate_sample_id": origin_intermediate_id,
            "kept": kept,
            "entry": entry,
            "branch": int(row["branch"]),
            "beam": int(row["beam"]),
            "orig_sample": int(row["orig_sample"]),
        }
        all_nodes.append(node)

    return step_data, all_nodes


def _create_demo_trajectory(pdb_path: str | None, output_dir: str) -> str:
    """Create synthetic trajectory from a PDB file and return trajectory_dir path."""
    try:
        from biotite.structure.io import load_structure, save_structure
    except ImportError as e:
        raise ImportError("biotite required for demo mode") from e

    if pdb_path is None:
        # Use bundled asset
        asset_path = _PROJECT_ROOT / "assets" / "binder_data" / "1ww1.cif"
        if not asset_path.exists():
            raise FileNotFoundError(
                f"No PDB provided and asset not found at {asset_path}. Use --pdb to specify a structure file."
            )
        pdb_path = str(asset_path)

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    struct = load_structure(pdb_path)
    # Handle AtomArrayStack (multi-model) - take first model
    try:
        from biotite.structure import AtomArrayStack

        if isinstance(struct, AtomArrayStack):
            struct = struct[0]
    except ImportError:
        pass
    # Get CA atoms only for simpler visualization
    if hasattr(struct, "atom_name"):
        ca_mask = struct.atom_name == "CA"
    else:
        ca_mask = np.array([a.atom_name == "CA" for a in struct])
    ca_struct = struct[ca_mask]
    if len(ca_struct) == 0:
        raise ValueError(f"No CA atoms found in {pdb_path}")

    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # Create 4 "steps" with increasing convergence (smaller perturbation each time)
    n_steps = 4
    sigma_start = 2.0
    sigma_end = 0.1

    manifest_path = os.path.join(output_dir, "trajectory_manifest.csv")
    with open(manifest_path, "w") as mf:
        mf.write(
            "step,end_step,sample_id,branch,beam,orig_sample,total_reward,selected,sample_type,parent_sample_id,origin_intermediate_sample_id,kept,pdb_path\n"
        )
        n_samples = 4
        for step in range(n_steps):
            t = step / max(n_steps - 1, 1)
            sigma = sigma_start * (1 - t) + sigma_end * t
            # Intermediate (demo: one per orig)
            step_int_dir = os.path.join(output_dir, f"step_{step}_intermediate")
            os.makedirs(step_int_dir, exist_ok=True)
            for sample_id in range(n_samples):
                coords = np.copy(ca_struct.coord)
                noise = np.random.normal(0, sigma * 0.5, coords.shape)
                coords += noise
                ca_struct_pert = ca_struct.copy()
                ca_struct_pert.coord = coords
                pdb_name = f"sample_{sample_id}.pdb"
                out_path = os.path.join(step_int_dir, pdb_name)
                save_structure(out_path, ca_struct_pert)
                parent = -1 if step == 0 else 0
                kept = 1 if sample_id == 0 else 0  # top one kept
                rel_path = os.path.join(f"step_{step}_intermediate", pdb_name)
                mf.write(
                    f"{step},{100 * step},{sample_id},0,{sample_id},0,,-1,intermediate,{parent},,{kept},{rel_path}\n"
                )
            # Lookahead
            step_dir = os.path.join(output_dir, f"step_{step}")
            os.makedirs(step_dir, exist_ok=True)
            for sample_id in range(n_samples):
                coords = np.copy(ca_struct.coord)
                noise = np.random.normal(0, sigma, coords.shape)
                coords += noise
                ca_struct_pert = ca_struct.copy()
                ca_struct_pert.coord = coords
                pdb_name = f"sample_{sample_id}.pdb"
                out_path = os.path.join(step_dir, pdb_name)
                save_structure(out_path, ca_struct_pert)
                reward = 0.3 + 0.6 * t + 0.1 * (1 - sample_id / n_samples)
                selected = 1 if sample_id == 0 else 0
                parent = -1 if step == 0 else 0
                origin_int = sample_id % n_samples
                rel_path = os.path.join(f"step_{step}", pdb_name)
                mf.write(
                    f"{step},{100 * (step + 1)},{sample_id},0,{sample_id},0,{reward:.4f},{selected},lookahead,{parent},{origin_int},{selected},{rel_path}\n"
                )

        # Final step
        step_dir = os.path.join(output_dir, "step_4_final")
        os.makedirs(step_dir, exist_ok=True)
        for sample_id in range(n_samples):
            coords = np.copy(ca_struct.coord)
            noise = np.random.normal(0, 0.05, coords.shape)
            coords += noise
            ca_struct_pert = ca_struct.copy()
            ca_struct_pert.coord = coords
            pdb_name = f"sample_{sample_id}.pdb"
            out_path = os.path.join(step_dir, pdb_name)
            save_structure(out_path, ca_struct_pert)
            parent = 0
            rel_path = os.path.join("step_4_final", pdb_name)
            mf.write(f"4,500,{sample_id},-1,{sample_id},0,,1,final,{parent},,1,{rel_path}\n")

    logger.info("Created demo trajectory at %s", output_dir)
    return output_dir


def _read_pdb_content(path: str) -> str:
    """Read PDB file content. Handle CIF by converting if needed."""
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read()


def _pdb_to_pdb_string(path: str) -> str:
    """Get PDB-format string. If CIF, try to convert via biotite."""
    content = _read_pdb_content(path)
    if path.endswith(".cif") and "ATOM" not in content[:100]:
        try:
            import biotite.structure.io as strucio
            from biotite.structure.io.pdb import PDBFile

            struct = strucio.load_structure(path)
            pdb_file = PDBFile()
            pdb_file.set_structure(struct)
            return pdb_file.get_string()
        except Exception:
            pass
    return content


def _hex_color_for_step(step: int, n_steps: int) -> str:
    """Return hex color that varies from blue (early) to green (late)."""
    t = step / max(n_steps - 1, 1)
    r = int(50 + 200 * t)
    g = int(100 + 155 * t)
    b = int(200 - 150 * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_tree_structure(all_nodes: list, step_data: dict) -> list:
    """Build hierarchical tree: intermediate expands beam->(beam*n_branch); kept selects beam.
    Lookahead originates from intermediate. Order same depth by beam_idx, then branch_idx.
    """
    key_to_idx: dict[tuple[int, str, int], int] = {}
    for idx, node in enumerate(all_nodes):
        key = (node["step"], node["sample_type"], node["sample_id"])
        key_to_idx[key] = idx

    steps_with_lookahead = sorted({k[0] for k in step_data if k[1] == "lookahead"})
    last_lookahead_step = max(steps_with_lookahead) if steps_with_lookahead else -1

    for idx, node in enumerate(all_nodes):
        node["node_idx"] = idx
        node["intermediate_children"] = []  # Next-step intermediates (expansion)
        node["lookahead_children"] = []  # Lookahead originating from this intermediate
        step, stype = node["step"], node["sample_type"]
        parent_idx = None

        if stype == "lookahead":
            origin_id = node.get("origin_intermediate_sample_id")
            if origin_id is not None:
                parent_key = (step, "intermediate", origin_id)
                parent_idx = key_to_idx.get(parent_key)
                if parent_idx is not None:
                    all_nodes[parent_idx]["lookahead_children"].append(idx)
        elif stype in ("intermediate", "final"):
            parent_id = node["parent_sample_id"]
            if parent_id >= 0:
                prev_step = last_lookahead_step if stype == "final" else step - 1
                lookahead_key = (prev_step, "lookahead", parent_id)
                lookahead_idx = key_to_idx.get(lookahead_key)
                if lookahead_idx is not None:
                    origin_id = all_nodes[lookahead_idx].get("origin_intermediate_sample_id")
                    if origin_id is not None:
                        parent_key = (prev_step, "intermediate", origin_id)
                        parent_idx = key_to_idx.get(parent_key)
                        if parent_idx is not None:
                            all_nodes[parent_idx]["intermediate_children"].append(idx)

        node["parent_idx"] = parent_idx

    # Sort children by beam_idx, then branch_idx (states with same beam_idx expanded from same parent)
    for node in all_nodes:

        def sort_key(idx: int) -> tuple:
            n = all_nodes[idx]
            return (n.get("beam", 0), n.get("branch", 0))

        node["intermediate_children"] = sorted(node["intermediate_children"], key=sort_key)
        node["lookahead_children"] = sorted(node["lookahead_children"], key=sort_key)

    return all_nodes


def _create_html(
    trajectory_dir: str,
    output_path: str,
    step_data: dict,
    all_nodes: list | None = None,
    max_samples_per_step: int = 4,
    show_reward_chart: bool = True,
    dryrun: bool = False,
) -> None:
    """Generate interactive HTML visualization using 3Dmol.js with search tree view."""
    # Flatten step_data for legacy step/sample dropdown: use (step, sample_type) keys
    sorted(set(k[0] for k in step_data))
    if all_nodes:
        all_nodes = _build_tree_structure(all_nodes, step_data)

    # Build HTML
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Beam Search Trajectory Visualization</title>
<style>
body { font-family: system-ui, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
h1 { color: #0f3460; }
.container { display: flex; flex-direction: column; gap: 20px; max-width: 1200px; margin: 0 auto; }
.controls {
  display: flex; gap: 15px; align-items: center; flex-wrap: wrap;
  padding: 16px; background: #16213e; border-radius: 8px;
  flex-shrink: 0; position: relative; z-index: 10;
}
select, button { padding: 8px 12px; font-size: 14px; border-radius: 6px; border: 1px solid #444; background: #2d2d44; color: #eee; }
button { cursor: pointer; }
button:hover { background: #3d3d54; }
.main-row { display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start; }
.tree-panel {
  flex: 0 0 280px; max-height: 500px; overflow-y: auto;
  padding: 12px; background: #16213e; border-radius: 8px;
  font-size: 13px;
}
.tree-panel h3 { margin: 0 0 10px 0; color: #e94560; }
.tree-node { padding: 4px 8px; cursor: pointer; border-radius: 4px; margin: 2px 0; }
.tree-node:hover { background: #2d2d44; }
.tree-node.selected { background: #0f3460; }
.tree-node.selected span { color: #fff; }
.tree-node.kept { border-left: 3px solid #22c55e; }
.tree-node.discarded { border-left: 3px solid #ef4444; opacity: 0.8; }
.tree-node-intermediate { color: #60a5fa; }
.tree-node-lookahead { color: #fbbf24; }
.tree-node-final { color: #a78bfa; }
.tree-children { margin-left: 16px; border-left: 1px solid #444; padding-left: 8px; }
.tree-children .from-label { font-size: 11px; color: #94a3b8; margin: 4px 0 2px 0; }
.tree-toggle { cursor: pointer; margin-right: 4px; font-size: 10px; opacity: 0.8; }
.tree-toggle:hover { opacity: 1; }
.tree-lookahead-children .tree-node { opacity: 0.9; }
.viewer-wrap {
  flex: 1; min-width: 400px; position: relative; height: 500px;
  border: 1px solid #333; border-radius: 8px; background: #0d1117;
}
#viewer { width: 100%; height: 500px; display: block; }
.info { padding: 12px; background: #16213e; border-radius: 8px; flex-shrink: 0; }
.chart { height: 200px; margin-top: 15px; flex-shrink: 0; }
.step-label { font-weight: bold; color: #e94560; }
.hint { color: #94a3b8; font-size: 13px; margin: -10px 0 0 0; }
</style>
</head>
<body>
<div class="container">
<h1>Beam Search Trajectory</h1>
<p>Beam search tree: at each step, beam_size intermediates expand to beam_size×n_branch; top beam_size kept. Look-ahead estimates rewards per intermediate.</p>
<div class="controls">
<label><strong>Orig sample:</strong> <select id="origSelect"></select></label>
<label><input type="checkbox" id="showLookahead"> Show look-ahead samples</label>
<label><input type="checkbox" id="expandChildren" checked> Expand kept children</label>
<button id="overlayBtn">Overlay all steps</button>
<button id="expandAllBtn">Expand all</button>
<button id="collapseAllBtn">Collapse all</button>
</div>
<p class="hint">Click a node in the tree to view its structure. Toggle ▼/▶ to expand/collapse.</p>
<div class="main-row">
<div class="tree-panel"><h3>Search tree (▼ expand / ▶ collapse kept nodes; same beam_idx = same parent)</h3><div id="treeContainer"></div></div>
<div class="viewer-wrap"><div id="viewer"></div></div>
</div>
<div class="info" id="info"></div>
""")

    if show_reward_chart:
        html_parts.append("""<div class="chart" id="rewardChart"></div>""")

    html_parts.append("</div>")
    html_parts.append("""
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<script>
var viewer = null;

function initViewer() {
    var mol = window["3Dmol"] || window["$3Dmol"];
    if (!mol) { console.error("3Dmol not loaded"); return; }
    var elem = document.getElementById("viewer");
    if (!elem) { console.error("viewer element not found"); return; }
    viewer = mol.createViewer(elem, { backgroundColor: "white" });
    viewer.setStyle({}, { cartoon: { color: "spectrum" } });
    viewer.zoomTo();
    if (typeof viewer.render === "function") viewer.render();
}

function loadStructure(pdbContent, color) {
    if (!viewer || !pdbContent) return;
    viewer.addModel(pdbContent, "pdb");
    const n = viewer.getModel().length;
    viewer.setStyle({ model: n - 1 }, { cartoon: { color: color || "spectrum" } });
    viewer.zoomTo();
}

function overlayAllSteps() {
    if (!viewer) return;
    viewer.clear();
    const colors = ["red", "orange", "yellow", "green", "blue"];
    stepList.forEach(function(stepIdx) {
        if (overlayData[stepIdx]) {
            viewer.addModel(overlayData[stepIdx], "pdb");
            const n = viewer.getModel().length;
            viewer.setStyle({ model: n - 1 }, { cartoon: { color: colors[stepIdx % colors.length], opacity: 0.7 } });
        }
    });
    viewer.zoomTo();
}

document.getElementById("overlayBtn").onclick = overlayAllSteps;
</script>
</body>
</html>""")

    # Build overlay_data (one structure per step/stype for overlay) and step_options
    step_keys = sorted(
        step_data.keys(),
        key=lambda k: (
            k[0],
            {"intermediate": 0, "lookahead": 1, "final": 2}.get(k[1], 1),
        ),
    )
    overlay_data = {}
    step_options = []
    node_data = []  # node_idx -> {pdb, reward, selected, label}

    def _js_escape(s: str) -> str:
        return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${").replace("\r", "")

    for step, stype in step_keys:
        samples = step_data[(step, stype)]
        disp_key = f"{step}_{stype}"
        step_options.append(disp_key)
        for i, s in enumerate(samples):
            if disp_key not in overlay_data:
                pdb_path = s["pdb_path"]
                pdb_content = _read_pdb_content(pdb_path)
                if "ATOM" not in pdb_content and "HETATM" not in pdb_content:
                    pdb_content = _pdb_to_pdb_string(pdb_path)
                if pdb_content.strip():
                    overlay_data[disp_key] = _js_escape(pdb_content)
                    break

    # Build node_data and tree structure for tree panel (when all_nodes available)
    tree_data = None
    orig_samples = []
    if all_nodes:
        for node in all_nodes:
            ent = node["entry"]
            pdb_path = ent["pdb_path"]
            pdb_content = _read_pdb_content(pdb_path)
            if "ATOM" not in pdb_content and "HETATM" not in pdb_content:
                pdb_content = _pdb_to_pdb_string(pdb_path)
            escaped = _js_escape(pdb_content)
            br = node["branch"]
            bm = node["beam"]
            orig = node["orig_sample"]
            stype = node["sample_type"]
            kept = node.get("kept", ent.get("kept", True))
            sel = ent.get("selected", False)
            origin_id = node.get("origin_intermediate_sample_id")
            if stype == "intermediate":
                label = f"step {node['step']}, beam {bm}, branch {br}"
            elif stype == "lookahead":
                label = f"step {node['step']}, beam {bm}, branch {br}, lookahead"
            else:
                label = f"step {node['step']}, beam {bm}, final"
            if orig >= 0:
                label += f" [orig={orig}]"
            node_data.append(
                {
                    "pdb": escaped,
                    "reward": ent.get("total_reward"),
                    "selected": sel,
                    "kept": kept,
                    "label": label,
                    "sample_type": stype,
                    "orig_sample": orig,
                    "origin_intermediate_sample_id": origin_id,
                }
            )
        tree_data = [
            {
                "idx": n["node_idx"],
                "label": node_data[i]["label"],
                "selected": node_data[i]["selected"],
                "kept": node_data[i]["kept"],
                "reward": node_data[i].get("reward"),
                "sample_type": node_data[i].get("sample_type", n["sample_type"]),
                "intermediate_children": n.get("intermediate_children", []),
                "lookahead_children": n.get("lookahead_children", []),
                "parent_idx": n.get("parent_idx"),
                "orig_sample": node_data[i]["orig_sample"],
                "beam": n.get("beam", -1),
                "branch": n.get("branch", -1),
            }
            for i, n in enumerate(all_nodes)
        ]
        orig_samples = sorted(set(t["orig_sample"] for t in tree_data))

    # Rewrite the HTML to inject data
    # We'll use a template approach - replace placeholder with JSON
    import json

    full_html = "".join(html_parts)
    # Inject data before </script> in the script tag
    data_script = f"""
var overlayData = {json.dumps(overlay_data)};
var stepList = {json.dumps(step_options)};
var nodeData = {json.dumps(node_data) if node_data else "[]"};
var treeData = {json.dumps(tree_data) if tree_data else "null"};
var origSamples = {json.dumps(orig_samples)};

function loadNode(idx) {{
    if (!nodeData[idx] || !viewer) return;
    viewer.clear();
    viewer.addModel(nodeData[idx].pdb, "pdb");
    viewer.setStyle({{}}, {{ cartoon: {{ color: "spectrum" }} }});
    viewer.zoomTo();
    if (typeof viewer.render === "function") viewer.render();
    var n = nodeData[idx];
    var selText = n.sample_type === "intermediate" ? (n.kept ? " [Kept]" : " [Discarded]") : (n.selected ? " [Selected]" : " [Pruned]");
    document.getElementById("info").innerHTML = "<span class='step-label'>" + (n.label || "Node " + idx) + "</span>" +
        (n.reward != null ? " &ndash; Reward: " + n.reward.toFixed(4) : "") +
        " <strong>" + selText + "</strong>";
    document.querySelectorAll(".tree-node").forEach(function(el) {{ el.classList.remove("selected"); }});
    var sel = document.querySelector("[data-node-idx='" + idx + "']");
    if (sel) sel.classList.add("selected");
}}

var expandedNodes = {{}};

function renderNode(n, parentEl, origFilter, showLookahead, expandChildren, depth) {{
    if (origFilter !== null && n.orig_sample !== origFilter) return;
    var div = document.createElement("div");
    var typeClass = "tree-node-" + (n.sample_type || "intermediate");
    div.className = "tree-node " + typeClass + (n.kept ? " kept" : " discarded") + (n.selected ? " selected" : "");
    div.setAttribute("data-node-idx", n.idx);
    var hasIntChildren = (n.intermediate_children || []).length > 0;
    var hasLookChildren = showLookahead && (n.lookahead_children || []).length > 0;
    var isExpanded = expandChildren && (expandedNodes[n.idx] !== false);
    if (expandChildren && expandedNodes[n.idx] === undefined) expandedNodes[n.idx] = true;
    var toggle = "";
    if (hasIntChildren && n.kept && expandChildren) {{
        toggle = "<span class='tree-toggle'>" + (isExpanded ? "▼" : "▶") + "</span>";
    }}
    var suffix = n.sample_type === "intermediate" ? (n.kept ? " ✓" : " ✗") : (n.selected ? " ✓" : "");
    if (n.sample_type === "lookahead" && n.reward != null) suffix += " R=" + n.reward.toFixed(3);
    div.innerHTML = toggle + "<span class='tree-label'>" + (n.label || "Node " + n.idx) + suffix + "</span>";
    div.querySelector(".tree-toggle")?.addEventListener("click", function(e) {{ e.stopPropagation(); expandedNodes[n.idx] = !isExpanded; refreshTree(); }});
    div.querySelector(".tree-label")?.addEventListener("click", function() {{ loadNode(n.idx); }});
    parentEl.appendChild(div);
    if (hasIntChildren && n.kept && expandChildren && isExpanded) {{
        var childDiv = document.createElement("div");
        childDiv.className = "tree-children tree-int-children";
        var childNodes = (n.intermediate_children || []).map(function(ci) {{ return treeData[ci]; }}).filter(Boolean);
        childNodes.filter(function(c) {{ return c && (origFilter === null || c.orig_sample === origFilter); }}).forEach(function(c) {{
            renderNode(c, childDiv, origFilter, showLookahead, expandChildren, (depth || 0) + 1);
        }});
        if (childDiv.children.length > 0) parentEl.appendChild(childDiv);
    }}
    if (hasLookChildren) {{
        var laDiv = document.createElement("div");
        laDiv.className = "tree-children tree-lookahead-children";
        var laLabel = document.createElement("div");
        laLabel.className = "from-label";
        laLabel.textContent = "look-ahead (rewards):";
        laDiv.appendChild(laLabel);
        (n.lookahead_children || []).map(function(ci) {{ return treeData[ci]; }}).filter(function(c) {{ return c && (origFilter === null || c.orig_sample === origFilter); }}).forEach(function(c) {{
            renderNode(c, laDiv, origFilter, false, expandChildren, (depth || 0) + 1);
        }});
        if (laDiv.children.length > 1) parentEl.appendChild(laDiv);
    }}
}}

function refreshTree() {{
    var tc = document.getElementById("treeContainer");
    if (!tc) return;
    tc.innerHTML = "";
    var origVal = document.getElementById("origSelect")?.value;
    var origFilter = origVal === "" || origVal === "all" ? null : parseInt(origVal, 10);
    var showLookahead = document.getElementById("showLookahead")?.checked === true;
    var expandChildren = document.getElementById("expandChildren")?.checked === true;
    if (treeData && treeData.length > 0) {{
        var roots = treeData.filter(function(n) {{ return n.parent_idx == null && n.sample_type === "intermediate"; }});
        if (origFilter !== null) roots = roots.filter(function(n) {{ return n.orig_sample === origFilter; }});
        roots.sort(function(a,b) {{ return (a.orig_sample - b.orig_sample) || (a.beam - b.beam) || (a.branch - b.branch); }});
        roots.forEach(function(n) {{ renderNode(n, tc, origFilter, showLookahead, expandChildren, 0); }});
    }}
}}

window.onload = function() {{
    const origSel = document.getElementById("origSelect");
    if (origSel && origSamples && origSamples.length > 0) {{
        var allOpt = document.createElement("option");
        allOpt.value = "all";
        allOpt.text = "All orig_samples";
        origSel.add(allOpt);
        origSamples.forEach(function(o) {{
            var opt = document.createElement("option");
            opt.value = o;
            opt.text = "Orig " + o;
            origSel.add(opt);
        }});
        origSel.onchange = function() {{ refreshTree(); }};
    }}
    document.getElementById("showLookahead")?.addEventListener("change", refreshTree);
    document.getElementById("expandChildren")?.addEventListener("change", refreshTree);
    document.getElementById("expandAllBtn")?.addEventListener("click", function() {{
        if (!treeData) return;
        treeData.forEach(function(n) {{ if ((n.intermediate_children || []).length > 0 && n.kept) expandedNodes[n.idx] = true; }});
        refreshTree();
    }});
    document.getElementById("collapseAllBtn")?.addEventListener("click", function() {{
        if (!treeData) return;
        treeData.forEach(function(n) {{ if ((n.intermediate_children || []).length > 0) expandedNodes[n.idx] = false; }});
        refreshTree();
    }});
    initViewer();
    refreshTree();
}};
"""

    # The original script had inline handlers - we need to integrate
    # Simpler: use a data attribute and a complete script
    full_html = full_html.replace(
        '<script src="https://3dmol.org/build/3Dmol-min.js"></script>',
        '<script src="https://3dmol.org/build/3Dmol-min.js"></script>\n<script>\n' + data_script + "\n</script>",
    )

    # Script uses 3Dmol.js API for structure viewing
    # Let me check - py3Dmol is for Python. For standalone HTML we use 3Dmol.js directly.
    # The viewer creation: $3Dmol.createViewer - yes that's the JS API.

    # Add reward chart if requested
    if show_reward_chart and step_data:
        rewards_by_step = {}
        for step, stype in step_keys:
            rewards = [s["total_reward"] for s in step_data[(step, stype)] if s.get("total_reward") is not None]
            if rewards:
                key = f"{step}_{stype}"
                rewards_by_step[key] = {
                    "min": float(min(rewards)),
                    "max": float(max(rewards)),
                    "mean": float(np.mean(rewards)),
                }

        chart_data = json.dumps(rewards_by_step)
        chart_script = f"""
var rewardData = {chart_data};
var chartDiv = document.getElementById("rewardChart");
if (chartDiv && Object.keys(rewardData).length > 0) {{
    chartDiv.innerHTML = "<p>Reward range per step: " + JSON.stringify(rewardData) + "</p>";
}}
"""
        full_html = full_html.replace("</div>\n</div>", f"<script>{chart_script}</script>\n</div>\n</div>")

    if not dryrun:
        with open(output_path, "w") as f:
            f.write(full_html)
        logger.info("Wrote %s", output_path)
    else:
        n_samples = sum(len(v) for v in step_data.values())
        logger.info(
            "[DRYRUN] Would write HTML to %s (%d step-types, %d total samples)",
            output_path,
            len(step_keys),
            n_samples,
        )


def main() -> int:
    _setup_logging()
    args = _parse_args()

    if args.demo:
        if args.trajectory_dir:
            logger.warning("--trajectory_dir ignored when --demo is set")
        traj_dir = os.path.join(os.path.dirname(args.output) or ".", "demo_trajectory_temp")
        traj_dir = os.path.abspath(traj_dir)
        if not args.dryrun:
            traj_dir = _create_demo_trajectory(args.pdb, traj_dir)
        else:
            logger.info(
                "[DRYRUN] Would create demo trajectory from %s",
                args.pdb or "assets/binder_data/1ww1.cif",
            )
            return 0
    elif args.trajectory_dir:
        traj_dir = os.path.abspath(args.trajectory_dir)
        if not os.path.isdir(traj_dir):
            logger.error("Trajectory directory not found: %s", traj_dir)
            return 1
    else:
        logger.error("Provide --trajectory_dir or --demo")
        return 1

    step_data, all_nodes = _load_trajectory(traj_dir)
    output_path = os.path.abspath(args.output)

    _create_html(
        trajectory_dir=traj_dir,
        output_path=output_path,
        step_data=step_data,
        all_nodes=all_nodes,
        max_samples_per_step=args.max_samples_per_step,
        show_reward_chart=args.show_reward_chart,
        dryrun=args.dryrun,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
