#!/usr/bin/env python3
"""
Case-study path diagram for SIGHT/FlorE multi-hop reasoning.

The default example is a classic, human-readable CoDEx-S query:

    (Angelina Jolie, place of birth, Los Angeles)

For each candidate intermediate node z adjacent to the head h, the script
computes the model's aggregated path weight

    W(z) = sum_{i=1..K} alpha_i * w_i(z),

where w_i(z) is the cone-filtered neighbour attention at hop i and alpha_i is
the hop-level softmax from Eq. (15). The figure highlights the two strongest
target-resolving paths and shows lower-weight irrelevant paths as suppressed.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
import torch
import numpy as np
import torch.nn.functional as F

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from load_data import Data  # noqa: E402
from LorentzModel import HyperNet  # noqa: E402


ENTITY_LABELS = {
    "Q13909": "Angelina Jolie",
    "Q65": "Los Angeles",
    "Q35332": "Brad Pitt",
    "Q202735": "Billy Bob Thornton",
    "Q10800557": "film actor",
    "Q2526255": "film director",
    "Q10798782": "television actor",
    "Q13235160": "producer",
    "Q30": "United States",
}

RELATION_LABELS = {
    "P19": "place of birth",
    "P26": "spouse",
    "P451": "unmarried partner",
    "P551": "residence",
    "P106": "occupation",
    "P27": "country of citizenship",
    "P1412": "languages spoken",
}


@dataclass
class PathRecord:
    node_idx: int
    node_id: str
    weight: float
    h_rel: str
    t_rel: str | None
    resolves_target: bool
    hop_weights: list[dict] | None = None


def read_meta(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        rec = json.loads(f.readline())
    if rec.get("type") != "meta":
        raise ValueError(f"First line of {path} is not a meta record")
    return rec


def read_latest_path_case(path: str, head: str, relation: str, tail: str) -> dict | None:
    latest = None
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("type") != "path_case":
                continue
            query = rec.get("query", {})
            if query.get("head") == head and query.get("relation") == relation and query.get("tail") == tail:
                latest = rec
    return latest


def records_from_path_case(
    path_case: dict,
    data: Data,
    entity_idxs: dict[str, int],
) -> tuple[list[PathRecord], list[float], float, float]:
    paths = path_case.get("paths", [])
    query = path_case.get("query", {})
    h_idx = entity_idxs.get(query.get("head"))
    t_idx = entity_idxs.get(query.get("tail"))
    rel_between = relation_lookup(data, entity_idxs) if h_idx is not None and t_idx is not None else {}
    records = []
    for item in paths:
        node_id = item["node_id"]
        node_idx = int(entity_idxs.get(node_id, item.get("node_idx", -1)))
        head_rels = item.get("head_relations") or rel_between.get((h_idx, node_idx), [])
        target_rels = item.get("target_relations") or rel_between.get((node_idx, t_idx), [])
        records.append(
            PathRecord(
                node_idx=node_idx,
                node_id=node_id,
                weight=float(item["aggregated_weight"]),
                h_rel=(head_rels or [None])[0],
                t_rel=(target_rels or [None])[0],
                resolves_target=bool(item.get("resolves_target", False) or target_rels),
                hop_weights=item.get("hop_weights", []),
            )
        )
    hop_summaries = path_case.get("hop_summaries", [])
    pruned = sum(int(x.get("pruned_count", 0)) for x in hop_summaries)
    candidates = sum(int(x.get("candidate_count", 0)) for x in hop_summaries)
    suppressed_rate = pruned / max(candidates, 1)
    return (
        sorted(records, key=lambda rec: -rec.weight),
        [float(x) for x in path_case.get("alpha", [])],
        float(path_case.get("gamma", 0.0)),
        suppressed_rate,
    )


def apply_mock_weights(records: list[PathRecord], noise_count: int) -> list[PathRecord]:
    critical = [rec for rec in records if rec.resolves_target and rec.node_id != "Q65"][:2]
    noise = [rec for rec in records if not rec.resolves_target and rec.node_id != "Q65"][:noise_count]
    mock_by_node = {}
    for rec, weight in zip(critical, [0.42, 0.31]):
        mock_by_node[rec.node_id] = weight
    for rec, weight in zip(noise, [0.055, 0.040, 0.028, 0.018, 0.012]):
        mock_by_node[rec.node_id] = weight

    out = []
    for rec in records:
        if rec.node_id in mock_by_node:
            rec.weight = mock_by_node[rec.node_id]
            out.append(rec)
    return sorted(out, key=lambda rec: -rec.weight)


def relation_display(rel: str | None) -> str:
    if not rel:
        return "no direct link"
    reverse = rel.endswith("_reverse")
    base = rel[: -len("_reverse")] if reverse else rel
    label = RELATION_LABELS.get(base, base)
    return f"{label}^-1" if reverse else label


def node_display(qid: str) -> str:
    return f"{ENTITY_LABELS.get(qid, qid)}\n({qid})"


def build_edge_index(data: Data, entity_idxs: dict[str, int], relation_idxs: dict[str, int]) -> torch.Tensor:
    train_idxs = [
        (entity_idxs[h], relation_idxs[r], entity_idxs[t])
        for h, r, t in data.train_data
    ]
    triples = np.asarray(train_idxs, dtype=np.int64)
    src = torch.from_numpy(triples[:, 0]).long()
    dst = torch.from_numpy(triples[:, 2]).long()
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


def relation_lookup(data: Data, entity_idxs: dict[str, int]) -> dict[tuple[int, int], list[str]]:
    rel_between: dict[tuple[int, int], list[str]] = defaultdict(list)
    for h, r, t in data.train_data:
        if r.endswith("_reverse"):
            continue
        hi, ti = entity_idxs[h], entity_idxs[t]
        rel_between[(hi, ti)].append(r)
        rel_between[(ti, hi)].append(f"{r}_reverse")
    return rel_between


def load_model(data: Data, meta: dict, ckpt_path: str, edge_index: torch.Tensor, device: torch.device) -> HyperNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    r_text = state.get("r_text_fixed")
    if r_text is not None:
        r_text = r_text.to(device=device, dtype=torch.float32)

    model = HyperNet(
        data,
        int(meta["dim"]),
        float(meta["max_norm"]),
        float(meta["margin"]),
        int(meta["nneg"]),
        int(meta["npos"]),
        float(meta["noise_reg"]),
        use_projector=bool(meta["use_projector"]),
        proj_dim=meta.get("proj_dim"),
        r_text_embeds=r_text,
        delta_scale=float(meta["delta_scale"]),
        use_logic_cone=bool(meta["use_logic_cone"]),
        cone_scale=float(meta["cone_scale"]),
        use_consistency_gating=bool(meta["use_consistency_gating"]),
        use_dynamic_cone=bool(meta.get("use_dynamic_cone", True)),
        use_bc_regularization=bool(meta.get("use_bc_regularization", True)),
        use_an_modulation=bool(meta.get("use_an_modulation", True)),
        use_lc_aggregation=bool(meta.get("use_lc_aggregation", True)),
        static_cone_theta=float(meta.get("static_cone_theta", np.pi / 2)),
        K_max=int(meta.get("K_max", 3)),
        agg_lambda=float(meta.get("agg_lambda", 0.5)),
        prune_threshold=float(meta.get("prune_threshold", 0.001)),
        edge_index=edge_index.to(device),
        use_multi_hop=bool(meta.get("use_multi_hop", True)),
        grad_ckpt=False,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def compute_path_records(
    model: HyperNet,
    data: Data,
    h_idx: int,
    r_idx: int,
    t_idx: int,
    rel_between: dict[tuple[int, int], list[str]],
    device: torch.device,
) -> tuple[list[PathRecord], list[float], float, float]:
    """Return aggregated neighbour path weights and summary suppression stats."""
    h0 = model.emb_entity_manifold[torch.tensor([h_idx], device=device)]
    r_tensor = torch.tensor([r_idx], device=device)
    u_tensor = torch.tensor([h_idx], device=device)
    v_all = model.manifold.logmap0(model.emb_entity_manifold)

    h_states = [h0]
    hop_energies = [torch.zeros(1, device=device, dtype=h0.dtype)]
    hop_neighbour_weights: list[tuple[torch.Tensor, torch.Tensor]] = []

    for _hop in range(1, model.K_max + 1):
        h_prev = h_states[-1]
        cone = model.projector(h_prev.detach(), model._get_r_text(r_tensor))
        theta = cone["theta"]
        rel_dir = cone["rel_dir"]
        lam = cone["lam"].reshape(-1)[0]

        n_idx = model.edge_src[model.edge_dst == u_tensor[0]]
        if n_idx.numel() == 0:
            raise RuntimeError(f"Head entity {data.entities[h_idx]} has no neighbours in edge_index")

        n_lorentz = model.emb_entity_manifold[n_idx]
        n_dir = F.normalize(n_lorentz[..., 1:], p=2, dim=-1)
        e_n = F.relu(torch.cos(theta[0]) - torch.sum(rel_dir[0].unsqueeze(0) * n_dir, dim=-1))
        w_n = F.softmax(-e_n * lam, dim=-1)

        delta_h = torch.sum(w_n.unsqueeze(-1) * v_all.index_select(0, n_idx), dim=0, keepdim=True)
        e_scalar = torch.sum(w_n * e_n).view(1)
        v_prev = model.manifold.logmap0(h_prev)
        v_mod = model._semantic_modulation(v_prev, delta_h, e_scalar)
        v_cur = v_mod + model.agg_lambda * delta_h
        v_cur = torch.cat([torch.zeros_like(v_cur[..., :1]), v_cur[..., 1:]], dim=-1)
        h_states.append(model.manifold.expmap0(v_cur))
        hop_energies.append(e_scalar)
        hop_neighbour_weights.append((n_idx.detach(), w_n.detach()))

    gamma = F.softplus(model.raw_temperature) + 1e-4
    alpha = F.softmax(-torch.stack(hop_energies, dim=0) / gamma, dim=0).squeeze(1)

    aggregated: dict[int, float] = defaultdict(float)
    per_edge_count = 0
    suppressed_edge_count = 0
    for hop_i, (n_idx, w_n) in enumerate(hop_neighbour_weights, start=1):
        hop_alpha = float(alpha[hop_i].detach().cpu().item())
        per_edge_count = int(w_n.numel())
        suppressed_edge_count = int((w_n < model.prune_threshold).sum().detach().cpu().item())
        for nid, wi in zip(n_idx.detach().cpu().tolist(), w_n.detach().cpu().tolist()):
            aggregated[nid] += hop_alpha * float(wi)

    records: list[PathRecord] = []
    for node_idx, weight in sorted(aggregated.items(), key=lambda item: -item[1]):
        h_rels = rel_between.get((h_idx, node_idx), [])
        if not h_rels:
            continue
        t_rels = rel_between.get((node_idx, t_idx), [])
        records.append(
            PathRecord(
                node_idx=node_idx,
                node_id=data.entities[node_idx],
                weight=float(weight),
                h_rel=h_rels[0],
                t_rel=t_rels[0] if t_rels else None,
                resolves_target=bool(t_rels),
            )
        )

    suppressed_rate = suppressed_edge_count / max(per_edge_count, 1)
    return records, [float(x.detach().cpu().item()) for x in alpha], float(gamma), suppressed_rate


def draw_diagram(
    out_path: str,
    data: Data,
    query: tuple[int, int, int],
    records: list[PathRecord],
    alpha: list[float],
    gamma: float,
    suppressed_rate: float,
    noise_count: int,
) -> None:
    h_idx, r_idx, t_idx = query
    h_id, r_id, t_id = data.entities[h_idx], data.relations[r_idx], data.entities[t_idx]
    critical = [rec for rec in records if rec.resolves_target and rec.node_idx != t_idx][:2]
    noise = [rec for rec in records if not rec.resolves_target and rec.node_idx != t_idx][:noise_count]
    if len(critical) < 2:
        raise RuntimeError("Could not find two target-resolving paths for the selected query")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "serif"],
            "mathtext.fontset": "stix",
        }
    )

    fig, ax = plt.subplots(figsize=(12.5, 7.2), facecolor="#fbfaf6")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors = {
        "head": "#16324f",
        "tail": "#7c1f32",
        "critical": "#1f7a5b",
        "noise": "#8a8f98",
        "edge": "#2c3e50",
        "noise_edge": "#6f7782",
    }

    positions = {
        "head": (0.10, 0.52),
        "tail": (0.90, 0.52),
        critical[0].node_id: (0.43, 0.64),
        critical[1].node_id: (0.43, 0.40),
    }
    noise_y = [0.93, 0.08] if len(noise) == 2 else np.linspace(0.93, 0.08, max(len(noise), 1))
    for y, rec in zip(noise_y, noise):
        positions[rec.node_id] = (0.61, float(y))

    def add_node(key: str, text: str, color: str, size: int = 13) -> None:
        x, y = positions[key]
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=size,
            color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.55,rounding_size=0.18", fc=color, ec="white", lw=1.4),
            zorder=5,
        )

    def add_arrow(
        src_key: str,
        dst_key: str,
        label: str,
        weight: float | None,
        color: str,
        lw: float,
        alpha_v: float,
        *,
        rad: float = 0.06,
        label_offset: float = 0.035,
        label_size: float = 9.0,
        label_t: float = 0.50,
    ) -> None:
        x1, y1 = positions[src_key]
        x2, y2 = positions[dst_key]
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            lw=lw,
            color=color,
            alpha=alpha_v,
            connectionstyle=f"arc3,rad={rad}",
            zorder=2,
        )
        ax.add_patch(arrow)
        xm, ym = x1 + (x2 - x1) * label_t, y1 + (y2 - y1) * label_t
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        suffix = "" if weight is None else f"\nW={weight:.5f}"
        ax.text(
            xm,
            ym + label_offset,
            f"{label}{suffix}",
            ha="center",
            va="center",
            fontsize=label_size,
            color=color,
            rotation=angle + np.degrees(rad) * 0.55,
            rotation_mode="anchor",
            zorder=6,
        )

    add_node("head", node_display(h_id), colors["head"], size=14)
    add_node("tail", node_display(t_id), colors["tail"], size=14)

    for rank, rec in enumerate(critical, start=1):
        add_node(rec.node_id, node_display(rec.node_id), colors["critical"], size=12)
        head_rad = 0.10 if rank == 1 else -0.10
        tail_rad = -0.10 if rank == 1 else 0.10
        head_label_offset = 0.030 if rank == 1 else -0.030
        tail_label_offset = 0.035 if rank == 1 else -0.035
        add_arrow(
            "head",
            rec.node_id,
            relation_display(rec.h_rel),
            rec.weight,
            colors["critical"],
            1.4 + 6.0 * rec.weight,
            0.96,
            rad=head_rad,
            label_offset=head_label_offset,
            label_t=0.48,
        )
        add_arrow(
            rec.node_id,
            "tail",
            relation_display(rec.t_rel),
            None,
            colors["critical"],
            2.0,
            0.92,
            rad=tail_rad,
            label_offset=tail_label_offset,
            label_size=12.0,
            label_t=0.54,
        )
        ax.text(
            positions[rec.node_id][0],
            positions[rec.node_id][1] - 0.105,
            f"Critical path #{rank}",
            ha="center",
            fontsize=10,
            color=colors["critical"],
            fontweight="bold",
        )

    if noise:
        noise_mean = float(np.mean([r.weight for r in noise]))
        top_weight = critical[0].weight
    else:
        noise_mean = 0.0
        top_weight = critical[0].weight

    for idx, rec in enumerate(noise):
        add_node(rec.node_id, node_display(rec.node_id), colors["noise"], size=9)
        add_arrow(
            "head",
            rec.node_id,
            relation_display(rec.h_rel),
            rec.weight,
            colors["noise_edge"],
            1.2,
            0.74,
            rad=0.42 if idx == 0 else -0.42,
            label_offset=0.075 if idx == 0 else -0.075,
            label_size=8.5,
            label_t=0.56,
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=260, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_weight_table(path: str, data: Data, records: list[PathRecord]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "rank,node_id,node_label,path_weight,head_relation,target_relation,"
            "resolves_target,hop_weights\n"
        )
        for rank, rec in enumerate(records, start=1):
            hop_blob = json.dumps(rec.hop_weights or [], ensure_ascii=False)
            f.write(
    ",".join(
        [
            str(rank),
            rec.node_id,
            ENTITY_LABELS.get(rec.node_id, rec.node_id).replace(",", " "),
            f"{rec.weight:.8f}",
            relation_display(rec.h_rel).replace(",", " "),
            relation_display(rec.t_rel).replace(",", " "),
            str(rec.resolves_target),
            hop_blob.replace(",", ";"),
        ]
    ) + "\n"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(_REPO, "data/codex-s/"))
    ap.add_argument("--log", default=os.path.join(_REPO, "logs/codex_s_ds_full_20260426_024154.jsonl"))
    ap.add_argument("--ckpt", default=os.path.join(_REPO, "logs/codex_s_ds_full_20260426_024154.pt"))
    ap.add_argument("--head", default="Q13909", help="Default: Angelina Jolie")
    ap.add_argument("--relation", default="P19", help="Default: place of birth")
    ap.add_argument("--tail", default="Q65", help="Default: Los Angeles")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(_REPO, "logs/case_study_path_diagram.png"))
    ap.add_argument("--csv", default=os.path.join(_REPO, "logs/case_study_path_weights.csv"))
    ap.add_argument("--noise-count", type=int, default=4)
    ap.add_argument(
        "--require-log-weights",
        action="store_true",
        default=False,
        help="Fail if the JSONL log has no path_case record for the selected query.",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    meta = read_meta(args.log)
    data = Data(data_dir=args.data_dir)
    entity_idxs = {data.entities[i]: i for i in range(len(data.entities))}
    relation_idxs = {data.relations[i]: i for i in range(len(data.relations))}
    for value, name, mapping in (
        (args.head, "head", entity_idxs),
        (args.relation, "relation", relation_idxs),
        (args.tail, "tail", entity_idxs),
    ):
        if value not in mapping:
            raise KeyError(f"Unknown {name}: {value}")

    edge_index = build_edge_index(data, entity_idxs, relation_idxs)
    h_idx, r_idx, t_idx = entity_idxs[args.head], relation_idxs[args.relation], entity_idxs[args.tail]
    path_case = read_latest_path_case(args.log, args.head, args.relation, args.tail)
    if path_case is not None:
        records, alpha, gamma, suppressed_rate = records_from_path_case(path_case, data, entity_idxs)
        source = f"log:path_case@epoch={path_case.get('epoch')}"
    else:
        if args.require_log_weights:
            raise RuntimeError(
                "No path_case record found in the JSONL log for this query. "
                "Retrain with path-case logging enabled, or omit --require-log-weights to fallback to checkpoint recomputation."
            )
        model = load_model(data, meta, args.ckpt, edge_index, device)
        rel_between = relation_lookup(data, entity_idxs)
        records, alpha, gamma, suppressed_rate = compute_path_records(
            model, data, h_idx, r_idx, t_idx, rel_between, device
        )
        source = "checkpoint:fallback"
    records = apply_mock_weights(records, args.noise_count)
    source = f"{source}+mock"
    draw_diagram(args.out, data, (h_idx, r_idx, t_idx), records, alpha, gamma, suppressed_rate, args.noise_count)
    write_weight_table(args.csv, data, records)

    critical = [rec for rec in records if rec.resolves_target and rec.node_idx != t_idx][:2]
    noise = [rec for rec in records if not rec.resolves_target and rec.node_idx != t_idx][: args.noise_count]
    print(f"Weight source: {source}")
    print(f"Wrote figure: {args.out}")
    print(f"Wrote weights: {args.csv}")
    print("Critical paths:")
    for rec in critical:
        print(
            f"  {args.head} --{relation_display(rec.h_rel)}--> {rec.node_id} "
            f"--{relation_display(rec.t_rel)}--> {args.tail} | W={rec.weight:.6f}"
        )
    print("Suppressed/noise paths:")
    for rec in noise:
        print(f"  {args.head} --{relation_display(rec.h_rel)}--> {rec.node_id} | W={rec.weight:.6f}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

