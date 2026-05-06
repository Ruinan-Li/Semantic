
from collections import defaultdict
from copy import deepcopy
import json
import os
import time
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
from load_data import Data
from LorentzModel import HyperNet
from optim.radam import RiemannianAdam
from optim.rsgd import RiemannianSGD
import csv
from torch.optim.lr_scheduler import StepLR

class Experiment:
    def __init__(self,
                 data=None,
                 margin=0.5,
                 noise_reg=0.15,
                 learning_rate=1e-3,
                 dim=40,
                 nneg=50,
                 npos=10,
                 valid_steps=10,
                 num_epochs=500,
                 batch_size=128,
                 max_norm=0.5,
                 max_grad_norm=1,
                 optimizer='radam',
                 cuda=True,
                 early_stop=10,
                 real_neg=True,
                 device='cuda:0',
                 step_size=30,
                 gamma=0.6,
                 use_projector=False,
                 proj_dim=None,
                 r_text_embeds=None,
                 use_tangent=True,
                 use_norm=True,
                 use_direction=True,
                 uniform_attn=False,
                 delta_scale=0.1,
                 use_logic_cone=True,
                 cone_scale=0.3,
                 use_consistency_gating=True,
                 use_dynamic_cone=True,
                 use_bc_regularization=True,
                 use_an_modulation=True,
                 use_lc_aggregation=True,
                 static_cone_theta=1.5707963267948966,
                 projector_lr=None,
                 log_path=None,
                 dataset_name=None,
                 checkpoint_path=None,
                 use_multi_hop=True,
                 K_max=3,
                 agg_lambda=0.5,
                 prune_threshold=0.005,
                 grad_ckpt=False,
                 log_path_case=True,
                 path_case_head="Q13909",
                 path_case_relation="P19",
                 path_case_tail="Q65",
                 path_case_topk=20,
                 path_case_every=1):
        self.data = data
        self.learning_rate = learning_rate
        self.dim = dim
        self.npos = npos
        self.nneg = nneg
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.max_grad_norm = max_grad_norm
        self.optimizer = optimizer
        self.valid_steps = valid_steps
        self.cuda = cuda
        self.early_stop = early_stop
        self.real_neg = real_neg
        self.device = device
        self.margin = margin
        self.noise_reg = noise_reg
        self.step_size = step_size
        self.gamma = gamma
        self.use_projector = use_projector
        self.proj_dim = proj_dim
        self.r_text_embeds = r_text_embeds
        self.use_tangent = use_tangent
        self.use_norm = use_norm
        self.use_direction = use_direction
        self.uniform_attn = uniform_attn
        self.delta_scale = delta_scale
        self.use_logic_cone = use_logic_cone
        self.cone_scale = cone_scale
        self.use_consistency_gating = use_consistency_gating
        self.use_dynamic_cone = use_dynamic_cone
        self.use_bc_regularization = use_bc_regularization
        self.use_an_modulation = use_an_modulation
        self.use_lc_aggregation = use_lc_aggregation
        self.static_cone_theta = static_cone_theta
        self.projector_lr = projector_lr
        self.log_path = log_path
        self.dataset_name = dataset_name
        self.checkpoint_path = checkpoint_path
        self.use_multi_hop = use_multi_hop
        self.K_max = K_max
        self.agg_lambda = agg_lambda
        self.prune_threshold = prune_threshold
        self.grad_ckpt = grad_ckpt
        self.log_path_case = bool(log_path_case)
        self.path_case_head = path_case_head
        self.path_case_relation = path_case_relation
        self.path_case_tail = path_case_tail
        self.path_case_topk = int(path_case_topk)
        self.path_case_every = max(1, int(path_case_every))
        self.entity_idxs = {data.entities[i]: i for i in range(len(data.entities))}
        self.relation_idxs = {data.relations[i]: i for i in range(len(data.relations))}
        self.relation_reverse_idxs = {vv: kk for kk, vv in self.relation_idxs.items()}

    def get_data_idxs(self, data):
        """ Return the training triplets """
        data_idxs = [
            (self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]],
             self.entity_idxs[data[i][2]]) for i in range(len(data))
        ]
        return data_idxs

    def get_er_vocab(self, data, idxs=[0, 1, 2]):
        """ Return the valid tail entities for (head, relation) pairs """
        er_vocab = defaultdict(set)
        for triple in data:
            er_vocab[(triple[idxs[0]], triple[idxs[1]])].add(triple[idxs[2]])
        return er_vocab


    def lorentz_distance_to_origin(x):  

        return np.arccosh(np.clip(x[:, 0], 1.0 + 1e-6, None))  # x[:, 0] must be >= 1



#---------------------------
    def evaluate(self, model, data, batch=100):
        d = self.data
        hits = [[] for _ in range(10)]
        ranks = []
        rank_by_rela = {}
        hit_by_rela = {}

        test_data_idxs = np.array(self.get_data_idxs(data))
        sr_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        tt = torch.tensor(np.arange(len(d.entities)), dtype=torch.int64).repeat(batch, 1)
        if self.cuda:
            tt = tt.cuda()


        relation_map = {

        }

        tail_emb_list = []
        group_list = []
        query_counter = 0
        max_queries = 20
        ranks = []
        hits = [[] for _ in range(10)]
        rank_by_rela = {}
        hit_by_rela = {}

        for i in range(0, len(test_data_idxs), batch):
            data_point = test_data_idxs[i:i + batch]
            e1_idx = torch.tensor(data_point[:, 0])
            r_idx = torch.tensor(data_point[:, 1])
            e2_idx = torch.tensor(data_point[:, 2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()

            predictions_s_h = model.forward(e1_idx, r_idx, tt[:min(batch, len(test_data_idxs) - i)])
            reverse_r_idx = torch.where(r_idx % 2 == 0, r_idx + 1, r_idx - 1)
            predictions_s_t = model.forward(tt[:min(batch, len(test_data_idxs) - i)], reverse_r_idx, e1_idx)
            predictions_s = torch.mean(torch.stack([predictions_s_t, predictions_s_h], dim=-1), dim=-1)

            for j in range(min(batch, len(test_data_idxs) - i)):
                filt = list(sr_vocab[(data_point[j][0], data_point[j][1])])
                target_value = predictions_s[j][e2_idx[j]].item()
                predictions_s[j][filt] = -np.Inf
                predictions_s[j][e1_idx[j]] = -np.Inf
                predictions_s[j][e2_idx[j]] = target_value

                rank = (predictions_s[j] >= target_value).sum().item() - 1
                ranks.append(rank + 1)
                rela_id = data_point[j][1] - 1 if data_point[j][1] % 2 == 1 else data_point[j][1]

              
                if rela_id in relation_map.values():  # Only include relations we care about
                    rank_by_rela.setdefault(rela_id, []).append(rank + 1)
                    hit_by_rela.setdefault(rela_id, [[] for _ in range(10)])[hits_level].append(hit)

                for hits_level in range(10):
                    hit = 1.0 if rank <= hits_level else 0.0
                    hits[hits_level].append(hit)

          
            if query_counter < max_queries:
                with torch.no_grad():
                    scores = predictions_s
                    topk_scores, topk_indices = torch.topk(scores[0], k=50)
                    tail_emb = model.emb_entity_manifold[topk_indices].detach().cpu().numpy()
                    tail_emb_list.append(tail_emb)
                    group_list.append(np.full(50, query_counter))
                    query_counter += 1

            if query_counter >= max_queries:
                break


        if tail_emb_list:
            os.makedirs("vis_data", exist_ok=True)
            np.save("vis_data/tail_embeddings.npy", np.vstack(tail_emb_list))
            np.save("vis_data/group_labels.npy", np.concatenate(group_list))
            print("Saved embeddings for visualization: vis_data/tail_embeddings.npy + group_labels.npy")


        for keys, values in hit_by_rela.items():
            if keys in relation_map.values():
                print(self.relation_reverse_idxs[keys], "->",
                    np.mean(values[9]), np.mean(values[2]), np.mean(values[0]),
                    np.mean(1. / (np.array(rank_by_rela[keys]) + 1e-6)))

        return np.mean(hits[9]), np.mean(hits[2]), np.mean(hits[0]), np.mean(1. / (np.array(ranks) + 1e-6))

    def _append_log(self, record: dict) -> None:
        if not self.log_path:
            return
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _relation_lookup(self, data_idxs: list[tuple[int, int, int]]) -> dict[tuple[int, int], list[int]]:
        rel_between: dict[tuple[int, int], list[int]] = defaultdict(list)
        _ = data_idxs
        for h_name, r_name, t_name in self.data.train_data:
            if r_name.endswith("_reverse"):
                continue
            h = self.entity_idxs[h_name]
            t = self.entity_idxs[t_name]
            r = self.relation_idxs[r_name]
            rel_between[(h, t)].append(r)
            reverse_name = f"{r_name}_reverse"
            if reverse_name in self.relation_idxs:
                rel_between[(t, h)].append(self.relation_idxs[reverse_name])
        return rel_between

    def _path_case_record(
        self,
        model,
        epoch: int,
        data_idxs: list[tuple[int, int, int]],
    ) -> dict | None:
        """Log concrete neighbour/path weights for one case-study query."""
        if not (self.log_path and self.log_path_case and self.use_projector and self.use_multi_hop):
            return None
        if (
            self.path_case_head not in self.entity_idxs
            or self.path_case_relation not in self.relation_idxs
            or self.path_case_tail not in self.entity_idxs
        ):
            return None
        if getattr(model, "edge_src", None) is None or getattr(model, "edge_dst", None) is None:
            return None

        was_training = model.training
        was_collecting = getattr(model, "_collect_pruned_nodes", True)
        model.eval()
        try:
            with torch.no_grad():
                device = model.emb_entity_manifold.device
                h_idx = self.entity_idxs[self.path_case_head]
                r_idx = self.relation_idxs[self.path_case_relation]
                t_idx = self.entity_idxs[self.path_case_tail]
                h_tensor = torch.tensor([h_idx], dtype=torch.long, device=device)
                r_tensor = torch.tensor([r_idx], dtype=torch.long, device=device)
                h_states = [model.emb_entity_manifold[h_tensor]]
                hop_energies = [torch.zeros(1, device=device, dtype=h_states[0].dtype)]
                v_all = model.manifold.logmap0(model.emb_entity_manifold)
                per_node: dict[int, dict] = {}
                hop_summaries = []

                model._collect_pruned_nodes = False
                for hop in range(1, model.K_max + 1):
                    h_prev = h_states[-1]
                    r_text = model._get_r_text(r_tensor)
                    cone = model.projector(h_prev.detach(), r_text)
                    theta = cone["theta"]
                    rel_dir = cone["rel_dir"]
                    lam = cone["lam"].reshape(-1)[0]

                    n_idx = model.edge_src[model.edge_dst == h_tensor[0]]
                    if n_idx.numel() == 0:
                        return None
                    n_lorentz = model.emb_entity_manifold[n_idx]
                    n_dir = torch.nn.functional.normalize(n_lorentz[..., 1:], p=2, dim=-1)
                    energy = torch.nn.functional.relu(
                        torch.cos(theta[0]) - torch.sum(rel_dir[0].unsqueeze(0) * n_dir, dim=-1)
                    )
                    if model.use_bc_regularization:
                        boundary_threshold = energy.detach().mean() + energy.detach().std(unbiased=False)
                        hard_mask = energy > boundary_threshold
                    else:
                        hard_mask = torch.zeros_like(energy, dtype=torch.bool)
                    valid = ~hard_mask
                    if model.use_an_modulation and torch.any(valid):
                        logits = (-energy * lam).masked_fill(hard_mask, -1e9)
                        weights = torch.nn.functional.softmax(logits, dim=-1)
                    else:
                        weights = torch.full_like(energy, 1.0 / max(energy.numel(), 1))

                    delta_h = torch.sum(weights.unsqueeze(-1) * v_all.index_select(0, n_idx), dim=0, keepdim=True)
                    e_scalar = torch.sum(weights * energy).view(1)
                    v_prev = model.manifold.logmap0(h_prev)
                    v_mod = model._semantic_modulation(v_prev, delta_h, e_scalar)
                    v_cur = v_mod + model.agg_lambda * delta_h
                    v_cur = torch.cat([torch.zeros_like(v_cur[..., :1]), v_cur[..., 1:]], dim=-1)
                    h_states.append(model.manifold.expmap0(v_cur))
                    hop_energies.append(e_scalar)

                    hop_summaries.append(
                        {
                            "hop": hop,
                            "candidate_count": int(weights.numel()),
                            "masked_count": int(hard_mask.sum().detach().cpu().item()),
                            "pruned_count": int((weights < model.prune_threshold).sum().detach().cpu().item()),
                            "energy_scalar": float(e_scalar.detach().cpu().item()),
                            "lambda": float(lam.detach().cpu().item()),
                            "theta": float(theta[0].detach().cpu().item()),
                        }
                    )
                    for nid, wi, ei, mi in zip(
                        n_idx.detach().cpu().tolist(),
                        weights.detach().cpu().tolist(),
                        energy.detach().cpu().tolist(),
                        hard_mask.detach().cpu().tolist(),
                    ):
                        rec = per_node.setdefault(
                            int(nid),
                            {
                                "node_idx": int(nid),
                                "node_id": self.data.entities[int(nid)],
                                "aggregated_weight": 0.0,
                                "hop_weights": [],
                            },
                        )
                        rec["hop_weights"].append(
                            {
                                "hop": hop,
                                "neighbor_weight": float(wi),
                                "energy": float(ei),
                                "masked": bool(mi),
                                "below_prune_threshold": bool(wi < model.prune_threshold),
                            }
                        )

                gamma = torch.nn.functional.softplus(model.raw_temperature) + 1e-4
                alpha = torch.nn.functional.softmax(torch.neg(torch.stack(hop_energies, dim=0)) / gamma, dim=0).squeeze(1)
                alpha_values = [float(x.detach().cpu().item()) for x in alpha]
                for rec in per_node.values():
                    rec["aggregated_weight"] = sum(
                        alpha_values[item["hop"]] * item["neighbor_weight"]
                        for item in rec["hop_weights"]
                    )

                rel_between = self._relation_lookup(data_idxs)
                for rec in per_node.values():
                    node_idx = int(rec["node_idx"])
                    head_rels = rel_between.get((h_idx, node_idx), [])
                    target_rels = rel_between.get((node_idx, t_idx), [])
                    rec["head_relations"] = [self.relation_reverse_idxs[x] for x in head_rels]
                    rec["target_relations"] = [self.relation_reverse_idxs[x] for x in target_rels]
                    rec["resolves_target"] = bool(target_rels)

                paths = sorted(per_node.values(), key=lambda item: -float(item["aggregated_weight"]))
                return {
                    "type": "path_case",
                    "epoch": int(epoch),
                    "query": {
                        "head": self.path_case_head,
                        "relation": self.path_case_relation,
                        "tail": self.path_case_tail,
                    },
                    "alpha": alpha_values,
                    "gamma": float(gamma.detach().cpu().item()),
                    "prune_threshold": float(model.prune_threshold),
                    "hop_summaries": hop_summaries,
                    "paths": paths[: self.path_case_topk],
                }
        finally:
            model._collect_pruned_nodes = was_collecting
            model.train(was_training)

    def _sample_pairwise_mad(self,
                             representations: torch.Tensor,
                             manifold=None,
                             max_nodes: int = 2048,
                             max_pairs: int = 8192) -> float:
        """Mean Average Distance based on sampled raw Lorentz geodesic distance."""
        if representations is None or representations.size(0) < 2:
            return float("nan")
        x = representations.detach()
        if x.dim() != 2:
            x = x.view(x.size(0), -1)
        n = x.size(0)
        if n > max_nodes:
            node_idx = torch.randperm(n, device=x.device)[:max_nodes]
            x = x.index_select(0, node_idx)
            n = x.size(0)
        pair_count = min(max_pairs, n * (n - 1))
        left = torch.randint(0, n, (pair_count,), device=x.device)
        right = torch.randint(0, n - 1, (pair_count,), device=x.device)
        right = right + (right >= left).long()
        if manifold is None:
            raise ValueError("Lorentz MAD requires a manifold instance")
        distances = manifold.dist(x[left], x[right])
        return float(distances.float().mean().detach().cpu())

    @staticmethod
    def _radial_normalize_lorentz(x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """Project Lorentz points to a shared radial shell while preserving spatial direction."""
        x = x.detach()
        if x.dim() != 2:
            x = x.view(x.size(0), -1)
        spatial = x[:, 1:]
        direction = torch.nn.functional.normalize(spatial.float(), p=2, dim=-1, eps=1e-12)
        radius = torch.as_tensor(radius, device=x.device, dtype=direction.dtype).clamp_min(1e-6)
        spatial_normed = direction * radius
        time = torch.sqrt(1.0 + torch.sum(spatial_normed * spatial_normed, dim=-1, keepdim=True))
        return torch.cat([time, spatial_normed], dim=-1).to(dtype=x.dtype)

    @staticmethod
    def _pairwise_mad_with_pairs(representations: torch.Tensor,
                                 manifold,
                                 left: torch.Tensor,
                                 right: torch.Tensor,
                                 radius: torch.Tensor | None = None) -> float:
        """Lorentz MAD over fixed pairs; optional shared radius exposes angular collapse."""
        x = representations.detach()
        if x.dim() != 2:
            x = x.view(x.size(0), -1)
        if radius is not None:
            x = Experiment._radial_normalize_lorentz(x, radius)
        distances = manifold.dist(x[left], x[right])
        return float(distances.float().mean().detach().cpu())

    def _compute_mad_metrics(self, model, train_data_idxs_np: np.ndarray) -> dict:
        """Track representation collapse; lower MAD means stronger oversmoothing.

        This version is hop-aware: h0..hK are measured before final hop fusion, so
        it is more sensitive to progressive smoothing than one global final MAD.
        """
        was_training = model.training
        was_collecting = getattr(model, "_collect_pruned_nodes", True)
        model.eval()
        try:
            with torch.no_grad():
                entity_raw_mad = self._sample_pairwise_mad(
                    model.emb_entity_manifold,
                    manifold=model.manifold,
                )
                entity_radius = torch.linalg.norm(
                    model.emb_entity_manifold.detach()[:, 1:].float(),
                    dim=-1,
                ).mean()
                metrics = {
                    "entity_mad_raw": entity_raw_mad,
                    "entity_mad": self._sample_pairwise_mad(
                        self._radial_normalize_lorentz(model.emb_entity_manifold, entity_radius),
                        manifold=model.manifold,
                    ),
                }
                if self.use_projector and self.use_multi_hop and train_data_idxs_np.size:
                    sample_size = min(512, len(train_data_idxs_np))
                    sample_idx = np.random.choice(len(train_data_idxs_np), size=sample_size, replace=False)
                    sample = train_data_idxs_np[sample_idx]
                    device = model.emb_entity_manifold.device
                    u_idx = torch.as_tensor(sample[:, 0], dtype=torch.long, device=device)
                    r_idx = torch.as_tensor(sample[:, 1], dtype=torch.long, device=device)
                    h_states = [model.emb_entity_manifold[u_idx]]
                    energies = [torch.zeros(sample_size, device=device, dtype=h_states[0].dtype)]
                    v_all = (
                        model.manifold.logmap0(model.emb_entity_manifold)
                        if getattr(model, "edge_src", None) is not None
                        else None
                    )
                    model._collect_pruned_nodes = False
                    for _ in range(1, model.K_max + 1):
                        h_cur, e_cur, _ = model._one_hop(h_states[-1], r_idx, u_idx, v_all)
                        h_states.append(h_cur)
                        energies.append(e_cur)

                    hop_mads = []
                    hop_raw_mads = []
                    pair_count = min(8192, sample_size * (sample_size - 1))
                    left = torch.randint(0, sample_size, (pair_count,), device=device)
                    right = torch.randint(0, sample_size - 1, (pair_count,), device=device)
                    right = right + (right >= left).long()
                    shell_radius = torch.linalg.norm(
                        h_states[0].detach()[:, 1:].float(),
                        dim=-1,
                    ).mean()
                    for hop_id, h_state in enumerate(h_states):
                        hop_raw_mad = self._pairwise_mad_with_pairs(
                            h_state,
                            model.manifold,
                            left,
                            right,
                        )
                        hop_mad = self._pairwise_mad_with_pairs(
                            h_state,
                            model.manifold,
                            left,
                            right,
                            radius=shell_radius,
                        )
                        metrics[f"hop{hop_id}_mad_raw"] = hop_raw_mad
                        metrics[f"hop{hop_id}_mad"] = hop_mad
                        hop_raw_mads.append(hop_raw_mad)
                        hop_mads.append(hop_mad)

                    e_stack = torch.stack(energies, dim=0)
                    gamma = torch.nn.functional.softplus(model.raw_temperature) + 1e-4
                    alpha = torch.nn.functional.softmax(-e_stack / gamma, dim=0)
                    v_stack = torch.stack([model.manifold.logmap0(h) for h in h_states], dim=0)
                    h_reasoned = model.manifold.expmap0((alpha.unsqueeze(-1) * v_stack).sum(dim=0))
                    metrics["reasoned_mad_raw"] = self._pairwise_mad_with_pairs(
                        h_reasoned,
                        model.manifold,
                        left,
                        right,
                    )
                    metrics["reasoned_mad"] = self._pairwise_mad_with_pairs(
                        h_reasoned,
                        model.manifold,
                        left,
                        right,
                        radius=shell_radius,
                    )
                    metrics["hop_mad_retention_raw"] = (
                        hop_raw_mads[-1] / hop_raw_mads[0]
                        if hop_raw_mads and hop_raw_mads[0] and np.isfinite(hop_raw_mads[0])
                        else float("nan")
                    )
                    metrics["hop_mad_retention"] = (
                        hop_mads[-1] / hop_mads[0]
                        if hop_mads and hop_mads[0] and np.isfinite(hop_mads[0])
                        else float("nan")
                    )
                    metrics["hop_mad_drop"] = (
                        hop_mads[0] - hop_mads[-1]
                        if hop_mads and np.isfinite(hop_mads[0]) and np.isfinite(hop_mads[-1])
                        else float("nan")
                    )
                    metrics["mad_retention"] = (
                        metrics["reasoned_mad"] / metrics["entity_mad"]
                        if metrics["entity_mad"] and np.isfinite(metrics["entity_mad"])
                        else float("nan")
                    )
                else:
                    metrics["reasoned_mad"] = float("nan")
                    metrics["mad_retention"] = float("nan")
                    metrics["hop_mad_retention"] = float("nan")
                    metrics["hop_mad_drop"] = float("nan")
                return metrics
        finally:
            if hasattr(model, "_collect_pruned_nodes"):
                model._collect_pruned_nodes = was_collecting
            model.train(was_training)

    @property
    def train_and_eval(self):
        d = self.data
        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        # ── Build symmetric edge_index for 1-hop neighbour aggregation ──
        edge_index = None
        if self.use_multi_hop:
            _trip = np.array(train_data_idxs, dtype=np.int64)          # [N, 3]
            _src = torch.from_numpy(_trip[:, 0]).long()
            _dst = torch.from_numpy(_trip[:, 2]).long()
            edge_index = torch.stack(
                [torch.cat([_src, _dst]),
                 torch.cat([_dst, _src])], dim=0,
            )  # [2, 2E]  symmetric
            print(
                "[SIGHT Multi-hop] enabled | K_max=%d | agg_lambda=%.3f | "
                "edges=%d (symmetric) | grad_ckpt=%s" %
                (self.K_max, self.agg_lambda, edge_index.shape[1], self.grad_ckpt)
            )
        else:
            print("[SIGHT Multi-hop] DISABLED (falling back to single-step refinement)")

        model = HyperNet(d, self.dim, self.max_norm, self.margin, self.nneg, self.npos, self.noise_reg,
                         use_projector=self.use_projector, proj_dim=self.proj_dim,
                         r_text_embeds=self.r_text_embeds,
                         use_tangent=self.use_tangent, use_norm=self.use_norm,
                         use_direction=self.use_direction,
                         uniform_attn=self.uniform_attn,
                         delta_scale=self.delta_scale,
                         use_logic_cone=self.use_logic_cone,
                         cone_scale=self.cone_scale,
                         use_consistency_gating=self.use_consistency_gating,
                         use_dynamic_cone=self.use_dynamic_cone,
                         use_bc_regularization=self.use_bc_regularization,
                         use_an_modulation=self.use_an_modulation,
                         use_lc_aggregation=self.use_lc_aggregation,
                         static_cone_theta=self.static_cone_theta,
                         K_max=self.K_max,
                         agg_lambda=self.agg_lambda,
                         prune_threshold=self.prune_threshold,
                         edge_index=edge_index,
                         use_multi_hop=self.use_multi_hop,
                         grad_ckpt=self.grad_ckpt)
        print("Training the %s model..." % model)

        if self.log_path:
            log_dir = os.path.dirname(self.log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            meta = {
                "type": "meta",
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "dataset": self.dataset_name,
                "num_entities": len(d.entities),
                "num_relations": len(d.relations),
                "train_triples": len(train_data_idxs),
                "dim": self.dim,
                "margin": self.margin,
                "noise_reg": self.noise_reg,
                "learning_rate": self.learning_rate,
                "projector_lr": self.projector_lr,
                "optimizer": self.optimizer,
                "batch_size": self.batch_size,
                "nneg": self.nneg,
                "npos": self.npos,
                "max_norm": self.max_norm,
                "max_grad_norm": self.max_grad_norm,
                "num_epochs": self.num_epochs,
                "valid_steps": self.valid_steps,
                "early_stop": self.early_stop,
                "real_neg": self.real_neg,
                "step_size": self.step_size,
                "gamma": self.gamma,
                "cuda": self.cuda,
                "device": self.device,
                "use_projector": self.use_projector,
                "proj_dim": self.proj_dim,
                "use_tangent": self.use_tangent,
                "use_norm": self.use_norm,
                "use_direction": self.use_direction,
                "uniform_attn": self.uniform_attn,
                "delta_scale": self.delta_scale,
                "use_logic_cone": self.use_logic_cone,
                "cone_scale": self.cone_scale,
                "use_consistency_gating": self.use_consistency_gating,
                "use_dynamic_cone": self.use_dynamic_cone,
                "use_bc_regularization": self.use_bc_regularization,
                "use_an_modulation": self.use_an_modulation,
                "use_lc_aggregation": self.use_lc_aggregation,
                "static_cone_theta": self.static_cone_theta,
                "use_multi_hop": self.use_multi_hop,
                "K_max": self.K_max,
                "agg_lambda": self.agg_lambda,
                "prune_threshold": self.prune_threshold,
                "grad_ckpt": self.grad_ckpt,
                "r_text_fixed": self.r_text_embeds is not None,
                "checkpoint_path": self.checkpoint_path,
            }
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            print("Writing JSONL log to: %s" % self.log_path)

        embedding_params = []
        other_params = []
        projector_params = []
        for name, param in model.named_parameters():
            if 'projector' in name or 'proj_back' in name or 'r_text_embed' in name:
                projector_params.append(param)
            elif 'rel_center' in name or 'dir' in name or 'linear' in name:
                embedding_params.append(param)
            else:
                other_params.append(param)

        proj_lr = self.projector_lr or self.learning_rate
        param_groups = [
            {'params': embedding_params, 'weight_decay': 1e-4},
            {'params': other_params, 'weight_decay': 0.0},
        ]
        if projector_params:
            param_groups.append({'params': projector_params, 'weight_decay': 0.0, 'lr': proj_lr})

        if self.optimizer == 'radam':
            opt = RiemannianAdam(param_groups, lr=self.learning_rate, stabilize=1)
        elif self.optimizer == 'rsgd':
            opt = RiemannianSGD(param_groups, lr=self.learning_rate, stabilize=1)
        elif self.optimizer == 'adam':
            opt = Adam(param_groups, lr=self.learning_rate)
        else:
            raise ValueError("Wrong optimizer")

        scheduler = StepLR(optimizer=opt, step_size=self.step_size, gamma=self.gamma, verbose=True)

        if self.cuda:
            model.cuda()

        train_data_idxs_np = np.array(train_data_idxs)
        train_data_idxs = torch.tensor(np.array(train_data_idxs)).cuda() if self.cuda else torch.tensor(
            np.array(train_data_idxs))

        train_order = list(range(len(train_data_idxs)))
        targets = np.zeros((self.batch_size, self.nneg * self.npos + self.npos))
        targets[:, 0:self.npos] = 1
        targets = torch.FloatTensor(targets).cuda() if self.cuda else torch.FloatTensor(targets)
        max_mrr = 0.0
        max_it = 0
        mrr = 0
        bad_cnt = 0
        print("Starting training...")
        sr_vocab = self.get_er_vocab(self.get_data_idxs(d.data))
        step = self.batch_size * self.npos
        n_train = len(train_data_idxs)
        tqdm_kwargs = {
            "dynamic_ncols": True,
            "mininterval": 2.0,
            "leave": True,
        }
        bar = tqdm(
            total=self.num_epochs,
            desc='Epoch',
            unit='epoch',
            **tqdm_kwargs,
        )
        best_model = None
        for it in range(1, self.num_epochs + 1):
            model.train()
            model.reset_pruned_nodes()
            losses = []
            np.random.shuffle(train_order)
            batch_starts = [j for j in range(0, n_train, step) if j + step <= n_train]
            epoch_batches = len(batch_starts)
            for batch_idx, j in enumerate(batch_starts, start=1):
                data_batch = train_data_idxs[train_order[j:j + step]]
                data_batch_np = train_data_idxs_np[train_order[j:j + step]]

                data_batch = data_batch.view(self.batch_size, -1, 3)
                data_batch_np = data_batch_np.reshape(self.batch_size, -1, 3)

                negsamples = np.random.randint(low=0,
                                               high=len(self.entity_idxs),
                                               size=(data_batch.size(0), self.npos, self.nneg),
                                               dtype=np.int32)
                if self.real_neg:
                    candidate = np.random.randint(low=0,
                                                  high=len(self.entity_idxs),
                                                  size=(data_batch.size(0)),
                                                  dtype=np.int32)
                    e1_idx_np = data_batch_np[:, :, 0]
                    r_idx_np = data_batch_np[:, :, 1]
                    for index in range(negsamples.shape[0]):
                        for index2 in range(negsamples.shape[1]):
                            filt = sr_vocab[(e1_idx_np[index, index2], r_idx_np[index, index2])]
                            for index_ in range(negsamples.shape[2]):
                                p_candidate = 0
                                while negsamples[index, index2][index_] in filt:
                                    negsamples[index, index2][index_] = candidate[p_candidate]
                                    p_candidate += 1
                                    if p_candidate == len(candidate):
                                        candidate = np.random.randint(low=0, high=len(self.entity_idxs),
                                                                      size=(self.batch_size), )
                                        p_candidate = 0
                negsamples = torch.LongTensor(negsamples).cuda() if self.cuda else torch.LongTensor(negsamples)

                opt.zero_grad()
                e1_idx = data_batch[:, :, 0]
                r_idx = data_batch[:, :, 1]
                e2_idx = torch.cat([data_batch[:, :, 2:3], negsamples], dim=-1)

                intervals = model.forward(e1_idx, r_idx, e2_idx)
                loss = model.loss(intervals, targets)

                e1_idx = data_batch[:, :, 0]
                r_idx = data_batch[:, :, 1]
                r_idx = torch.where(r_idx % 2 == 0, r_idx + 1, r_idx - 1)
                e2_idx = torch.cat([data_batch[:, :, 2:3], negsamples], dim=-1)
                intervals = model.forward(e2_idx, r_idx, e1_idx)
                loss += model.loss(intervals, targets)

                loss.backward()
                skip_update = not torch.isfinite(loss).item()
                if self.max_grad_norm > 0 and not skip_update:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=self.max_grad_norm,
                        error_if_nonfinite=False,
                    )
                    skip_update = skip_update or not torch.isfinite(grad_norm).item()
                if skip_update:
                    opt.zero_grad()
                else:
                    opt.step()
                losses.append(loss.detach().item())
                if epoch_batches:
                    # Show progress inside the current epoch on the same tqdm line.
                    bar.n = (it - 1) + batch_idx / epoch_batches
                if batch_idx == 1 or batch_idx == epoch_batches or batch_idx % 10 == 0:
                    bar.set_postfix(
                        epoch=f'{it}/{self.num_epochs}',
                        batch=f'{batch_idx}/{len(batch_starts)}',
                        loss=f'{loss.item():.4f}',
                        mean_loss=f'{np.mean(losses):.4f}',
                        refresh=False,
                    )
                    bar.refresh()

            bar.n = it
            bar.set_postfix(
                best_mrr=f'{max_mrr:.4f}@{max_it}',
                cur_mrr=f'{mrr:.4f}',
                loss=f'{np.mean(losses):.6f}' if losses else 'n/a',
            )
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            pruning_stats = model.get_pruning_stats()
            mad_metrics = self._compute_mad_metrics(model, train_data_idxs_np)
            print(
                "[PrunedNodes] epoch=%d pruned=%d/%d rate=%.2f%% total=%d/%d threshold=%.6g | "
                "MAD entity=%.4f reasoned=%.4f retention=%.4f hop_ret=%.4f hop_drop=%.4f" %
                (
                    it,
                    pruning_stats["pruned_nodes"],
                    pruning_stats["candidate_nodes"],
                    pruning_stats["prune_rate"] * 100.0,
                    pruning_stats["pruned_nodes_total"],
                    pruning_stats["candidate_nodes_total"],
                    self.prune_threshold,
                    mad_metrics["entity_mad"],
                    mad_metrics["reasoned_mad"],
                    mad_metrics["mad_retention"],
                    mad_metrics["hop_mad_retention"],
                    mad_metrics["hop_mad_drop"],
                )
            )
            lrs = [float(x) for x in scheduler.get_last_lr()]
            train_record = {
                "type": "train",
                "epoch": it,
                "loss": mean_loss,
                "lr": lrs[0] if len(lrs) == 1 else lrs,
                "pruned_nodes": int(pruning_stats["pruned_nodes"]),
                "candidate_nodes": int(pruning_stats["candidate_nodes"]),
                "prune_rate": float(pruning_stats["prune_rate"]),
                "pruned_nodes_total": int(pruning_stats["pruned_nodes_total"]),
                "candidate_nodes_total": int(pruning_stats["candidate_nodes_total"]),
                "prune_rate_total": float(pruning_stats["prune_rate_total"]),
                "masked_nodes": int(pruning_stats["masked_nodes"]),
                "masked_nodes_total": int(pruning_stats["masked_nodes_total"]),
                "masked_rate": float(pruning_stats["masked_rate"]),
                "masked_rate_total": float(pruning_stats["masked_rate_total"]),
                "attention_entropy": float(pruning_stats["attention_entropy"]),
                "attention_entropy_total": float(pruning_stats["attention_entropy_total"]),
                "entity_mad": float(mad_metrics["entity_mad"]),
                "reasoned_mad": float(mad_metrics["reasoned_mad"]),
                "mad_retention": float(mad_metrics["mad_retention"]),
                "hop_mad_retention": float(mad_metrics["hop_mad_retention"]),
                "hop_mad_drop": float(mad_metrics["hop_mad_drop"]),
            }
            train_record.update({
                key: float(value)
                for key, value in mad_metrics.items()
                if key.startswith("hop") and key.endswith("_mad")
            })
            self._append_log(train_record)
            if it % self.path_case_every == 0:
                path_case = self._path_case_record(model, it, train_data_idxs)
                if path_case is not None:
                    self._append_log(path_case)
            scheduler.step()
            model.eval()
            with torch.no_grad():
                if not it % self.valid_steps:
                    hit10, hit3, hit1, mrr = self.evaluate(model, d.valid_data)
                    if mrr > max_mrr:
                        max_mrr = mrr
                        max_it = it
                        bad_cnt = 0
                        best_model = deepcopy(model.state_dict())
                    else:
                        bad_cnt += 1
                        if bad_cnt == self.early_stop:
                            break
                    bar.set_postfix(
                        best_mrr=f'{max_mrr:.4f}@{max_it}',
                        cur_mrr=f'{mrr:.4f}',
                        loss=f'{loss.item():.6f}',
                    )
                    self._append_log({
                        "type": "valid",
                        "epoch": it,
                        "hit10": float(hit10),
                        "hit3": float(hit3),
                        "hit1": float(hit1),
                        "mrr": float(mrr),
                        "best_mrr": float(max_mrr),
                        "best_epoch": int(max_it),
                        "bad_cnt": int(bad_cnt),
                    })
        bar.close()
        with torch.no_grad():
            model.load_state_dict(best_model)
            model.eval()
            if self.checkpoint_path:
                os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "best_epoch": int(max_it),
                        "best_mrr": float(max_mrr),
                        "meta": {
                            "use_projector": self.use_projector,
                            "delta_scale": self.delta_scale,
                            "use_logic_cone": self.use_logic_cone,
                            "cone_scale": self.cone_scale,
                            "use_consistency_gating": self.use_consistency_gating,
                            "dim": self.dim,
                        },
                    },
                    self.checkpoint_path,
                )
                print("Checkpoint saved: %s" % self.checkpoint_path)
            hit10, hit3, hit1, mrr = self.evaluate(model, d.test_data)
        print(
            'Test Result\nBest it:%d\nHit@10:%f\nHit@3:%f\nHit@1:%f\nMRR:%f\nPrunedNodes:%d' %
            (max_it, hit10, hit3, hit1, mrr, model.pruned_nodes_total))
        self._append_log({
            "type": "test",
            "best_epoch": int(max_it),
            "hit10": float(hit10),
            "hit3": float(hit3),
            "hit1": float(hit1),
            "mrr": float(mrr),
            "pruned_nodes_total": int(model.pruned_nodes_total),
        })
        if self.log_path:
            print("JSONL log saved: %s" % self.log_path)

        return mrr, hit1, hit3, hit10
