import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt import ManifoldParameter

from manifolds.lorentz import Lorentz
from TripleViewHyperbolicProjector import SemanticConeProjector
import numpy as np


class FLG(nn.Module):
    def __init__(self, manifold, num_emb, dim):
        super().__init__()
        self.manifold = manifold
        self.dim = dim
        self.num_emb = num_emb
        self.linear = nn.Embedding(num_emb, (dim - 1) * (dim - 1))
        self.register_buffer('I3', torch.eye(self.dim - 1).view(1, 1, self.dim - 1, self.dim - 1).repeat([self.num_emb, self.dim - 1, 1, 1]))
        self.register_buffer('Iw', torch.eye(self.dim - 1).view(1, self.dim - 1, self.dim - 1).repeat([self.num_emb, 1, 1]))
        self.flip_sign = nn.Parameter(torch.tensor(0.0))

    def forward(self, para):
        x = para[0]
        r_idx = para[1]

        x_0 = x.narrow(-1, 0, 1)
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)

        ww = self.linear.weight
        ww = torch.nn.functional.gelu(ww)
        ww = ww.view(-1, self.dim - 1, self.dim - 1)

        bvv = torch.einsum('bwv, bwk -> bwvk', ww, ww)
        nbvv = torch.einsum('bwlv, bwvi -> bwli', ww.unsqueeze(-2), ww.unsqueeze(-1))
        qbvvt = (self.I3 - 2 * bvv / nbvv).permute([1, 0, 2, 3])
        ww = self.Iw
        for i in range(qbvvt.shape[0]):
            ww = ww @ qbvvt[i]
        ww[:, :, -1] *= self.flip_sign.to(ww.device)
        ww = ww[r_idx]
        x_narrow = torch.einsum('bnd, bdc -> bnc', x_narrow, ww)

        xo = torch.cat([x_0, x_narrow], dim=-1)
        return (xo, r_idx)


class DO(nn.Module):
    def __init__(self, manifold, num_rel, dim):
        super().__init__()
        self.manifold = manifold
        self.dim = dim
        self.rel_center = nn.Embedding(num_rel, dim)
        self.dir = nn.Embedding(num_rel, dim)

        nn.init.normal_(self.rel_center.weight, std=0.01)
        nn.init.normal_(self.dir.weight, std=0.01)

    def forward(self, para):
        x, r_idx = para

        c = self.rel_center(r_idx).unsqueeze(1)
        d = self.dir(r_idx).unsqueeze(1)

        inner = torch.sum(c * d, dim=-1, keepdim=True)
        tangent = d + inner * c
        tangent = tangent / (torch.norm(tangent, dim=-1, keepdim=True) + 1e-6)

        offset = self.manifold.expmap(c, 0.1 * tangent)
        x_out = self.manifold.mobius_add(x, offset)

        return x_out, r_idx


class HyperNet(nn.Module):
    """
    FlorE backbone (FLG + DO + Lorentz scoring) augmented with the
    Semantic-guided Dynamic Sequential Lorentz Entailment Cone (Section 4 of SIHR).

    Multi-hop reasoning loop (Eq.(2)+(6)-(15)):

        for i = 1..K:
            θ^i   = Sigmoid( MLP(h_r) · ||h_spatial|| ) · π           (6)
            a^i   = h_spatial / ||h_spatial||                         (7)
            E^i_u = ReLU( cos(θ^i) - cos(h_r, h_u) )  for each u∈N(v) (8')
            w^i_u = softmax(-E^i_u · λ)                               (10)
            Δh^i  = Σ_u w^i_u · log_0(h_u)                            (9)
            h^{i} = exp_0( log_0(h^{i-1}) ⊛ Δh^i )                    (12)(13)
        h^K    = Σ_i α^i · h^i  with α^i = softmax(-E^i_v / Γ)         (14)(15)
    """

    def __init__(self, d, dims, max_norm, margin, neg_sample, npos, noise_reg,
                 use_projector=False, proj_dim=None, r_text_embeds=None,
                 delta_scale=0.1,
                 use_logic_cone=True, cone_scale=0.3,
                 use_consistency_gating=True,
                 use_dynamic_cone: bool = True,
                 use_bc_regularization: bool = True,
                 use_an_modulation: bool = True,
                 use_lc_aggregation: bool = True,
                 static_cone_theta: float = math.pi / 2,
                 K_max: int = 3,
                 agg_lambda: float = 0.5,
                 prune_threshold: float = 0.005,
                 edge_index: Optional[torch.Tensor] = None,
                 use_multi_hop: bool = True,
                 grad_ckpt: bool = False,
                 # Kept for CLI backward-compatibility – no longer used.
                 use_tangent: bool = True,
                 use_norm: bool = True,
                 use_direction: bool = True,
                 uniform_attn: bool = False):
        super(HyperNet, self).__init__()
        self.manifold = Lorentz(max_norm=max_norm)
        self.dim = dims
        self.noise_reg = noise_reg
        self.num_r_emb = len(d.relations)
        self.num_e_emb = len(d.entities)
        self.emb_entity_manifold = ManifoldParameter(
            self.manifold.random_normal((self.num_e_emb, dims), std=1. / math.sqrt(dims)),
            manifold=self.manifold,
        )
        self.margin = margin
        self.bias_head = torch.nn.Parameter(torch.zeros(self.num_e_emb))
        self.bias_tail = torch.nn.Parameter(torch.zeros(self.num_e_emb))
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.neg_sample = neg_sample
        self.npos = npos

        # ── FlorE main pipeline ─────────────────────────────────────
        self.head_linear = nn.Sequential(
            FLG(self.manifold, self.num_r_emb, dims),
            DO(self.manifold, self.num_r_emb, dims),
        )
        self.tail_linear = nn.Sequential(
            FLG(self.manifold, self.num_r_emb, dims),
            DO(self.manifold, self.num_r_emb, dims),
        )

        # ── Lorentzian Semantic Logical Cone (Section 4) ────────────
        self.use_projector = use_projector
        self.use_logic_cone = use_logic_cone
        self.use_consistency_gating = use_consistency_gating
        self.use_dynamic_cone = bool(use_dynamic_cone)
        self.use_bc_regularization = bool(use_bc_regularization)
        self.use_an_modulation = bool(use_an_modulation)
        self.use_lc_aggregation = bool(use_lc_aggregation)
        self.static_cone_theta = float(static_cone_theta)
        if use_projector:
            spatial_dim = dims - 1
            proj_dim = proj_dim or spatial_dim

            if r_text_embeds is not None:
                rel_dim = r_text_embeds.shape[1]
                self.register_buffer("r_text_fixed", r_text_embeds)
                self.r_text_embed = None
                self._r_text_llm_mode = True
            else:
                rel_dim = spatial_dim
                self.r_text_embed = nn.Embedding(self.num_r_emb, rel_dim)
                nn.init.normal_(self.r_text_embed.weight, std=1e-3)
                self._r_text_llm_mode = False

            self.projector = SemanticConeProjector(
                lorentz_dim=dims,
                rel_dim=rel_dim,
                output_dim=proj_dim,
            )

            # Scale / shift heads feeding the ⊛ operator (Eq. 13).
            # They take the aggregated neighbourhood Δh (in tangent space) as input.
            self.scale_head = nn.Linear(dims, spatial_dim, bias=False)
            self.shift_head = nn.Linear(dims, spatial_dim, bias=False)
            nn.init.normal_(self.scale_head.weight, std=1e-3)
            nn.init.normal_(self.shift_head.weight, std=1e-3)

            self.register_buffer("projector_delta_scale", torch.tensor(delta_scale))
            self.register_buffer("logic_cone_scale", torch.tensor(cone_scale))

        # ── Adaptive Multi-hop Reasoning (Eq. 14, 15) ───────────────
        self.use_multi_hop = use_multi_hop
        self.K_max = int(K_max)
        self.grad_ckpt = grad_ckpt
        self.prune_threshold = float(prune_threshold)
        self.pruned_nodes_epoch = 0
        self.pruned_nodes_total = 0
        self.candidate_nodes_epoch = 0
        self.candidate_nodes_total = 0
        self.masked_nodes_epoch = 0
        self.masked_nodes_total = 0
        self.attention_entropy_epoch = 0.0
        self.attention_entropy_total = 0.0
        self.attention_observations_epoch = 0
        self.attention_observations_total = 0
        self._collect_pruned_nodes = True
        # Learnable temperature Γ (softplus-positive) for hop-level softmax.
        self.raw_temperature = nn.Parameter(torch.tensor(1.0))
        # Residual coefficient for the tangent-space aggregation.
        self.register_buffer("agg_lambda", torch.tensor(float(agg_lambda)))

        # Pre-computed adjacency used during the per-hop cone filtration.
        # edge_index : LongTensor [2, E]; row 0 = src (message), row 1 = dst.
        if edge_index is not None:
            src = edge_index[0].long().contiguous()
            dst = edge_index[1].long().contiguous()
            deg = torch.zeros(self.num_e_emb, dtype=torch.float)
            deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
            self.register_buffer("edge_src", src)
            self.register_buffer("edge_dst", dst)
            self.register_buffer("deg_inv", 1.0 / deg.clamp_min_(1.0))
        else:
            self.edge_src = None
            self.edge_dst = None
            self.deg_inv = None

    def reset_pruned_nodes(self) -> None:
        self.pruned_nodes_epoch = 0
        self.candidate_nodes_epoch = 0
        self.masked_nodes_epoch = 0
        self.attention_entropy_epoch = 0.0
        self.attention_observations_epoch = 0

    def get_pruned_nodes(self) -> int:
        return int(self.pruned_nodes_epoch)

    def get_pruning_stats(self) -> dict[str, Union[float, int]]:
        rate = (
            self.pruned_nodes_epoch / self.candidate_nodes_epoch
            if self.candidate_nodes_epoch
            else 0.0
        )
        total_rate = (
            self.pruned_nodes_total / self.candidate_nodes_total
            if self.candidate_nodes_total
            else 0.0
        )
        masked_rate = (
            self.masked_nodes_epoch / self.candidate_nodes_epoch
            if self.candidate_nodes_epoch
            else 0.0
        )
        total_masked_rate = (
            self.masked_nodes_total / self.candidate_nodes_total
            if self.candidate_nodes_total
            else 0.0
        )
        attention_entropy = (
            self.attention_entropy_epoch / self.attention_observations_epoch
            if self.attention_observations_epoch
            else 0.0
        )
        total_attention_entropy = (
            self.attention_entropy_total / self.attention_observations_total
            if self.attention_observations_total
            else 0.0
        )
        return {
            "pruned_nodes": int(self.pruned_nodes_epoch),
            "candidate_nodes": int(self.candidate_nodes_epoch),
            "prune_rate": float(rate),
            "pruned_nodes_total": int(self.pruned_nodes_total),
            "candidate_nodes_total": int(self.candidate_nodes_total),
            "prune_rate_total": float(total_rate),
            "masked_nodes": int(self.masked_nodes_epoch),
            "masked_nodes_total": int(self.masked_nodes_total),
            "masked_rate": float(masked_rate),
            "masked_rate_total": float(total_masked_rate),
            "attention_entropy": float(attention_entropy),
            "attention_entropy_total": float(total_attention_entropy),
        }

    # ==================================================================
    # Public forward (unchanged FlorE calling convention)
    # ==================================================================
    def forward(self, u, r, v):
        if self.training:
            npos = v.shape[1]
            n1, p1 = None, None
            for i in range(npos):
                if len(u.shape) == 2:
                    u_idx = u[:, i]
                    t_idx = r[:, i]
                    v_idx = v[:, i, :]
                else:
                    u_idx = u[:, i, :]
                    t_idx = r[:, i]
                    v_idx = v[:, i]

                n_1 = self._forward(u_idx, t_idx, v_idx)
                if p1 is None:
                    p1 = n_1[:, 0:1]
                    n1 = n_1[:, 1:]
                else:
                    p1 = torch.cat([p1, n_1[:, 0:1]], dim=1)
                    n1 = torch.cat([n1, n_1[:, 1:]], dim=1)
                del n_1
            ndist = torch.cat([p1, n1], dim=1)
            del n1, p1
            return ndist
        else:
            return self._forward(u, r, v)

    def _get_r_text(self, r_idx: torch.Tensor) -> torch.Tensor:
        if self._r_text_llm_mode:
            return self.r_text_fixed[r_idx]
        return self.r_text_embed(r_idx)

    # ==================================================================
    # Logical-cone primitives  (Eq. 6-10, 13)
    # ==================================================================
    @staticmethod
    def _cone_energy(rel_dir: torch.Tensor,
                     neigh_dir: torch.Tensor,
                     theta: torch.Tensor) -> torch.Tensor:
        """
        Eq.(8) (sign-corrected):  E^i_u = ReLU( cos(θ^i) - cos(h_r, h_u) ).
        Neighbours whose direction is aligned with the relation within the
        aperture θ^i yield E = 0; those outside are penalised proportionally.
        """
        # rel_dir   : [B, d-1]
        # neigh_dir : [B, N, d-1]
        cos_rn = torch.sum(rel_dir.unsqueeze(1) * neigh_dir, dim=-1)       # [B, N]
        cos_theta = torch.cos(theta).unsqueeze(-1)                          # [B, 1]
        return F.relu(cos_theta - cos_rn)                                   # [B, N]

    def _semantic_modulation(
        self,
        v_prev: torch.Tensor,
        delta_h: torch.Tensor,
        energy_scalar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Eq.(13):  h^{i-1} ⊛ Δh = h^{i-1} ⊙ (1 + β·exp(-E)·tanh(Δh_1)) + β·tanh(Δh_2)
        All operations live in the tangent space at the origin.
        """
        beta = self.projector_delta_scale
        scale_raw = self.scale_head(delta_h)                                # [B, d-1]
        shift_raw = self.shift_head(delta_h)                                # [B, d-1]
        scale_term = beta * torch.tanh(scale_raw)
        shift_term = beta * torch.tanh(shift_raw)
        if self.use_consistency_gating:
            gate = torch.exp(-energy_scalar).unsqueeze(-1)                  # [B, 1]
        else:
            gate = torch.ones_like(scale_term[..., :1])
        v_spatial = v_prev[..., 1:]
        v_spatial_mod = v_spatial * (1.0 + gate * scale_term) + shift_term
        v_mod = torch.cat([torch.zeros_like(v_prev[..., :1]), v_spatial_mod], dim=-1)
        return v_mod

    # ==================================================================
    # One reasoning hop (Eq. 6-13)
    # ==================================================================
    def _one_hop(self,
                 h_prev: torch.Tensor,
                 r_idx: torch.Tensor,
                 u_idx: torch.Tensor,
                 v_all: Optional[torch.Tensor]):
        """
        Performs a single multi-hop step with semantic-guided cone filtration.

        Args
            h_prev : [B, dim]  current entity state h^{i-1}
            r_idx  : [B]       relation indices
            u_idx  : [B]       center node indices (used to fetch neighbours)
            v_all  : [N, dim]  precomputed log_0 of the entity table

        Returns
            h_cur, E_cur (scalar per sample), cone_state dict
        """
        B, dim = h_prev.shape
        r_text = self._get_r_text(r_idx)                                   # [B, rel_dim]

        # Step 1 — cone parameterisation from the *evolving* head h^{i-1}.
        cone = self.projector(h_prev.detach(), r_text)
        theta = cone["theta"]                                              # [B]
        axis = cone["axis"]
        rel_dir = cone["rel_dir"]                                          # [B, d-1]
        lam = cone["lam"]
        if not self.use_dynamic_cone:
            theta = torch.full_like(theta, self.static_cone_theta)
            axis = rel_dir

        # Step 2 — neighbour gather + cone filtration (Eq. 8-10)
        delta_h = torch.zeros(B, dim, device=h_prev.device, dtype=h_prev.dtype)
        E_scalar = torch.zeros(B, device=h_prev.device, dtype=h_prev.dtype)
        pruned_count = torch.zeros((), device=h_prev.device, dtype=torch.long)
        candidate_count = torch.zeros((), device=h_prev.device, dtype=torch.long)

        if self.edge_src is not None and v_all is not None:
            for b in range(B):
                mask = (self.edge_dst == u_idx[b])
                n_idx = self.edge_src[mask]
                if n_idx.numel() == 0:
                    continue
                if self.training and self._collect_pruned_nodes:
                    candidate_count = candidate_count + n_idx.numel()
                n_lorentz = self.emb_entity_manifold[n_idx]                # [K, dim]
                n_dir = F.normalize(n_lorentz[..., 1:], p=2, dim=-1)        # [K, d-1]

                # Eq. (8') — sign-corrected energy per neighbour.
                cos_rn = torch.sum(rel_dir[b].unsqueeze(0) * n_dir, dim=-1) # [K]
                if self.use_bc_regularization:
                    E_n = F.relu(torch.cos(theta[b]) - cos_rn)              # [K]
                else:
                    E_n = torch.zeros_like(cos_rn)

                # Eq. (10) — attention weights. Full/BC-enabled runs hard-mask
                # boundary-violating neighbours so ablations expose smoothing.
                if self.use_bc_regularization:
                    # Be permissive: only remove severe boundary violations.
                    # Masking every E_n > 0 was too strict and often discarded
                    # almost the whole neighbourhood.
                    boundary_threshold = E_n.detach().mean() + E_n.detach().std(unbiased=False)
                    hard_mask = E_n > boundary_threshold
                else:
                    hard_mask = torch.zeros_like(E_n, dtype=torch.bool)
                valid = ~hard_mask
                if self.use_an_modulation and torch.any(valid):
                    logits = -E_n * lam
                    logits = logits.masked_fill(hard_mask, -1e9)
                    w_n = F.softmax(logits, dim=-1)                         # [K]
                else:
                    w_n = torch.full_like(E_n, 1.0 / max(E_n.numel(), 1))
                if self.training and self._collect_pruned_nodes:
                    masked_count = torch.count_nonzero(hard_mask.detach())
                    masked_nodes = int(masked_count.item())
                    self.masked_nodes_epoch += masked_nodes
                    self.masked_nodes_total += masked_nodes
                    entropy = -(w_n.detach() * torch.log(w_n.detach().clamp_min(1e-12))).sum()
                    self.attention_entropy_epoch += float(entropy.item())
                    self.attention_entropy_total += float(entropy.item())
                    self.attention_observations_epoch += 1
                    self.attention_observations_total += 1
                    pruned_count = pruned_count + torch.count_nonzero(
                        w_n.detach() < self.prune_threshold
                    )

                # Eq. (9) — Δh in tangent space.
                n_tangent = v_all.index_select(0, n_idx)                    # [K, dim]
                delta_h[b] = torch.sum(w_n.unsqueeze(-1) * n_tangent, dim=0)

                # Per-sample scalar energy used by Eq. 13 and 15.
                E_scalar[b] = torch.sum(w_n * E_n)

        if self.training and self._collect_pruned_nodes:
            count = int(pruned_count.item())
            candidates = int(candidate_count.item())
            self.pruned_nodes_epoch += count
            self.pruned_nodes_total += count
            self.candidate_nodes_epoch += candidates
            self.candidate_nodes_total += candidates

        # Step 3 — semantic modulation + exp-map back to the manifold.
        v_prev = self.manifold.logmap0(h_prev)
        v_mod = self._semantic_modulation(v_prev, delta_h, E_scalar)
        # Eq. (12) with optional residual on Δh in T_0 L.
        if self.use_lc_aggregation:
            v_cur = v_mod + self.agg_lambda * delta_h
            # Keep time-component of v_cur at 0 (T_0 L guarantee).
            v_cur = torch.cat([torch.zeros_like(v_cur[..., :1]), v_cur[..., 1:]], dim=-1)
            h_cur = self.manifold.expmap0(v_cur)
        else:
            # Ablation: keep message passing but remove Lorentz-constrained
            # modulation/residual gating, which is intentionally prone to
            # smoothing under repeated neighbour averaging.
            v_cur = torch.cat([torch.zeros_like(delta_h[..., :1]), delta_h[..., 1:]], dim=-1)
            h_cur = self.manifold.expmap0(v_cur)

        cone_state = {
            "theta": theta,
            "axis": axis,
            "rel_dir": rel_dir,
            "energy": E_scalar,
        }
        return h_cur, E_scalar, cone_state

    # ==================================================================
    # Multi-hop orchestration (Eq. 14-15)
    # ==================================================================
    def _multi_hop_reason(self, h0: torch.Tensor,
                          u_idx: torch.Tensor, r_idx: torch.Tensor):
        h_states, energies, cone_states = [], [], []

        v_all = None
        if self.edge_src is not None:
            v_all = self.manifold.logmap0(self.emb_entity_manifold)        # [N, dim]

        # Hop 0 — plain refinement (no neighbourhood agg) so that Eq.(14)
        # can also mix the raw encoded head.
        r_text = self._get_r_text(r_idx)
        cone0 = self.projector(h0.detach(), r_text)
        E0 = torch.zeros(h0.size(0), device=h0.device, dtype=h0.dtype)
        h_states.append(h0)
        energies.append(E0)
        cone_states.append({
            "theta": cone0["theta"],
            "axis": cone0["axis"],
            "rel_dir": cone0["rel_dir"],
            "energy": E0,
        })

        for _ in range(1, self.K_max + 1):
            h_prev = h_states[-1]
            if self.grad_ckpt and self.training:
                from torch.utils.checkpoint import checkpoint
                self._collect_pruned_nodes = True
                h_cur, E_cur, cs_cur = checkpoint(
                    lambda _h: self._one_hop(_h, r_idx, u_idx, v_all),
                    h_prev, use_reentrant=False,
                )
                # Backward recomputation calls the lambda directly; keep counting off
                # outside the original forward pass to avoid double-counting.
                self._collect_pruned_nodes = False
            else:
                self._collect_pruned_nodes = True
                h_cur, E_cur, cs_cur = self._one_hop(h_prev, r_idx, u_idx, v_all)
            h_states.append(h_cur)
            energies.append(E_cur)
            cone_states.append(cs_cur)

        # Eq. (15) — adaptive hop-level softmax over negative energies.
        E_stack = torch.stack(energies, dim=0)                              # [K+1, B]
        gamma = F.softplus(self.raw_temperature) + 1e-4
        alpha = F.softmax(-E_stack / gamma, dim=0)                          # [K+1, B]

        # Eq. (14) — fuse in tangent space for manifold safety.
        v_stack = torch.stack(
            [self.manifold.logmap0(h) for h in h_states], dim=0,
        )                                                                   # [K+1, B, dim]
        v_final = (alpha.unsqueeze(-1) * v_stack).sum(dim=0)                # [B, dim]
        h_final = self.manifold.expmap0(v_final)                            # on Lorentz

        return h_final, cone_states[-1]

    # ==================================================================
    # Triple-level cone bonus used at scoring time
    # ==================================================================
    def _semantic_logic_cone_bonus(
        self,
        head_lorentz: torch.Tensor,
        tail_lorentz: torch.Tensor,
        cone_state: dict,
    ) -> torch.Tensor:
        """
        Triple-level geometric bonus at the last hop.
            bonus = tanh( ReLU( cos(a^i, d_t) - cos(θ^i) ) )
        where a^i is the relation-transformed head direction (here we
        simply reuse the current cone axis) and d_t is the spatial direction
        of the candidate tail.
        """
        if tail_lorentz.dim() == 2:
            tail_lorentz = tail_lorentz.unsqueeze(1)

        axis = cone_state.get("axis")
        if axis is None:
            axis = head_lorentz[:, 1:]
        axis = F.normalize(axis, p=2, dim=-1)                              # [B, d-1]
        tail_dir = F.normalize(tail_lorentz[..., 1:], p=2, dim=-1)         # [B, n, d-1]
        cos_alpha = torch.sum(axis.unsqueeze(1) * tail_dir, dim=-1)        # [B, n]
        threshold = torch.cos(cone_state["theta"]).unsqueeze(-1)           # [B, 1]
        cone_bonus = F.relu(cos_alpha - threshold)
        return torch.tanh(cone_bonus)

    # ==================================================================
    # Inner forward
    # ==================================================================
    def _forward(self, u_idx, r_idx, v_idx):
        h = self.emb_entity_manifold[u_idx]                                # [batch, dim]
        t = self.emb_entity_manifold[v_idx]                                # [batch, nneg+1, dim]
        cone_state = None

        if self.use_projector and h.dim() == 2:
            if self.use_multi_hop:
                h, cone_state = self._multi_hop_reason(h, u_idx, r_idx)
            else:
                r_text = self._get_r_text(r_idx)
                cone = self.projector(h.detach(), r_text)
                cone_state = {
                    "theta": cone["theta"],
                    "axis": cone["axis"],
                    "rel_dir": cone["rel_dir"],
                    "energy": torch.zeros(h.size(0), device=h.device, dtype=h.dtype),
                }

        if len(h.shape) == 2:
            h = h.unsqueeze(1)
            u_idx = u_idx.unsqueeze(1)
        elif len(t.shape) == 2:
            t = t.unsqueeze(1)
            v_idx = v_idx.unsqueeze(1)

        transformed_h, *_ = self.head_linear((h, r_idx))
        transformed_t, *_ = self.tail_linear((t, r_idx))

        mkv_interval = self.manifold.cinner2(
            (transformed_t - transformed_h), (transformed_t - transformed_h)
        ).squeeze()

        cone_bonus = 0.0
        if self.use_projector and self.use_logic_cone and cone_state is not None:
            cone_bonus = self._semantic_logic_cone_bonus(
                transformed_h.squeeze(1), transformed_t, cone_state,
            ).squeeze()

        _dev = self.bias_head.device
        if self.training:
            rnd_regular_head = self.noise_reg * torch.randn(
                (mkv_interval.shape[0], 1), device=_dev, requires_grad=False,
            )
        else:
            rnd_regular_head = torch.zeros(1, dtype=torch.float, device=_dev, requires_grad=False)

        int_dist = (
            self.margin - mkv_interval
            + torch.tanh(self.bias_head[u_idx])
            + rnd_regular_head
            + torch.tanh(self.bias_tail[v_idx])
        )
        if self.use_projector and self.use_logic_cone and cone_state is not None:
            int_dist = int_dist + self.logic_cone_scale * cone_bonus

        return int_dist
