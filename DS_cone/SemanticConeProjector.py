import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticConeProjector(nn.Module):
    """
    Dynamic Sequential Lorentz Entailment Cone (Section 4.1 of SIHR paper).

    For a node v at the i-th reasoning hop with current state h^{i-1}_v and
    relation r, this module produces:

      (Eq.6) theta^i  = Sigmoid( MLP(h_r) * ||h_spatial|| ) * pi       # aperture
      (Eq.7) a^i      = h_spatial / ||h_spatial||                      # axis (unit)

    We also provide two tangent-space delta heads that later feed the
    semantic modulation operator  \boxdot  in Eq.(13).
    """

    def __init__(self, lorentz_dim: int, rel_dim: int, output_dim: int):
        super().__init__()
        self.lorentz_dim = lorentz_dim
        self.spatial_dim = lorentz_dim - 1
        self.output_dim = output_dim

        # Eq.(6) scalar head producing a per-sample aperture coefficient
        hidden = max(16, min(4 * output_dim, 256))
        self.aperture_head = nn.Sequential(
            nn.Linear(rel_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        # Project relation semantic vector into the spatial direction space
        # so that cosine similarity with the neighbour direction is well defined.
        self.rel_direction_proj = nn.Linear(rel_dim, self.spatial_dim)

        # Learnable inverse temperature λ for Eq.(10)
        self.raw_lambda = nn.Parameter(torch.tensor(1.0))

    # ------------------------------------------------------------------
    def compute_axis(self, h_lorentz: torch.Tensor) -> torch.Tensor:
        """Eq.(7): a^i = h_spatial / ||h_spatial||  (unit vector)."""
        return F.normalize(h_lorentz[..., 1:], p=2, dim=-1)

    def compute_aperture(self, r_embed: torch.Tensor, h_lorentz: torch.Tensor) -> torch.Tensor:
        """Eq.(6): theta^i = Sigmoid( MLP(h_r) * ||h_spatial|| ) * pi  (scalar per sample)."""
        scalar_r = self.aperture_head(r_embed).squeeze(-1)                  # [B]
        h_norm = torch.norm(h_lorentz[..., 1:], p=2, dim=-1)                # [B]
        return torch.sigmoid(scalar_r * h_norm) * math.pi                   # [B]

    def rel_direction(self, r_embed: torch.Tensor) -> torch.Tensor:
        """Project the semantic relation vector to the spatial unit direction."""
        return F.normalize(self.rel_direction_proj(r_embed), p=2, dim=-1)

    # ------------------------------------------------------------------
    def forward(self, h_lorentz: torch.Tensor, r_embed: torch.Tensor):
        """
        Args
            h_lorentz : [B, lorentz_dim]   entity on the hyperboloid
            r_embed   : [B, rel_dim]       (LLM-derived) relation embedding
        Returns
            cone_state : dict with
                - theta   : [B]         dynamic aperture
                - axis    : [B, d-1]    dynamic axis (unit vector, Eq. 7)
                - rel_dir : [B, d-1]    normalized relation direction
                - lam     : scalar      softplus(raw_lambda)
        """
        theta = self.compute_aperture(r_embed, h_lorentz)
        axis = self.compute_axis(h_lorentz)
        rel_dir = self.rel_direction(r_embed)
        lam = F.softplus(self.raw_lambda) + 1e-4

        return {
            "theta": theta,
            "axis": axis,
            "rel_dir": rel_dir,
            "lam": lam,
        }

