"""
time_omni_vae.py — Core generative engine for synthetic tabular time-series data.

Addresses three industrial data challenges:

1. **Weak Temporal Relations** — A residual decoder architecture projects the
   latent code *z* to a static base and lets the GRU learn only temporal
   residuals that are added back.  This avoids hallucinating sequential
   dependencies in quasi-static sensor data.

2. **Raw Missing Values (NaN)** — NaNs are zeroed at the input gate so the RNN
   never sees them, but reconstruction loss is masked: the gradient flows only
   through observed positions.  Mathematically the masked MSE is

       L_recon = Σ  m_ij · (x_ij − x̂_ij)² / Σ m_ij

   where m_ij = 1 iff x_ij is observed.  This prevents the model from being
   rewarded (or penalised) for imputing sensor gaps.

3. **Multi-modal Joint Distribution** — One-hot operational modes concatenated
   with continuous features form a mixed distribution.  A learnable set of
   cluster centres in latent space pulls encodings towards discrete modes via a
   soft nearest-neighbour penalty, keeping the modes separable during generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TimeOmniVAEConfig:
    """Hyper-parameters and loss weights for :class:`TimeOmniVAE`."""

    # Architecture
    input_dim: int = 1
    latent_dim: int = 16
    rnn_hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1

    # Loss weights
    beta: float = 1.0
    """Weight for the KL divergence term (β-VAE style)."""
    alpha: float = 0.5
    """Weight for the latent clustering penalty."""
    lambda_temporal: float = 0.1
    """Weight for the temporal consistency (first-order derivative) loss."""
    lambda_phys: float = 0.0
    """Reserved weight for an optional physics-informed penalty."""

    # Clustering
    learn_cluster_centers: bool = True
    """If *True*, cluster centres are registered as learnable parameters."""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TimeOmniVAE(nn.Module):
    """Variational auto-encoder for tabular time-series with a residual decoder.

    **Encoder** — A multi-layer GRU reads the (possibly NaN-sanitised) input
    sequence and maps the final hidden state to ``mu`` and ``logvar``.

    **Decoder (residual)** — The sampled latent vector *z* is projected to a
    *static base* of shape ``(B, Features)`` that captures the time-invariant
    component.  Simultaneously *z* is repeated across the sequence length and
    fed through a decoder GRU whose output is projected to a *temporal
    residual*.  The reconstruction is ``static_base + temporal_residual``,
    giving the GRU freedom to model *only* the sequential deviations — critical
    when true temporal structure is weak.
    """

    def __init__(self, config: TimeOmniVAEConfig) -> None:
        super().__init__()
        self.config = config
        d = config.input_dim
        h = config.rnn_hidden_dim
        z = config.latent_dim
        n = config.num_layers
        p = config.dropout

        # --- Encoder ---
        self.encoder_gru = nn.GRU(
            input_size=d,
            hidden_size=h,
            num_layers=n,
            batch_first=True,
            dropout=p if n > 1 else 0.0,
        )
        self.fc_mu = nn.Linear(h, z)
        self.fc_logvar = nn.Linear(h, z)

        # --- Decoder ---
        self.decoder_gru = nn.GRU(
            input_size=z,
            hidden_size=h,
            num_layers=n,
            batch_first=True,
            dropout=p if n > 1 else 0.0,
        )
        # Static base projection (time-invariant component)
        self.fc_static_base = nn.Linear(z, d)
        # Temporal residual projection (time-varying component)
        self.fc_temporal_residual = nn.Linear(h, d)

    # ----- utilities -----

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample *z* via the Gaussian reparameterisation trick.

        z = μ + σ · ε,   ε ~ N(0, I)

        During evaluation ``ε`` is still sampled so that generation is
        stochastic; call ``torch.no_grad()`` externally if deterministic
        behaviour is desired.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    # ----- forward -----

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a NaN-sanitised sequence to ``(mu, logvar)``.

        Parameters
        ----------
        x : Tensor of shape ``(B, T, D)``
            Input sequence.  **Must already be NaN-free** (use
            ``torch.nan_to_num`` before calling).
        """
        _, h_n = self.encoder_gru(x)  # h_n: (num_layers, B, H)
        h_last = h_n[-1]  # (B, H)
        return self.fc_mu(h_last), self.fc_logvar(h_last)

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode latent *z* into a reconstructed sequence.

        The static base captures the time-invariant "average" of the window,
        while the GRU learns only the residual temporal fluctuations.

        Parameters
        ----------
        z : Tensor ``(B, Z)``
        seq_len : int
            Length of the output sequence.

        Returns
        -------
        Tensor ``(B, T, D)``
        """
        static_base = self.fc_static_base(z)  # (B, D)

        z_repeated = z.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, Z)
        gru_out, _ = self.decoder_gru(z_repeated)  # (B, T, H)
        temporal_residual = self.fc_temporal_residual(gru_out)  # (B, T, D)

        return static_base.unsqueeze(1) + temporal_residual  # (B, T, D)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass with NaN sanitisation at the input gate.

        Parameters
        ----------
        x : Tensor ``(B, T, D)``
            Raw input that **may contain NaN** values.

        Returns
        -------
        x_recon : Tensor ``(B, T, D)``
        mu      : Tensor ``(B, Z)``
        logvar  : Tensor ``(B, Z)``
        """
        x_safe = torch.nan_to_num(x, nan=0.0)
        mu, logvar = self.encode(x_safe)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, seq_len=x.size(1))
        return x_recon, mu, logvar


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def unified_time_vae_loss(
    x_recon: torch.Tensor,
    x_target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    config: TimeOmniVAEConfig,
    cluster_centers: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the composite VAE objective.

    All sub-losses are NaN-safe: positions where ``x_target`` is NaN are
    excluded from every term that touches the observation space.

    Parameters
    ----------
    x_recon : ``(B, T, D)``
    x_target : ``(B, T, D)`` — may contain NaN.
    mu, logvar : ``(B, Z)``
    config : :class:`TimeOmniVAEConfig`
    cluster_centers : ``(K, Z)`` or *None*

    Returns
    -------
    total_loss : scalar Tensor
    metrics : dict mapping loss names to Python floats
    """
    # ---- NaN mask ----
    valid_mask = ~torch.isnan(x_target)  # (B, T, D)
    target_safe = torch.nan_to_num(x_target, nan=0.0)

    # ---- Reconstruction (masked MSE) ----
    sq_err = (x_recon - target_safe) ** 2  # (B, T, D)
    recon_loss = (sq_err * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

    # ---- KL divergence  D_KL(q(z|x) || p(z))  ----
    kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

    # ---- Temporal consistency (NaN-safe first-order derivative MSE) ----
    temporal_loss = torch.tensor(0.0, device=x_recon.device)
    if config.lambda_temporal > 0.0 and x_target.size(1) > 1:
        # First-order differences along the time axis
        dx_recon = x_recon[:, 1:, :] - x_recon[:, :-1, :]
        dx_target = target_safe[:, 1:, :] - target_safe[:, :-1, :]
        # A derivative value is valid only when *both* adjacent time-steps are
        # observed, so the mask for Δx is the logical AND of consecutive masks.
        diff_mask = valid_mask[:, 1:, :] & valid_mask[:, :-1, :]
        sq_diff = (dx_recon - dx_target) ** 2
        temporal_loss = (
            (sq_diff * diff_mask).sum() / diff_mask.sum().clamp(min=1.0)
        )

    # ---- Cluster penalty ----
    cluster_loss = torch.tensor(0.0, device=mu.device)
    if config.alpha > 0.0 and cluster_centers is not None:
        z = TimeOmniVAE.reparameterize(mu, logvar)
        # Pairwise L2² distances: (B, K)
        dists = torch.cdist(z.unsqueeze(0), cluster_centers.unsqueeze(0)).squeeze(0)
        # Minimum distance to any centre
        cluster_loss = dists.min(dim=1).values.mean()

    # ---- Aggregate ----
    total_loss = (
        recon_loss
        + config.beta * kl_loss
        + config.lambda_temporal * temporal_loss
        + config.alpha * cluster_loss
    )

    metrics: Dict[str, float] = {
        "loss/total": total_loss.item(),
        "loss/recon": recon_loss.item(),
        "loss/kl": kl_loss.item(),
        "loss/temporal": temporal_loss.item(),
        "loss/cluster": cluster_loss.item(),
    }
    return total_loss, metrics


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TimeOmniVAETrainer:
    """Training harness for :class:`TimeOmniVAE`.

    Handles learnable cluster centres, NaN-safe training loops, and
    cluster-aware generation.

    Parameters
    ----------
    model : TimeOmniVAE
    config : dict or TimeOmniVAEConfig
    device : torch.device or str
    num_clusters : int
        Number of latent cluster centres.  If > 1, centres are initialised
        as a learnable ``nn.Parameter``.
    lr : float
    """

    def __init__(
        self,
        model: TimeOmniVAE,
        config: Dict[str, Any] | TimeOmniVAEConfig,
        device: torch.device | str = "cpu",
        num_clusters: int = 1,
        lr: float = 1e-3,
    ) -> None:
        if isinstance(config, dict):
            self.config = TimeOmniVAEConfig(**config)
        else:
            self.config = config

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.num_clusters = num_clusters

        # Learnable cluster centres
        self.cluster_centers: Optional[nn.Parameter] = None
        if num_clusters > 1 and self.config.alpha > 0.0:
            init = torch.randn(num_clusters, self.config.latent_dim) * 0.5
            if self.config.learn_cluster_centers:
                self.cluster_centers = nn.Parameter(init.to(self.device))
            else:
                self.cluster_centers = nn.Parameter(
                    init.to(self.device), requires_grad=False
                )

        # Build parameter list including cluster centres
        params = list(self.model.parameters())
        if self.cluster_centers is not None and self.cluster_centers.requires_grad:
            params.append(self.cluster_centers)
        self.optimizer = torch.optim.Adam(params, lr=lr)

    # ------------------------------------------------------------------ #

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run a single training epoch.

        The ``DataLoader`` must yield tensors of shape ``(B, T, D)`` which
        may contain NaN values.
        """
        self.model.train()
        accum: Dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            # Support DataLoader returning tuples (take first element)
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device)

            x_recon, mu, logvar = self.model(batch)
            loss, metrics = unified_time_vae_loss(
                x_recon,
                batch,
                mu,
                logvar,
                self.config,
                cluster_centers=self.cluster_centers,
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            for k, v in metrics.items():
                accum[k] = accum.get(k, 0.0) + v
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in accum.items()}

    # ------------------------------------------------------------------ #

    def fit(
        self,
        loader: DataLoader,
        epochs: int = 50,
        verbose: bool = True,
    ) -> list[Dict[str, float]]:
        """Full training loop over multiple epochs.

        Parameters
        ----------
        loader : DataLoader yielding ``(B, T, D)`` tensors.
        epochs : Number of epochs.
        verbose : Print per-epoch summary.

        Returns
        -------
        List of per-epoch metric dictionaries.
        """
        history: list[Dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch(loader)
            history.append(metrics)
            if verbose:
                parts = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                print(f"[Epoch {epoch:>3d}/{epochs}] {parts}")
        return history

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(self, num_samples: int, seq_len: int) -> torch.Tensor:
        """Generate synthetic time-series samples.

        If clustering is active (``alpha > 0`` and centres exist), each sample
        is seeded by uniformly choosing a cluster centre and adding Gaussian
        noise (σ = 0.25).  This keeps the generated distribution faithful to
        the learned multi-modal structure (e.g. distinct operational modes).

        Otherwise, *z* is sampled from the standard normal prior N(0, I).

        Parameters
        ----------
        num_samples : int
        seq_len : int

        Returns
        -------
        Tensor ``(num_samples, seq_len, D)`` on CPU.
        """
        self.model.eval()
        z_dim = self.config.latent_dim

        if (
            self.config.alpha > 0.0
            and self.cluster_centers is not None
            and self.num_clusters > 1
        ):
            # Uniformly assign each sample to a cluster
            indices = torch.randint(0, self.num_clusters, (num_samples,))
            centers = self.cluster_centers[indices]  # (N, Z)
            noise = torch.randn_like(centers) * 0.25
            z = centers + noise
        else:
            z = torch.randn(num_samples, z_dim, device=self.device)

        z = z.to(self.device)
        x_synthetic = self.model.decode(z, seq_len)
        return x_synthetic.cpu()
