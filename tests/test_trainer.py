"""
Tests for PINNTrainer — training pipeline integration.
"""

import pytest
import torch
from src.training.trainer import PINNTrainer, TrainerConfig, _data_loss, _pde_loss_chunk
from src.geometry.geometry import OblateSpheroid, sdf_oblate_spheroid, sdf_weight
from src.network.pinn_model import StreamingPINN


@pytest.fixture
def cfg():
    return TrainerConfig(
        n_epochs=3,
        n_pde_pts=60,
        mini_batch_size=30,
        hidden=16,
        layers=2,
        log_every=1,
    )


@pytest.fixture
def trainer(cfg):
    torch.manual_seed(0)
    return PINNTrainer(cfg)


@pytest.fixture
def mock_piv():
    torch.manual_seed(1)
    xyz = torch.rand(20, 3) * 2e-3 + 5e-4
    u   = torch.randn(20, 2) * 1e-3
    return xyz, u


# ---------------------------------------------------------------------------
# Trainer initialisation
# ---------------------------------------------------------------------------

def test_trainer_device_is_cpu(trainer):
    assert trainer.device.type == 'cpu'


def test_model_on_correct_device(trainer):
    p = next(trainer.model.parameters())
    assert p.device.type == 'cpu'


# ---------------------------------------------------------------------------
# Data loss
# ---------------------------------------------------------------------------

def test_data_loss_scalar(trainer, mock_piv):
    xyz_piv, u_piv = mock_piv
    xyz_leaf = xyz_piv.float().requires_grad_(True)
    loss = _data_loss(trainer.model, xyz_leaf, u_piv.float())
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_data_loss_2c_and_3c(trainer):
    """Data loss must work for both 2-component and 3-component PIV."""
    torch.manual_seed(2)
    xyz = torch.rand(10, 3, requires_grad=True).float()
    for n_comp in (2, 3):
        u = torch.randn(10, n_comp).float()
        loss = _data_loss(trainer.model, xyz, u)
        assert torch.isfinite(loss), f"Non-finite loss for {n_comp}C PIV"


# ---------------------------------------------------------------------------
# PDE loss chunk
# ---------------------------------------------------------------------------

def test_pde_loss_chunk_scalar(trainer):
    torch.manual_seed(3)
    xyz = torch.rand(30, 3, requires_grad=True).float() * 2e-3
    phi = sdf_oblate_spheroid(
        xyz.double(), trainer.geom
    ).float().detach()
    loss = _pde_loss_chunk(trainer.model, xyz, phi, trainer.cfg.nu, trainer.geom)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_pde_loss_sdf_weight_applied(trainer):
    """Points near the surface (phi~0) should have higher weight than far points."""
    geom = trainer.geom
    phi_near = torch.tensor([0.0, 1e-5], dtype=torch.float32)
    phi_far  = torch.tensor([geom.R_e, 2 * geom.R_e], dtype=torch.float32)
    w_near = sdf_weight(phi_near.double(), geom=geom).float()
    w_far  = sdf_weight(phi_far.double(),  geom=geom).float()
    assert w_near.mean() > w_far.mean()


# ---------------------------------------------------------------------------
# Gradient aggregation: verify grad accumulation equals full-batch grad
# ---------------------------------------------------------------------------

def test_gradient_aggregation_equivalence():
    """
    Gradient aggregation over K mini-batches must produce the same parameter
    gradients as a single full-batch backward pass (up to float32 tolerance).
    """
    torch.manual_seed(42)
    cfg_full = TrainerConfig(n_pde_pts=60, mini_batch_size=60, hidden=16, layers=2)
    cfg_agg  = TrainerConfig(n_pde_pts=60, mini_batch_size=20, hidden=16, layers=2)

    # Share identical weights
    model_full = StreamingPINN(hidden=16, layers=2)
    model_agg  = StreamingPINN(hidden=16, layers=2)
    model_agg.load_state_dict(model_full.state_dict())

    geom = OblateSpheroid(R_e=cfg_full.R_e, R_p=cfg_full.R_p)
    device = torch.device('cpu')

    # Same collocation points for both
    from src.geometry.geometry import sample_volume
    xyz_pde, phi_pde = sample_volume(geom, 60, device, exclude_interior=True, seed=7)
    xyz_pde = xyz_pde.float()
    phi_pde = phi_pde.float()

    # Full batch
    model_full.zero_grad()
    xyz_full = xyz_pde.detach().requires_grad_(True)
    loss_full = _pde_loss_chunk(model_full, xyz_full, phi_pde, cfg_full.nu, geom)
    loss_full.backward()

    # Aggregated (3 mini-batches of 20)
    model_agg.zero_grad()
    K = 3
    for start in range(0, 60, 20):
        chunk = xyz_pde[start:start+20].detach().requires_grad_(True)
        phi_c = phi_pde[start:start+20]
        loss_c = _pde_loss_chunk(model_agg, chunk, phi_c, cfg_agg.nu, geom)
        (loss_c / K).backward()

    # Compare gradients of first linear layer
    g_full = list(model_full.parameters())[0].grad
    g_agg  = list(model_agg.parameters())[0].grad
    assert g_full is not None and g_agg is not None
    assert torch.allclose(g_full, g_agg, atol=1e-5), \
        f"Gradient aggregation mismatch: max diff = {(g_full - g_agg).abs().max():.2e}"


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def test_train_returns_history(trainer, mock_piv):
    xyz_piv, u_piv = mock_piv
    history = trainer.train(xyz_piv, u_piv)
    assert len(history) == trainer.cfg.n_epochs


def test_train_losses_finite(trainer, mock_piv):
    xyz_piv, u_piv = mock_piv
    history = trainer.train(xyz_piv, u_piv)
    for row in history:
        assert torch.isfinite(torch.tensor(row['total'])), \
            f"Non-finite total loss at epoch {row['epoch']}"


def test_train_history_keys(trainer, mock_piv):
    xyz_piv, u_piv = mock_piv
    history = trainer.train(xyz_piv, u_piv)
    for row in history:
        assert {'epoch', 'data', 'pde', 'total'} <= row.keys()


def test_train_parameters_update(trainer, mock_piv):
    """Model parameters must change after at least one training epoch."""
    params_before = [p.clone() for p in trainer.model.parameters()]
    xyz_piv, u_piv = mock_piv
    trainer.train(xyz_piv, u_piv)
    params_after = list(trainer.model.parameters())
    changed = any(
        not torch.equal(b, a) for b, a in zip(params_before, params_after)
    )
    assert changed, "No parameter update detected after training"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
