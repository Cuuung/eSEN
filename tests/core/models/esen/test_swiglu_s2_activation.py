"""Smoke test for SwiGLU_S2_Activation: forward shape, equivariance, autograd."""

from __future__ import annotations

import torch
from e3nn import o3

from fairchem.core.models.esen.nn.activation import (
    SwiGLU_S2_Activation,
    _SwiGLU,
)


def _make_wigner_d(rot: torch.Tensor, lmax: int) -> torch.Tensor:
    irreps = o3.Irreps("+".join(f"1x{l}e" for l in range(lmax + 1)))
    return irreps.D_from_matrix(rot)


def test_forward_shape():
    lmax, mmax, c, n = 2, 2, 32, 8
    act = SwiGLU_S2_Activation(lmax=lmax, mmax=mmax, num_channels=c)
    inputs = torch.randn(n, (lmax + 1) ** 2, 2 * c)
    scalars = torch.randn(n, 3 * c)
    out = act(scalars, inputs)
    assert out.shape == (n, (lmax + 1) ** 2, c)


def test_swiglu_halves_channels():
    swiglu = _SwiGLU()
    x = torch.randn(4, 16)
    y = swiglu(x)
    assert y.shape == (4, 8)
    x_1, x_2 = torch.chunk(x, 2, dim=-1)
    expected = torch.nn.functional.silu(x_1) * x_2
    assert torch.allclose(y, expected)


def test_equivariance(seed: int, lmax: int):
    """act(D x, s) ≡ D act(x, s) up to grid aliasing.

    Scalars (gating_scalars) are L=0-derived and rotation-invariant in the
    real model; here we use random scalars and keep them fixed under rotation.
    """
    torch.manual_seed(seed)
    mmax, c, n = lmax, 16, 4
    act = SwiGLU_S2_Activation(lmax=lmax, mmax=mmax, num_channels=c).double()
    inputs = torch.randn(n, (lmax + 1) ** 2, 2 * c, dtype=torch.float64)
    scalars = torch.randn(n, 3 * c, dtype=torch.float64)

    rot = o3.rand_matrix().double()
    wigner = _make_wigner_d(rot, lmax).double()

    inputs_rot = torch.einsum("ji,nic->njc", wigner, inputs)
    out_rot_first = act(scalars, inputs_rot)

    out_act_first = act(scalars, inputs)
    out_then_rot = torch.einsum("ji,nic->njc", wigner, out_act_first)

    diff = (out_rot_first - out_then_rot).abs()
    # 1e-6 in float64 corresponds to a tight grid; aliasing would surface as
    # diff >> 1e-3.
    assert diff.max() < 1e-6, (
        f"equivariance broken: lmax={lmax}, seed={seed}, "
        f"max_diff={diff.max():.3e}, mean_diff={diff.mean():.3e}"
    )


def test_autograd_through_inputs():
    """Conservative force training requires backward through bilinear path."""
    lmax, mmax, c, n = 2, 2, 16, 4
    act = SwiGLU_S2_Activation(lmax=lmax, mmax=mmax, num_channels=c)
    inputs = torch.randn(n, (lmax + 1) ** 2, 2 * c, requires_grad=True)
    scalars = torch.randn(n, 3 * c, requires_grad=True)
    out = act(scalars, inputs)
    out.sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert scalars.grad is not None and torch.isfinite(scalars.grad).all()


if __name__ == "__main__":
    test_forward_shape()
    test_swiglu_halves_channels()
    for s in range(5):
        for l in (1, 2):
            test_equivariance(s, l)
    test_autograd_through_inputs()
    print("All SwiGLU-S2 smoke tests passed.")
