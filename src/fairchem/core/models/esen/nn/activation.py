"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

from ..common.so3 import SO3_Grid


class GateActivation(torch.nn.Module):
    def __init__(self, lmax: int, mmax: int, num_channels: int) -> None:
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax
        self.num_channels = num_channels

        # compute `expand_index` based on `lmax` and `mmax`
        num_components = 0
        for lval in range(1, self.lmax + 1):
            num_m_components = min((2 * lval + 1), (2 * self.mmax + 1))
            num_components = num_components + num_m_components
        expand_index = torch.zeros([num_components]).long()
        start_idx = 0
        for lval in range(1, self.lmax + 1):
            length = min((2 * lval + 1), (2 * self.mmax + 1))
            expand_index[start_idx : (start_idx + length)] = lval - 1
            start_idx = start_idx + length
        self.register_buffer("expand_index", expand_index)

        self.scalar_act = (
            torch.nn.SiLU()
        )  # SwiGLU(self.num_channels, self.num_channels)  # #
        self.gate_act = torch.nn.Sigmoid()  # torch.nn.SiLU() # #

    def forward(self, gating_scalars, input_tensors):
        """
        `gating_scalars`: shape [N, lmax * num_channels]
        `input_tensors`: shape  [N, (lmax + 1) ** 2, num_channels]
        """

        gating_scalars = self.gate_act(gating_scalars)
        gating_scalars = gating_scalars.reshape(
            gating_scalars.shape[0], self.lmax, self.num_channels
        )
        gating_scalars = torch.index_select(
            gating_scalars, dim=1, index=self.expand_index
        )

        input_tensors_scalars = input_tensors.narrow(1, 0, 1)
        input_tensors_scalars = self.scalar_act(input_tensors_scalars)

        input_tensors_vectors = input_tensors.narrow(1, 1, input_tensors.shape[1] - 1)
        input_tensors_vectors = input_tensors_vectors * gating_scalars

        return torch.cat((input_tensors_scalars, input_tensors_vectors), dim=1)


class S2Activation(torch.nn.Module):
    """
    Assume we only have one resolution
    """

    def __init__(self, lmax: int, mmax: int, SO3_grid) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.act = torch.nn.SiLU()
        self.SO3_grid = SO3_grid

    def forward(self, inputs):
        to_grid_mat = self.SO3_grid["lmax_mmax"].get_to_grid_mat()
        from_grid_mat = self.SO3_grid["lmax_mmax"].get_from_grid_mat()
        x_grid = torch.einsum("bai, zic -> zbac", to_grid_mat, inputs)
        x_grid = self.act(x_grid)
        return torch.einsum("bai, zbac -> zic", from_grid_mat, x_grid)


class SeparableS2Activation(torch.nn.Module):
    def __init__(self, lmax: int, mmax: int, SO3_grid) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.scalar_act = torch.nn.SiLU()
        self.s2_act = S2Activation(self.lmax, self.mmax, SO3_grid)

    def forward(self, input_scalars, input_tensors):
        output_scalars = self.scalar_act(input_scalars)
        output_scalars = output_scalars.reshape(
            output_scalars.shape[0], 1, output_scalars.shape[-1]
        )
        output_tensors = self.s2_act(input_tensors)
        return torch.cat(
            (
                output_scalars,
                output_tensors.narrow(1, 1, output_tensors.shape[1] - 1),
            ),
            dim=1,
        )


class _SwiGLU(torch.nn.Module):
    """SwiGLU(x) = SiLU(x_1) * x_2 where (x_1, x_2) = chunk(x, 2, dim=-1).

    Output channels = input channels // 2.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1, x_2 = torch.chunk(x, 2, dim=-1)
        return torch.nn.functional.silu(x_1) * x_2


class SwiGLU_S2_Activation(torch.nn.Module):
    """EquiformerV3 SwiGLU-S2 activation, paper Eq. 7.

    Implements ``SeparableGateS2Activation_SwiGLU_Merge`` from
    equiformer_v3/experimental/models/equiformer_v3/activation.py:419 in the
    eSEN convention (gating scalars passed first).

    Forward pass:
        1. Split ``gating_scalars`` into a SwiGLU branch (2C) and a Sigmoid
           gating branch (C).
        2. ``SwiGLU(swiglu_in) -> [N, 1, C]`` for the L=0 path.
        3. ``Sigmoid(gate_in) -> [N, 1, C]`` for irreps gating.
        4. Project ``input_tensors`` to grid, chunk along channel into two
           halves, multiply elementwise (the "many-body" self tensor product),
           and project back to irreps.
        5. Multiply irreps output by the Sigmoid gate (broadcast across L).
        6. Add the SwiGLU output additively into L=0 (the "merge").

    Shapes:
        gating_scalars: [N, 3 * num_channels]
        input_tensors:  [N, (lmax+1)**2, 2 * num_channels]
        returns:        [N, (lmax+1)**2, num_channels]

    A private higher-resolution SO3_Grid is constructed because the elementwise
    grid product introduces frequency content up to 2*lmax. Default resolution
    follows V3's empirical table (lmax=2 -> 8, lmax=3 -> 10, lmax=4 -> 14,
    lmax=6 -> 20); a safe formula is 2*(2*lmax + 1).
    """

    _DEFAULT_RESOLUTION = {1: 4, 2: 8, 3: 10, 4: 14, 6: 20}

    def __init__(
        self,
        lmax: int,
        mmax: int,
        num_channels: int,
        grid_resolution: int | None = None,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.num_channels = num_channels
        if grid_resolution is None:
            grid_resolution = self._DEFAULT_RESOLUTION.get(lmax, 2 * (2 * lmax + 1))
        # mmax=lmax so the grid carries the full (lmax+1)**2 irrep set; the
        # bilinear product needs all m components even when the rest of the
        # network truncates at mmax<lmax.
        self.so3_grid = SO3_Grid(lmax=lmax, mmax=lmax, resolution=grid_resolution)
        self.scalar_act = _SwiGLU()
        self.gate_act = torch.nn.Sigmoid()

    def forward(
        self, gating_scalars: torch.Tensor, input_tensors: torch.Tensor
    ) -> torch.Tensor:
        gating_scalars = gating_scalars.view(
            gating_scalars.shape[0], 1, gating_scalars.shape[-1]
        )
        swiglu_in = gating_scalars.narrow(2, 0, 2 * self.num_channels)
        gate_in = gating_scalars.narrow(
            2, 2 * self.num_channels, self.num_channels
        )
        scalar_out = self.scalar_act(swiglu_in)
        gate_out = self.gate_act(gate_in)

        x_grid = self.so3_grid.to_grid(input_tensors, self.lmax, self.lmax)
        x_grid_1, x_grid_2 = torch.chunk(x_grid, 2, dim=-1)
        x_grid = x_grid_1 * x_grid_2
        vec_out = self.so3_grid.from_grid(x_grid, self.lmax, self.lmax)

        vec_out = vec_out * gate_out

        l0 = vec_out.narrow(1, 0, 1) + scalar_out
        rest = vec_out.narrow(1, 1, vec_out.shape[1] - 1)
        return torch.cat([l0, rest], dim=1)
