"""Exact differentiable state-vector QNN circuits implemented in PyTorch.

The legacy architecture mirrors the former PennyLane QNode and keeps its
parameter names and shapes, so existing checkpoints remain strictly loadable.
The optional QCNN architecture adds RY/RZ data re-uploading between U_SU4
ring blocks.
"""

import math
from itertools import combinations
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_complex(value: torch.Tensor) -> torch.Tensor:
    return torch.complex(value, torch.zeros_like(value))


def _phase(angle: torch.Tensor) -> torch.Tensor:
    return torch.complex(torch.cos(angle), torch.sin(angle))


def u3_matrix(weights: torch.Tensor) -> torch.Tensor:
    """Return PennyLane-compatible ``U3(theta, phi, delta)``."""
    theta, phi, delta = weights.unbind()
    cosine = _as_complex(torch.cos(theta / 2.0))
    sine = _as_complex(torch.sin(theta / 2.0))
    return torch.stack(
        (
            torch.stack((cosine, -sine * _phase(delta))),
            torch.stack((sine * _phase(phi), cosine * _phase(phi + delta))),
        )
    )


def ry_matrix(theta: torch.Tensor) -> torch.Tensor:
    """Return the one-qubit RY matrix."""
    cosine = _as_complex(torch.cos(theta / 2.0))
    sine = _as_complex(torch.sin(theta / 2.0))
    return torch.stack(
        (
            torch.stack((cosine, -sine)),
            torch.stack((sine, cosine)),
        )
    )


def rz_matrix(theta: torch.Tensor) -> torch.Tensor:
    """Return the one-qubit RZ matrix."""
    zero = _as_complex(torch.zeros_like(theta))
    return torch.stack(
        (
            torch.stack((_phase(-theta / 2.0), zero)),
            torch.stack((zero, _phase(theta / 2.0))),
        )
    )


def ry_matrix_batched(theta: torch.Tensor) -> torch.Tensor:
    """Return one RY matrix per sample for a one-dimensional angle tensor."""
    if theta.ndim != 1:
        raise ValueError("batched RY angles must have shape [batch]")
    cosine = _as_complex(torch.cos(theta / 2.0))
    sine = _as_complex(torch.sin(theta / 2.0))
    return torch.stack(
        (
            torch.stack((cosine, -sine), dim=-1),
            torch.stack((sine, cosine), dim=-1),
        ),
        dim=-2,
    )


def rz_matrix_batched(theta: torch.Tensor) -> torch.Tensor:
    """Return one RZ matrix per sample for a one-dimensional angle tensor."""
    if theta.ndim != 1:
        raise ValueError("batched RZ angles must have shape [batch]")
    zero = _as_complex(torch.zeros_like(theta))
    return torch.stack(
        (
            torch.stack((_phase(-theta / 2.0), zero), dim=-1),
            torch.stack((zero, _phase(theta / 2.0)), dim=-1),
        ),
        dim=-2,
    )


def rot_matrix(weights: torch.Tensor) -> torch.Tensor:
    """Return PennyLane's ``Rot(phi, theta, omega)`` matrix.

    The convention is ``RZ(omega) @ RY(theta) @ RZ(phi)``.
    """
    phi, theta, omega = weights.unbind()
    cosine = _as_complex(torch.cos(theta / 2.0))
    sine = _as_complex(torch.sin(theta / 2.0))
    return torch.stack(
        (
            torch.stack(
                (
                    _phase(-0.5 * (phi + omega)) * cosine,
                    -_phase(0.5 * (phi - omega)) * sine,
                )
            ),
            torch.stack(
                (
                    _phase(-0.5 * (phi - omega)) * sine,
                    _phase(0.5 * (phi + omega)) * cosine,
                )
            ),
        )
    )


def apply_1q(state: torch.Tensor, gate: torch.Tensor, wire: int) -> torch.Tensor:
    """Apply one shared 2x2 gate to a batched tensor-product state."""
    axis = wire + 1  # axis zero is the batch dimension
    moved = torch.movedim(state, axis, -1)
    moved_shape = moved.shape
    flat = moved.reshape(-1, 2)
    evolved = flat @ gate.transpose(-1, -2)
    return torch.movedim(evolved.reshape(moved_shape), -1, axis)


def apply_1q_batched(state: torch.Tensor, gate: torch.Tensor, wire: int) -> torch.Tensor:
    """Apply a different 2x2 gate to every tensor-product state in a batch."""
    if gate.ndim != 3 or tuple(gate.shape[-2:]) != (2, 2):
        raise ValueError("batched one-qubit gates must have shape [batch, 2, 2]")
    if gate.shape[0] != state.shape[0]:
        raise ValueError("gate and state batch dimensions must match")
    axis = wire + 1
    moved = torch.movedim(state, axis, -1)
    moved_shape = moved.shape
    flat = moved.reshape(state.shape[0], -1, 2)
    # This is the same contraction as ``einsum("bij,bsj->bsi")`` but bmm
    # avoids the generic einsum planner on every sample-dependent gate.  The
    # explicit transpose preserves the original multiplication order and is
    # bit-identical to the former path for float32/complex64 on CPU.
    evolved = torch.bmm(flat, gate.transpose(1, 2))
    return torch.movedim(evolved.reshape(moved_shape), -1, axis)


def apply_cnot(
    state: torch.Tensor,
    control: int,
    target: int,
    permutation: torch.Tensor,
) -> torch.Tensor:
    """Apply CNOT by permuting amplitudes instead of building a dense gate."""
    if control == target:
        raise ValueError("CNOT control and target must be different")
    axes = (control + 1, target + 1)
    moved = torch.movedim(state, axes, (-2, -1))
    moved_shape = moved.shape
    flat = moved.reshape(*moved_shape[:-2], 4)
    evolved = flat.index_select(-1, permutation).reshape(moved_shape)
    return torch.movedim(evolved, (-2, -1), axes)


def z_observables(n_qubits: int, max_order: int) -> Tuple[Tuple[int, ...], ...]:
    """List Z-correlation wire tuples in the former QNode's output order."""
    return tuple(
        wires
        for order in range(1, min(max_order, n_qubits) + 1)
        for wires in combinations(range(n_qubits), order)
    )


def _z_sign_table(
    n_qubits: int,
    observables: Sequence[Sequence[int]],
) -> torch.Tensor:
    basis = torch.arange(2**n_qubits, dtype=torch.int64)
    rows = []
    for wires in observables:
        signs = torch.ones_like(basis, dtype=torch.float32)
        for wire in wires:
            bit = (basis >> (n_qubits - 1 - wire)) & 1
            signs = signs * (1.0 - 2.0 * bit.to(torch.float32))
        rows.append(signs)
    return torch.stack(rows)


class TensorQuantumCircuit(nn.Module):
    """Pure-PyTorch replacement for the bridge's PennyLane ``TorchLayer``."""

    _VECTOR_WEIGHT_NAMES = (
        "weights_0",
        "weights_1",
        "weights_5",
        "weights_6",
        "weights_7",
        "weights_8",
        "weights_12",
        "weights_13",
    )
    _SCALAR_WEIGHT_NAMES = (
        "weights_2",
        "weights_3",
        "weights_4",
        "weights_9",
        "weights_10",
        "weights_11",
    )

    def __init__(
        self,
        n_qubits: int,
        qnn_layers: int,
        max_z_order: int,
        unitary_fusion: bool = False,
        edge_specific: bool = False,
        architecture: str = "legacy",
        entangling: bool = True,
    ):
        super().__init__()
        if n_qubits < 2:
            raise ValueError("n_qubits must be >= 2")
        if qnn_layers < 1:
            raise ValueError("qnn_layers must be >= 1")
        if max_z_order < 1 or max_z_order > n_qubits:
            raise ValueError(f"max_z_order must be in [1, {n_qubits}]")
        reupload_architectures = (
            "qcnn_reupload",
            "qcnn_angle_reupload",
            "qcnn_shared_reupload",
        )
        if architecture not in ("legacy", *reupload_architectures):
            raise ValueError(
                "architecture must be 'legacy', 'qcnn_reupload', "
                "'qcnn_angle_reupload', or 'qcnn_shared_reupload'"
            )
        if not entangling and architecture not in reupload_architectures:
            raise ValueError("removing SU(4) blocks requires a re-upload architecture")
        if architecture in reupload_architectures and unitary_fusion:
            raise ValueError(
                "unitary_fusion is unavailable for reupload architectures because their "
                "RY/RZ gates are sample dependent"
            )

        self.n_qubits = n_qubits
        self.qnn_layers = qnn_layers
        self.max_z_order = max_z_order
        self.amplitude_dim = 2**n_qubits
        self.unitary_fusion = unitary_fusion
        self.edge_specific = edge_specific
        self.architecture = architecture
        self.entangling = entangling
        self.ring_wires = tuple((wire, (wire + 1) % n_qubits) for wire in range(n_qubits))
        self.observables = z_observables(n_qubits, max_z_order)

        if architecture in reupload_architectures:
            # Each re-upload layer has its own U_SU4 block.  Shared mode ties
            # the block across ring edges but never across re-upload depth.
            shape = (qnn_layers, n_qubits, 15) if edge_specific else (qnn_layers, 15)
            self.qcnn_weights = nn.Parameter(torch.empty(shape))
            nn.init.uniform_(self.qcnn_weights, 0.0, 2.0 * math.pi)
            if not entangling:
                # Keep the draw and checkpoint key so paired seeds initialize
                # the encoder/readout identically. These 15*l angles are
                # frozen and never applied by the no-SU(4) ablation.
                self.qcnn_weights.requires_grad_(False)
        elif edge_specific:
            # Two U_SU4 sweeps, one independent 15-angle block per ring edge.
            # This removes the unintended weight sharing that constrained the
            # old ring model to the same local operation on every AP pair.
            self.qcnn_weights = nn.Parameter(torch.empty(2, n_qubits, 15))
            nn.init.uniform_(self.qcnn_weights, 0.0, 2.0 * math.pi)
        else:
            # Legacy registration order, names, shapes and initialization match
            # qml.qnn.TorchLayer, including zero-dimensional scalar parameters.
            for index in range(14):
                name = f"weights_{index}"
                if name in self._VECTOR_WEIGHT_NAMES:
                    shape = (3,)
                elif name in self._SCALAR_WEIGHT_NAMES:
                    shape = ()
                else:  # Guard the checkpoint layout if the parameter list changes.
                    raise RuntimeError(f"missing shape declaration for {name}")
                parameter = nn.Parameter(torch.empty(shape))
                nn.init.uniform_(parameter, 0.0, 2.0 * math.pi)
                self.register_parameter(name, parameter)
        if architecture == "legacy":
            self.entangling_weights = nn.Parameter(torch.empty(qnn_layers, n_qubits, 3))
            nn.init.uniform_(self.entangling_weights, 0.0, 2.0 * math.pi)

        # Non-persistent buffers follow the model across devices without adding
        # keys that would break strict loading of old PennyLane checkpoints.
        self.register_buffer(
            "_cnot_permutation",
            torch.tensor((0, 1, 3, 2), dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "_measurement_signs",
            _z_sign_table(n_qubits, self.observables),
            persistent=False,
        )
        # Re-upload angles depend on each sample, so the complete circuit cannot
        # be fused. Between optimizer updates, however, the U_SU4 ring following
        # each re-upload block is constant. Keep those per-layer unitaries
        # outside checkpoints and rebuild them after circuit weights change.
        self.register_buffer(
            "_frozen_layer_unitaries",
            torch.empty(0, dtype=torch.complex64),
            persistent=False,
        )

    @property
    def n_measurements(self) -> int:
        return len(self.observables)

    def _apply_u_su4(
        self,
        state: torch.Tensor,
        weights: torch.Tensor,
        wires: Tuple[int, int],
    ) -> torch.Tensor:
        if tuple(weights.shape) != (15,):
            raise ValueError(f"U_SU4 expects 15 angles, got shape {tuple(weights.shape)}")
        state = apply_1q(state, u3_matrix(weights[0:3]), wires[0])
        state = apply_1q(state, u3_matrix(weights[3:6]), wires[1])
        state = apply_cnot(state, wires[0], wires[1], self._cnot_permutation)
        state = apply_1q(state, ry_matrix(weights[6]), wires[0])
        state = apply_1q(state, rz_matrix(weights[7]), wires[1])
        state = apply_cnot(state, wires[1], wires[0], self._cnot_permutation)
        state = apply_1q(state, ry_matrix(weights[8]), wires[0])
        state = apply_cnot(state, wires[0], wires[1], self._cnot_permutation)
        state = apply_1q(state, u3_matrix(weights[9:12]), wires[0])
        state = apply_1q(state, u3_matrix(weights[12:15]), wires[1])
        return state

    def _legacy_sweep_weights(self, sweep: int) -> torch.Tensor:
        offset = 7 * sweep
        return torch.cat(
            (
                getattr(self, f"weights_{offset}"),
                getattr(self, f"weights_{offset + 1}"),
                getattr(self, f"weights_{offset + 2}").reshape(1),
                getattr(self, f"weights_{offset + 3}").reshape(1),
                getattr(self, f"weights_{offset + 4}").reshape(1),
                getattr(self, f"weights_{offset + 5}"),
                getattr(self, f"weights_{offset + 6}"),
            )
        )

    @torch.no_grad()
    def prepare_frozen_inference(self) -> None:
        """Fuse the current U_SU4 ring in every re-upload layer.

        The resulting matrices are derived data rather than model parameters,
        so they deliberately do not participate in ``state_dict``.  Training
        mode always bypasses them; callers must rebuild them after weights are
        updated and before returning to evaluation mode.
        """
        if not self.entangling or self.architecture not in (
            "qcnn_reupload",
            "qcnn_angle_reupload",
            "qcnn_shared_reupload",
        ):
            return
        real_dtype = self.qcnn_weights.dtype
        device = self.qcnn_weights.device
        basis = torch.eye(self.amplitude_dim, dtype=real_dtype, device=device)
        basis = torch.complex(basis, torch.zeros_like(basis))
        layer_unitaries = []
        for layer in range(self.qnn_layers):
            state = basis.reshape(self.amplitude_dim, *([2] * self.n_qubits))
            shared_weights = None if self.edge_specific else self.qcnn_weights[layer]
            for edge, wires in enumerate(self.ring_wires):
                weights = self.qcnn_weights[layer, edge] if self.edge_specific else shared_weights
                state = self._apply_u_su4(state, weights, wires)
            layer_unitaries.append(state.reshape(self.amplitude_dim, self.amplitude_dim))
        self._frozen_layer_unitaries = torch.stack(layer_unitaries).detach()

    def clear_frozen_inference(self) -> None:
        """Discard derived fused matrices before a model is trained again."""
        self._frozen_layer_unitaries = torch.empty(
            0,
            dtype=self._frozen_layer_unitaries.dtype,
            device=self._frozen_layer_unitaries.device,
        )

    def _apply_legacy_circuit(self, flat_state: torch.Tensor) -> torch.Tensor:
        state = flat_state.reshape(flat_state.shape[0], *([2] * self.n_qubits))
        for sweep in range(2):
            shared_weights = None if self.edge_specific else self._legacy_sweep_weights(sweep)
            for edge, wires in enumerate(self.ring_wires):
                weights = self.qcnn_weights[sweep, edge] if self.edge_specific else shared_weights
                state = self._apply_u_su4(state, weights, wires)

        for layer in range(self.qnn_layers):
            for wire in range(self.n_qubits):
                state = apply_1q(
                    state,
                    rot_matrix(self.entangling_weights[layer, wire]),
                    wire,
                )
            circuit_range = (layer % (self.n_qubits - 1)) + 1
            for wire in range(self.n_qubits):
                state = apply_cnot(
                    state,
                    wire,
                    (wire + circuit_range) % self.n_qubits,
                    self._cnot_permutation,
                )
        return state.reshape(flat_state.shape[0], self.amplitude_dim)

    def _apply_qcnn_reupload_circuit(
        self,
        flat_state: torch.Tensor,
        reupload_angles: torch.Tensor,
    ) -> torch.Tensor:
        expected = (flat_state.shape[0], self.qnn_layers, self.n_qubits, 2)
        if tuple(reupload_angles.shape) != expected:
            raise ValueError(f"expected reupload angle shape {expected}, got {tuple(reupload_angles.shape)}")
        state = flat_state.reshape(flat_state.shape[0], *([2] * self.n_qubits))
        for layer in range(self.qnn_layers):
            # The fused branch below leaves a flat [batch, 2**n] state.
            state = state.reshape(flat_state.shape[0], *([2] * self.n_qubits))
            # Data re-upload: the raw input controls both a population-changing
            # RY rotation and a phase-changing RZ rotation on every qubit.
            for wire in range(self.n_qubits):
                # RY is followed by RZ, hence their equivalent single gate is
                # RZ @ RY.  Combining them halves the sample-dependent state
                # permutations and batched contractions.
                data_gate = torch.bmm(
                    rz_matrix_batched(reupload_angles[:, layer, wire, 1]),
                    ry_matrix_batched(reupload_angles[:, layer, wire, 0]),
                )
                state = apply_1q_batched(state, data_gate, wire)

            if not self.entangling:
                # This removes trainable SU(4) blocks, not entanglement that
                # may already exist in the arbitrary amplitude-encoded state.
                continue

            if self._frozen_layer_unitaries.numel() > 0 and not self.training:
                state = (
                    state.reshape(flat_state.shape[0], self.amplitude_dim)
                    @ self._frozen_layer_unitaries[layer]
                )
                continue

            shared_weights = None if self.edge_specific else self.qcnn_weights[layer]
            for edge, wires in enumerate(self.ring_wires):
                weights = self.qcnn_weights[layer, edge] if self.edge_specific else shared_weights
                state = self._apply_u_su4(state, weights, wires)
        return state.reshape(flat_state.shape[0], self.amplitude_dim)

    def _evolve(
        self,
        amplitudes: torch.Tensor,
        reupload_angles: torch.Tensor = None,
    ) -> torch.Tensor:
        normalized = F.normalize(amplitudes, p=2.0, dim=-1, eps=1.0e-12)
        complex_state = torch.complex(normalized, torch.zeros_like(normalized))
        if self.architecture in (
            "qcnn_reupload",
            "qcnn_angle_reupload",
            "qcnn_shared_reupload",
        ):
            if reupload_angles is None:
                raise ValueError("reupload architectures require RY/RZ angles")
            return self._apply_qcnn_reupload_circuit(complex_state, reupload_angles)
        if reupload_angles is not None:
            raise ValueError("legacy architecture does not accept reupload angles")
        if not self.unitary_fusion:
            return self._apply_legacy_circuit(complex_state)

        basis = torch.eye(
            self.amplitude_dim,
            dtype=amplitudes.dtype,
            device=amplitudes.device,
        )
        basis = torch.complex(basis, torch.zeros_like(basis))
        unitary_rows = self._apply_legacy_circuit(basis)
        return complex_state @ unitary_rows

    def forward(
        self,
        amplitudes: torch.Tensor,
        reupload_angles: torch.Tensor = None,
    ) -> torch.Tensor:
        if amplitudes.shape[-1] != self.amplitude_dim:
            raise ValueError(
                f"expected amplitude dimension {self.amplitude_dim}, got {amplitudes.shape[-1]}"
            )
        if not amplitudes.is_floating_point():
            raise TypeError("amplitudes must be a real floating-point tensor")

        single = amplitudes.ndim == 1
        if single:
            amplitudes = amplitudes.unsqueeze(0)
            if reupload_angles is not None and reupload_angles.ndim == 3:
                reupload_angles = reupload_angles.unsqueeze(0)
        elif amplitudes.ndim != 2:
            raise ValueError("amplitudes must have shape [2**n] or [batch, 2**n]")

        evolved = self._evolve(amplitudes, reupload_angles)
        probabilities = evolved.abs().square()
        measurement_signs = self._measurement_signs.to(probabilities.dtype)
        measurements = probabilities @ measurement_signs.transpose(0, 1)
        return measurements.squeeze(0) if single else measurements
