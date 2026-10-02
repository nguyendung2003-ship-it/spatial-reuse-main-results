#!/usr/bin/env python3
"""TCP bridge for NeuralTS control of ns-3.

Protocol:
  Client -> Server:
      OBSERVE <dim> <reward> <x0> ... <x_{dim-1}>
      ASK_CHOICE <dim> <n_candidates>
      CURRENT <x0> ... <x_{dim-1}>    # optional
      BEST <x0> ... <x_{dim-1}>       # optional
      CANDIDATE <x0> ... <x_{dim-1}>  # repeated n_candidates times
      END
      ASK_CONFIG <dim>
      CURRENT <x0> ... <x_{dim-1}>  # optional
      BEST <x0> ... <x_{dim-1}>     # optional
      END
  Server -> Client:
      CHOICE <candidate_index>
      CONFIG <dim> <x0> ... <x_{dim-1}>

ASK_CHOICE is the production path used by ns-3: C++ generates valid candidates
with the same sampler family as the paper, and this server only scores/selects
one of those candidates. ASK_CONFIG remains as a legacy direct-generator path.

Each TCP connection owns an independent Neural-Linear state by default. This
keeps repeated ns-3 simulations statistically separate even when the ns-3
driver forks multiple child processes.
"""

import argparse
import csv
import math
import os
import socket
import socketserver
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.linalg import solve_triangular as _scipy_solve_triangular
except ImportError:
    _scipy_solve_triangular = None

try:
    from .tensor_quantum import TensorQuantumCircuit
except ImportError:  # Support direct execution: python neural_bandit_bridge/nb_server.py
    from tensor_quantum import TensorQuantumCircuit


_TRAIN_LOG_LOCK = threading.Lock()
_TIMING_LOG_LOCK = threading.Lock()
_FEATURE_DIAGNOSTICS_LOG_LOCK = threading.Lock()


def _solve_triangular(matrix: np.ndarray, rhs: np.ndarray, lower: bool) -> np.ndarray:
    """Use the triangular structure of the posterior Cholesky factor."""
    if _scipy_solve_triangular is not None:
        return _scipy_solve_triangular(matrix, rhs, lower=lower, check_finite=False)
    return np.linalg.solve(matrix, rhs)


class SurrogateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, feature_dim: int):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(feature_dim, 1, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)


class NormalizedSurrogateMLP(SurrogateMLP):
    """The original MLP with the QNN's tanh and fixed-norm feature readout.

    Only the parameter-free final activation changes.  The hidden layers,
    linear weights, initialization, and temporary reward head match the
    original SurrogateMLP, permitting a controlled readout ablation.
    """

    def __init__(self, input_dim: int, hidden: int, feature_dim: int):
        super().__init__(input_dim, hidden, feature_dim)
        self.feature_net[-1] = nn.Identity()

    def features(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.tanh(self.feature_net(x))
        return F.normalize(features, p=2.0, dim=-1, eps=1.0e-6) * math.sqrt(features.shape[-1])


class ReluNormalizedSurrogateMLP(SurrogateMLP):
    """Keep the original ReLU readout and normalize its feature length."""

    def features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(x)
        return F.normalize(features, p=2.0, dim=-1, eps=1.0e-6) * math.sqrt(features.shape[-1])


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class ParameterMatchedMLP(nn.Module):
    """Three-layer MLP with exactly the trainable parameter count of a QNN.

    Integer hidden widths rarely hit the QNN count exactly.  The remaining
    weights form a sparse, trainable skip connection from the first hidden
    layer to the feature layer; every counted weight affects the prediction.
    """

    def __init__(self, input_dim: int, feature_dim: int, target_params: int):
        super().__init__()
        if target_params <= 0:
            raise ValueError("target_params must be positive")
        def base_count(width: int) -> int:
            return (input_dim + 1) * width + (width + 1) * width + (width + 1) * feature_dim + feature_dim

        width = 1
        while base_count(width + 1) <= target_params:
            width += 1
        if base_count(width) > target_params:
            raise ValueError("target_params is too small for a matched MLP")
        self.hidden_width = width
        self.target_params = target_params
        self.skip_parameter_count = target_params - base_count(width)
        self.first = nn.Linear(input_dim, width)
        self.second = nn.Linear(width, width)
        self.third = nn.Linear(width, feature_dim)
        self.head = nn.Linear(feature_dim, 1, bias=False)
        self.skip_weights = nn.Parameter(torch.zeros(self.skip_parameter_count))
        self.register_buffer(
            "skip_source_indices",
            torch.arange(self.skip_parameter_count, dtype=torch.int64) % width,
            persistent=False,
        )
        self.register_buffer(
            "skip_target_indices",
            torch.arange(self.skip_parameter_count, dtype=torch.int64) % feature_dim,
            persistent=False,
        )
        for layer in (self.first, self.second, self.third, self.head):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        assert trainable_parameter_count(self) == target_params

    def features(self, x: torch.Tensor) -> torch.Tensor:
        first = F.relu(self.first(x))
        second = F.relu(self.second(first))
        values = self.third(second)
        if self.skip_parameter_count:
            skip = first.index_select(-1, self.skip_source_indices) * self.skip_weights
            additions = torch.zeros_like(values)
            indices = self.skip_target_indices.expand(*skip.shape[:-1], -1)
            additions.scatter_add_(-1, indices, skip)
            values = values + additions / math.sqrt(self.skip_parameter_count)
        return F.relu(values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)


class RandomFourierFeatureMap(nn.Module):
    """Fixed RBF-kernel random Fourier map for coherent approximate GP-TS."""

    def __init__(self, input_dim: int, output_dim: int, length_scale: float, seed: int):
        super().__init__()
        if input_dim < 1 or output_dim < 1 or length_scale <= 0:
            raise ValueError("RFF dimensions and length scale must be positive")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        # Dividing by sqrt(input_dim) defines length_scale per normalized AP
        # coordinate and keeps the prior comparable across AP counts.
        frequency = torch.randn(input_dim, output_dim, generator=generator)
        frequency /= length_scale * math.sqrt(input_dim)
        phase = 2.0 * math.pi * torch.rand(output_dim, generator=generator)
        self.register_buffer("frequency", frequency)
        self.register_buffer("phase", phase)
        self.output_dim = output_dim

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return math.sqrt(2.0 / self.output_dim) * torch.cos(x @ self.frequency + self.phase)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class QNNSurrogate(nn.Module):
    """Pure-PyTorch QNN with Z-correlation measurements mapped to feature_dim."""

    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        qnn_layers: int,
        max_z_order: int,
        n_qubits: int,
        unitary_fusion: bool = False,
        edge_specific: bool = False,
        input_tanh: bool = False,
        architecture: str = "legacy",
        normalize_readout: bool = True,
        entangling: bool = True,
        freeze_quantum: bool = False,
    ):
        super().__init__()
        if (not entangling or freeze_quantum) and architecture != "qcnn_shared_reupload":
            raise ValueError("SU(4) ablations require qcnn_shared_reupload")
        self.n_qubits = n_qubits
        self.qnn_layers = qnn_layers
        self.input_tanh = (
            True
            if architecture == "qcnn_shared_reupload"
            else input_tanh and architecture != "qcnn_angle_reupload"
        )
        self.architecture = architecture
        self.normalize_readout = normalize_readout
        if qnn_layers < 1:
            raise ValueError("qnn_layers must be >= 1")
        if self.n_qubits < 2:
            raise ValueError("qnn_n_qubits must be >= 2")
        if max_z_order < 1 or max_z_order > self.n_qubits:
            raise ValueError(f"qnn_max_z_order must be in [1, {self.n_qubits}]")
        if architecture == "qcnn_angle_reupload" and input_dim % 2 != 0:
            raise ValueError("qcnn_angle_reupload requires sensitivity/power input pairs")
        self.amplitude_dim = 2 ** self.n_qubits
        self.input_dim = input_dim
        if architecture != "qcnn_angle_reupload":
            self.input_layer = nn.Linear(input_dim, self.amplitude_dim)
        if architecture == "qcnn_reupload":
            # This path sees x before amplitude normalization.  Its learned
            # RY/RZ angles therefore distinguish inputs that lie on the same
            # ray but have different absolute magnitudes.
            self.angle_layer = nn.Linear(input_dim, qnn_layers * n_qubits * 2)
        self.qlayer = TensorQuantumCircuit(
            n_qubits=self.n_qubits,
            qnn_layers=qnn_layers,
            max_z_order=max_z_order,
            unitary_fusion=unitary_fusion,
            edge_specific=edge_specific,
            architecture=architecture,
            entangling=entangling,
        )
        if freeze_quantum:
            # Keep applying the SU(4) blocks, but update only the classical
            # encoder, correlator readout, and temporary reward head.
            self.qlayer.qcnn_weights.requires_grad_(False)
        self.measure_to_feature = nn.Linear(self.qlayer.n_measurements, feature_dim)
        self.head = nn.Linear(feature_dim, 1, bias=False)
        angle_count = qnn_layers * n_qubits * 2
        self.register_buffer(
            "_shared_angle_indices",
            torch.arange(angle_count, dtype=torch.int64) % self.amplitude_dim,
            persistent=False,
        )
        pair_count = max(1, input_dim // 2)
        self.register_buffer(
            "_direct_pair_indices",
            torch.arange(qnn_layers * n_qubits, dtype=torch.int64) % pair_count,
            persistent=False,
        )
        self._init_classical_weights()

    def _init_classical_weights(self) -> None:
        modules = [self.head]
        if hasattr(self, "input_layer"):
            modules.append(self.input_layer)
        if hasattr(self, "angle_layer"):
            modules.append(self.angle_layer)
        for module in modules:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        linear = self.measure_to_feature
        nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)

    def optimizer_param_groups(
        self,
        lr: float,
        weight_decay: float,
        quantum_lr: float,
        quantum_weight_decay: float,
    ):
        classical_params = []
        if hasattr(self, "input_layer"):
            classical_params += list(self.input_layer.parameters())
        if hasattr(self, "angle_layer"):
            classical_params += list(self.angle_layer.parameters())
        classical_params += list(self.measure_to_feature.parameters())
        classical_params += list(self.head.parameters())
        quantum_params = [parameter for parameter in self.qlayer.parameters() if parameter.requires_grad]
        groups = [
            {
                "params": classical_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "classical",
            },
        ]
        if quantum_params:
            groups.append({
                "params": quantum_params,
                "lr": quantum_lr,
                "weight_decay": quantum_weight_decay,
                "name": "quantum",
            })
        return groups

    def _direct_reupload_angles(self, x: torch.Tensor) -> torch.Tensor:
        # Direct, parameter-free encoding of every WLAN control pair.  Each
        # circuit slot consumes one (sensitivity, txPower) pair; extra slots
        # cycle through APs and thus perform literal data re-uploading.
        pairs = x.reshape(*x.shape[:-1], self.input_dim // 2, 2)
        angle_shape = (*x.shape[:-1], self.qnn_layers, self.n_qubits, 2)
        return math.pi * pairs.index_select(-2, self._direct_pair_indices).reshape(angle_shape)

    def _shared_reupload_angles(self, encoded: torch.Tensor) -> torch.Tensor:
        # The exact same Linear+tanh latent initializes the amplitude state and
        # supplies every RY/RZ angle.  Fixed cyclic indexing adds no second
        # encoder or trainable branch-specific parameters.
        angle_shape = (*encoded.shape[:-1], self.qnn_layers, self.n_qubits, 2)
        return math.pi * encoded.index_select(-1, self._shared_angle_indices).reshape(angle_shape)

    def quantum_measurements(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected input dimension {self.input_dim}, got {x.shape[-1]}")
        reupload_angles = None
        if self.architecture == "qcnn_angle_reupload":
            reupload_angles = self._direct_reupload_angles(x)
            q_input = torch.zeros(
                *x.shape[:-1],
                self.amplitude_dim,
                dtype=x.dtype,
                device=x.device,
            )
            q_input[..., 0] = 1.0
        else:
            q_input = self.input_layer(x)
            if self.input_tanh:
                q_input = torch.tanh(q_input)
        if self.architecture == "qcnn_shared_reupload":
            reupload_angles = self._shared_reupload_angles(q_input)
        elif self.architecture == "qcnn_reupload":
            angle_shape = (*x.shape[:-1], self.qnn_layers, self.n_qubits, 2)
            reupload_angles = math.pi * torch.tanh(self.angle_layer(x)).reshape(angle_shape)
        measurements = self.qlayer(q_input, reupload_angles)
        if measurements.ndim == 1:
            measurements = measurements.unsqueeze(0)
        return measurements

    def features(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.measure_to_feature(self.quantum_measurements(x))
        if not self.normalize_readout:
            return projected
        # A linear measurement projection followed immediately by the linear
        # training head is not an identifiable feature representation.  The
        # head can fit rewards while the representation collapses, after which
        # the separately rebuilt Bayesian-linear head cannot rank candidates.
        # Bound the projection and fix its per-sample scale so the learned
        # geometry, rather than an arbitrary feature magnitude, drives BLR.
        features = torch.tanh(projected)
        return F.normalize(features, p=2.0, dim=-1, eps=1.0e-6) * (features.shape[-1] ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)


class _FeatureOnlyModule(nn.Module):
    """Traceable view of a frozen surrogate's feature extractor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.features(x)


@dataclass
class BanditConfig:
    hidden: int = 64
    feature_dim: int = 32
    surrogate: str = "mlp"
    policy: str = "neural_ts"
    # Zero means match the QNN with the QNN flags below for this input dim.
    matched_params: int = 0
    ucb_beta: float = 1.0
    gp_features: int = 128
    gp_length_scale: float = 0.5
    gp_noise: float = 0.1
    qnn_layers: int = 20
    qnn_n_qubits: int = 4
    qnn_max_z_order: int = 4
    qnn_unitary_fusion: bool = False
    qnn_edge_specific: bool = False
    qnn_input_tanh: bool = False
    qnn_architecture: str = "legacy"
    retrain_interval: int = 20
    early_retrain_interval: int = 10
    # -1 keeps periodic online retraining, 0 uses a fixed random feature map,
    # and N>0 freezes the surrogate after N online retraining events.
    max_retrains: int = -1
    # -1 disables the environment-step cutoff. Otherwise the surrogate is
    # frozen before processing the first observation after this shared budget.
    freeze_after_step: int = -1
    cold_start: int = 20
    train_epochs: int = 80
    # 0 preserves train_epochs for every update. A positive value uses fewer
    # epochs after the initial fit while retaining continuous online adaptation.
    update_epochs: int = 0
    batch_size: int = 256
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    quantum_lr: float = 3.0e-3
    quantum_weight_decay: float = 0.0
    train_log: str = ""
    timing_log: str = ""
    feature_diagnostics_log: str = ""
    timing_method: str = ""
    timing_topology: str = ""
    checkpoint_dir: str = ""
    checkpoint_every: int = 0
    save_final_checkpoint: bool = False
    load_checkpoint: str = ""
    restore_checkpoint_state: bool = False
    freeze_loaded_model: bool = True
    qnn_frozen_fusion: bool = True
    qnn_frozen_torchscript: bool = True
    feature_cache_size: int = 65536
    prior_lambda: float = 1.0
    prior_a: float = 0.5
    prior_b: float = 0.025
    virtual_budget: int = 2048
    cem_iters: int = 3
    elite_frac: float = 0.05
    top_centers: int = 8
    early_step: int = 300
    late_step: int = 800
    early_changed_aps: int = 4
    mid_changed_aps: int = 8
    late_changed_aps: int = 16
    early_delta_db: float = 4.0
    mid_delta_db: float = 6.0
    late_delta_db: float = 8.0
    acquisition_top_k: int = 1
    acquisition_temperature: float = 0.0
    acquisition_epsilon: float = 0.0
    posterior_scale: float = 1.0
    local_resample_attempts: int = 64
    seed: int = 0
    device: str = "cpu"


def valid_pair(sensitivity: float, tx_power: float) -> bool:
    return (
        0.0 <= sensitivity <= 1.0
        and 0.0 <= tx_power <= 1.0
        and sensitivity + tx_power < 0.95 - 1.0e-6
    )


def sample_feasible_pair(rng: np.random.Generator) -> Tuple[float, float]:
    headroom = 0.95 * float(rng.random())
    split = float(rng.beta(2.0, 2.0))
    return headroom * split, headroom * (1.0 - split)


def project_pair(sensitivity: float, tx_power: float) -> Tuple[float, float]:
    s = float(np.clip(sensitivity, 0.0, 1.0))
    p = float(np.clip(tx_power, 0.0, 1.0))
    if s + p >= 0.95 - 1.0e-6:
        total = max(s + p, 1.0e-12)
        scale = 0.90 / total
        s *= scale
        p *= scale
    return s, p


def project_config(x: np.ndarray) -> np.ndarray:
    projected = np.asarray(x, dtype=np.float32).copy()
    for i in range(0, projected.shape[0], 2):
        projected[i], projected[i + 1] = project_pair(projected[i], projected[i + 1])
    return projected


def config_valid(x: np.ndarray) -> bool:
    pairs = np.asarray(x, dtype=np.float32).reshape(-1, 2)
    return bool(np.all((pairs >= 0.0) & (pairs <= 1.0)) and np.all(np.sum(pairs, axis=1) < 0.95 - 1.0e-6))


def boundary_rate(xs: np.ndarray, eps: float = 1.0e-6) -> float:
    array = np.asarray(xs, dtype=np.float32)
    if array.size == 0:
        return 0.0
    pairs = array.reshape(-1, 2)
    near_constraint = np.sum(pairs, axis=1) >= 0.95 - eps
    near_corner = (
        ((pairs[:, 0] <= eps) & (pairs[:, 1] >= 0.95 - eps))
        | ((pairs[:, 1] <= eps) & (pairs[:, 0] >= 0.95 - eps))
    )
    return float(np.mean(near_constraint | near_corner))


class NeuralLinearBandit:
    def __init__(self, cfg: BanditConfig):
        if cfg.policy not in ("neural_ts", "neural_ucb", "gp_ts"):
            raise ValueError(f"unknown policy '{cfg.policy}'")
        if cfg.policy == "gp_ts" and (cfg.gp_features < 1 or cfg.gp_noise <= 0):
            raise ValueError("GP feature count and observation noise must be positive")
        self.cfg = cfg
        self.posterior_dim = cfg.gp_features if cfg.policy == "gp_ts" else cfg.feature_dim
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.input_dim: Optional[int] = None
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.frozen_feature_extractor = None
        self.feature_cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self.feature_cache_hits: int = 0
        self.feature_cache_misses: int = 0

        self.history_x: List[np.ndarray] = []
        self.history_y: List[float] = []
        self.precision: Optional[np.ndarray] = None
        self.precision_cholesky: Optional[np.ndarray] = None
        self.phi_y: Optional[np.ndarray] = None
        self.yty: float = 0.0
        self.n_obs: int = 0
        self.last_stats = {}
        self.train_calls: int = 0
        self.surrogate_frozen: bool = False
        self.loaded_checkpoint: str = ""
        self._loaded_checkpoint_payload: Optional[dict] = None
        self.timing_records: List[dict] = []
        self.timing_counts = {}
        self.environment_step: int = 0
        self.feature_norm_sum: float = 0.0
        self.last_feature_norm: float = float("nan")

    def log_feature_diagnostics(self) -> None:
        """Record the current design's spectrum and observed feature scale.

        The first cold-start observations do not enter the Bayesian head until
        its first fit. Those rows retain the round index but mark posterior
        diagnostics as unavailable rather than treating the prior as evidence.
        """
        path = self.cfg.feature_diagnostics_log
        if not path:
            return
        ready = self.precision is not None and self.n_obs > 0
        if ready:
            eigenvalues = np.linalg.eigvalsh(self.precision)
            lambda_min = float(eigenvalues[0])
            lambda_max = float(eigenvalues[-1])
            condition_number = (
                lambda_max / lambda_min if lambda_min > 0.0 else float("nan")
            )
            mean_feature_norm = self.feature_norm_sum / self.n_obs
            last_feature_norm = self.last_feature_norm
        else:
            lambda_min = lambda_max = condition_number = float("nan")
            mean_feature_norm = last_feature_norm = float("nan")
        fields = [
            "seed", "topology", "method", "surrogate", "policy", "input_dim", "feature_dim",
            "decision_round", "environment_step", "n_obs", "train_calls",
            "posterior_ready", "lambda_min", "lambda_max",
            "lambda_condition_number", "mean_feature_l2_norm",
            "latest_feature_l2_norm",
        ]
        row = {
            "seed": self.cfg.seed,
            "topology": self.cfg.timing_topology,
            "method": self.cfg.timing_method,
            "surrogate": self.cfg.surrogate,
            "policy": self.cfg.policy,
            "input_dim": self.input_dim,
            "feature_dim": self.posterior_dim,
            "decision_round": len(self.history_y),
            "environment_step": self.environment_step,
            "n_obs": self.n_obs,
            "train_calls": self.train_calls,
            "posterior_ready": int(ready),
            "lambda_min": lambda_min,
            "lambda_max": lambda_max,
            "lambda_condition_number": condition_number,
            "mean_feature_l2_norm": mean_feature_norm,
            "latest_feature_l2_norm": last_feature_norm,
        }
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with _FEATURE_DIAGNOSTICS_LOG_LOCK:
            exists = os.path.exists(path) and os.path.getsize(path) > 0
            with open(path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if not exists:
                    writer.writeheader()
                writer.writerow(row)

    def timing_start(self) -> int:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter_ns()

    def record_timing(
        self,
        operation: str,
        started_ns: int,
        n_candidates: int = 0,
    ) -> None:
        if not self.cfg.timing_log:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ns = time.perf_counter_ns() - started_ns
        call_index = int(self.timing_counts.get(operation, 0))
        self.timing_counts[operation] = call_index + 1
        self.timing_records.append({
            "source": "python",
            "topology": self.cfg.timing_topology,
            "method": self.cfg.timing_method,
            "optimizer": "NB",
            "sampler": "",
            "hcm_profile": "",
            "seed": self.cfg.seed,
            "surrogate": self.cfg.surrogate,
            "policy": self.cfg.policy,
            "backend": (
                "rff-rbf-gp"
                if self.cfg.policy == "gp_ts"
                else (
                "torch-statevector"
                if self.cfg.surrogate in ("qnn", "qnn_no_norm", "qnn_no_ent", "qnn_frozen_quantum")
                else "torch-mlp"
                )
            ),
            "architecture": (
                self.cfg.qnn_architecture
                if self.cfg.surrogate in ("qnn", "qnn_no_norm", "qnn_no_ent", "qnn_frozen_quantum")
                else ("rff-rbf" if self.cfg.policy == "gp_ts" else self.cfg.surrogate)
            ),
            "operation": operation,
            "call_index": call_index,
            "n_obs": len(self.history_y),
            "env_step": self.environment_step,
            "n_candidates": int(n_candidates),
            "elapsed_ns": int(elapsed_ns),
        })

    def flush_timing(self) -> None:
        if not self.cfg.timing_log or not self.timing_records:
            return
        path = self.cfg.timing_log
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fields = [
            "source",
            "topology",
            "method",
            "optimizer",
            "sampler",
            "hcm_profile",
            "seed",
            "surrogate",
            "policy",
            "backend",
            "architecture",
            "operation",
            "call_index",
            "n_obs",
            "env_step",
            "n_candidates",
            "elapsed_ns",
        ]
        with _TIMING_LOG_LOCK:
            exists = os.path.exists(path) and os.path.getsize(path) > 0
            with open(path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if not exists:
                    writer.writeheader()
                writer.writerows(self.timing_records)
        self.timing_records.clear()

    def _ensure_model(self, input_dim: int) -> None:
        if self.model is not None:
            if input_dim != self.input_dim:
                raise ValueError(f"input dim changed from {self.input_dim} to {input_dim}")
            return

        self.input_dim = input_dim
        if self.cfg.policy == "gp_ts":
            self.model = RandomFourierFeatureMap(
                input_dim,
                self.cfg.gp_features,
                self.cfg.gp_length_scale,
                self.cfg.seed,
            ).to(self.device)
        elif self.cfg.surrogate == "mlp":
            self.model = SurrogateMLP(input_dim, self.cfg.hidden, self.cfg.feature_dim).to(self.device)
        elif self.cfg.surrogate == "mlp_relu_norm":
            self.model = ReluNormalizedSurrogateMLP(input_dim, self.cfg.hidden, self.cfg.feature_dim).to(self.device)
        elif self.cfg.surrogate in ("mlp_norm", "mlp_norm_small"):
            hidden = 22 if self.cfg.surrogate == "mlp_norm_small" else self.cfg.hidden
            self.model = NormalizedSurrogateMLP(input_dim, hidden, self.cfg.feature_dim).to(self.device)
        elif self.cfg.surrogate == "mlp_matched":
            target = self.cfg.matched_params
            if target == 0:
                reference = QNNSurrogate(
                    input_dim,
                    self.cfg.feature_dim,
                    self.cfg.qnn_layers,
                    self.cfg.qnn_max_z_order,
                    self.cfg.qnn_n_qubits,
                    self.cfg.qnn_unitary_fusion,
                    self.cfg.qnn_edge_specific,
                    self.cfg.qnn_input_tanh,
                    self.cfg.qnn_architecture,
                )
                target = trainable_parameter_count(reference)
                del reference
            self.model = ParameterMatchedMLP(input_dim, self.cfg.feature_dim, target).to(self.device)
        elif self.cfg.surrogate in ("qnn", "qnn_no_norm", "qnn_no_ent", "qnn_frozen_quantum"):
            self.model = QNNSurrogate(
                input_dim,
                self.cfg.feature_dim,
                self.cfg.qnn_layers,
                self.cfg.qnn_max_z_order,
                self.cfg.qnn_n_qubits,
                self.cfg.qnn_unitary_fusion,
                self.cfg.qnn_edge_specific,
                self.cfg.qnn_input_tanh,
                self.cfg.qnn_architecture,
                normalize_readout=self.cfg.surrogate != "qnn_no_norm",
                entangling=self.cfg.surrogate != "qnn_no_ent",
                freeze_quantum=self.cfg.surrogate == "qnn_frozen_quantum",
            ).to(self.device)
        else:
            raise ValueError(f"unknown surrogate '{self.cfg.surrogate}'")
        if self.cfg.load_checkpoint:
            self._load_checkpoint(self.cfg.load_checkpoint)
        if self.cfg.policy == "gp_ts" or self.cfg.max_retrains == 0 or (
            self.loaded_checkpoint and self.cfg.freeze_loaded_model
        ):
            self._freeze_surrogate()
        elif isinstance(self.model, QNNSurrogate):
            param_groups = self.model.optimizer_param_groups(
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                quantum_lr=self.cfg.quantum_lr,
                quantum_weight_decay=self.cfg.quantum_weight_decay,
            )
            self.optimizer = torch.optim.Adam(param_groups)
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )
        self._reset_blr()
        self._restore_loaded_checkpoint_state()

    def _freeze_surrogate(self) -> None:
        if self.model is None:
            return
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if isinstance(self.model, QNNSurrogate) and self.cfg.qnn_frozen_fusion:
            self.model.qlayer.prepare_frozen_inference()

        self.frozen_feature_extractor = None
        if isinstance(self.model, QNNSurrogate) and self.cfg.qnn_frozen_torchscript:
            assert self.input_dim is not None
            example = torch.zeros((1, self.input_dim), dtype=torch.float32, device=self.device)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    traced = torch.jit.trace(
                        _FeatureOnlyModule(self.model).eval(),
                        example,
                        check_trace=False,
                    )
                    self.frozen_feature_extractor = torch.jit.freeze(traced.eval())
                    # Pay the one-time graph specialization cost while the
                    # surrogate is being prepared, not on its first action.
                    warmup = torch.zeros(
                        (512, self.input_dim),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    self.frozen_feature_extractor(warmup)
            except (RuntimeError, TypeError, ValueError) as exc:
                warnings.warn(
                    f"TorchScript frozen QNN fast path unavailable; using eager fusion: {exc}",
                    RuntimeWarning,
                )

        self.feature_cache.clear()
        self.optimizer = None
        self.surrogate_frozen = True

    def _load_checkpoint(self, path: str) -> None:
        assert self.model is not None and self.input_dim is not None
        resolved = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"QNN checkpoint does not exist: {resolved}")
        try:
            payload = torch.load(resolved, map_location=self.device, weights_only=False)
        except TypeError:  # PyTorch before the weights_only keyword.
            payload = torch.load(resolved, map_location=self.device)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            checkpoint_dim = payload.get("input_dim")
            if checkpoint_dim is not None and int(checkpoint_dim) != self.input_dim:
                raise ValueError(
                    f"checkpoint input_dim={checkpoint_dim} does not match {self.input_dim}"
                )
            state_dict = payload["model_state_dict"]
        elif isinstance(payload, dict):
            state_dict = payload
        else:
            raise ValueError("checkpoint must be a state_dict or contain model_state_dict")
        self.model.load_state_dict(state_dict, strict=True)
        self.loaded_checkpoint = resolved
        self._loaded_checkpoint_payload = payload if isinstance(payload, dict) else None

    def _restore_loaded_checkpoint_state(self) -> None:
        if not self.cfg.restore_checkpoint_state or not self._loaded_checkpoint_payload:
            return
        payload = self._loaded_checkpoint_payload
        required = ("history_x", "history_y", "precision", "phi_y", "yty")
        missing = [name for name in required if payload.get(name) is None]
        if missing:
            raise ValueError(
                "checkpoint does not contain complete bandit state: " + ", ".join(missing)
            )
        history_x = np.asarray(payload["history_x"], dtype=np.float32)
        history_y = np.asarray(payload["history_y"], dtype=np.float32).reshape(-1)
        if history_x.ndim != 2 or history_x.shape[0] != history_y.shape[0]:
            raise ValueError("checkpoint history_x/history_y shapes are inconsistent")
        if self.input_dim is None or history_x.shape[1] != self.input_dim:
            raise ValueError("checkpoint history input dimension is inconsistent")
        precision = np.asarray(payload["precision"], dtype=np.float64)
        phi_y = np.asarray(payload["phi_y"], dtype=np.float64).reshape(-1)
        expected = (self.posterior_dim, self.posterior_dim)
        if precision.shape != expected or phi_y.shape != (self.posterior_dim,):
            raise ValueError("checkpoint Bayesian posterior dimension is inconsistent")
        cholesky_value = payload.get("precision_cholesky")
        cholesky = (
            np.asarray(cholesky_value, dtype=np.float64)
            if cholesky_value is not None
            else np.linalg.cholesky(precision)
        )
        if cholesky.shape != expected:
            raise ValueError("checkpoint precision Cholesky dimension is inconsistent")
        self.history_x = [row.copy() for row in history_x]
        self.history_y = [float(value) for value in history_y]
        self.precision = precision.copy()
        self.precision_cholesky = cholesky.copy()
        self.phi_y = phi_y.copy()
        self.yty = float(payload["yty"])
        self.n_obs = int(payload.get("n_obs", len(self.history_y)))
        if payload.get("feature_norm_sum") is not None:
            self.feature_norm_sum = float(payload["feature_norm_sum"])
            self.last_feature_norm = float(
                payload.get("last_feature_norm", float("nan"))
            )
        elif self.cfg.feature_diagnostics_log and self.history_x:
            # Older checkpoints lack this optional diagnostic state. Reproject
            # once so resumed runs report a mean over the full observed design.
            features = self._features_np(history_x)
            norms = np.linalg.norm(features, axis=1)
            self.feature_norm_sum = float(np.sum(norms))
            self.last_feature_norm = float(norms[-1])
        self.train_calls = int(payload.get("train_calls", 0))
        self.environment_step = int(payload.get("environment_step", 0))
        optimizer_state = payload.get("optimizer_state_dict")
        if self.optimizer is not None and optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)
        numpy_rng_state = payload.get("numpy_rng_state")
        if numpy_rng_state is not None:
            self.rng.bit_generator.state = numpy_rng_state
        torch_rng_state = payload.get("torch_rng_state")
        if torch_rng_state is not None:
            torch.set_rng_state(torch.as_tensor(torch_rng_state, dtype=torch.uint8).cpu())

    def _reset_blr(self) -> None:
        d = self.posterior_dim
        prior_precision = 1.0 if self.cfg.policy == "gp_ts" else self.cfg.prior_lambda
        self.precision = prior_precision * np.eye(d, dtype=np.float64)
        self.precision_cholesky = math.sqrt(prior_precision) * np.eye(
            d,
            dtype=np.float64,
        )
        self.phi_y = np.zeros(d, dtype=np.float64)
        self.yty = 0.0
        self.n_obs = 0
        self.feature_norm_sum = 0.0
        self.last_feature_norm = float("nan")

    @torch.inference_mode()
    def _compute_features_np(self, xs: np.ndarray) -> np.ndarray:
        assert self.model is not None
        x_tensor = torch.as_tensor(xs, dtype=torch.float32, device=self.device)
        if x_tensor.ndim == 1:
            x_tensor = x_tensor.unsqueeze(0)
        extractor = self.frozen_feature_extractor
        phi = extractor(x_tensor) if extractor is not None else self.model.features(x_tensor)
        return phi.cpu().numpy().astype(np.float64)

    @torch.inference_mode()
    def _features_np(self, xs: np.ndarray) -> np.ndarray:
        assert self.model is not None
        array = np.asarray(xs, dtype=np.float32)
        if array.ndim == 1:
            matrix = array.reshape(1, -1)
        elif array.ndim == 2:
            matrix = array
        else:
            raise ValueError("feature input must be a vector or matrix")

        if not self.surrogate_frozen or self.cfg.feature_cache_size <= 0:
            return self._compute_features_np(matrix)

        keys = [np.ascontiguousarray(row).tobytes() for row in matrix]
        cached = [self.feature_cache.get(key) for key in keys]
        hit_indices = [index for index, feature in enumerate(cached) if feature is not None]
        miss_indices = [index for index, feature in enumerate(cached) if feature is None]
        self.feature_cache_hits += len(hit_indices)
        self.feature_cache_misses += len(miss_indices)

        # Preserve the direct all-miss path when every candidate is unique; it
        # has the least Python dispatch for continuously changing proposal
        # pools.  Discrete HGM/HCM pools often contain repeated configurations,
        # however, so evaluate each exact float32 row only once and broadcast
        # its feature to every duplicate position.
        all_misses_unique = not hit_indices and len(set(keys)) == matrix.shape[0]
        if all_misses_unique:
            unique_miss_indices = miss_indices
            miss_positions = None
            output = self._compute_features_np(matrix)
        else:
            unique_miss_indices = []
            miss_positions = {}
            for index in miss_indices:
                key = keys[index]
                positions = miss_positions.get(key)
                if positions is None:
                    unique_miss_indices.append(index)
                    miss_positions[key] = [index]
                else:
                    positions.append(index)
            output = np.empty((matrix.shape[0], self.posterior_dim), dtype=np.float64)
            if hit_indices:
                output[hit_indices] = np.stack([cached[index] for index in hit_indices])
            if unique_miss_indices:
                unique_features = self._compute_features_np(matrix[unique_miss_indices])
                for feature, index in zip(unique_features, unique_miss_indices):
                    output[miss_positions[keys[index]]] = feature
            for index in hit_indices:
                self.feature_cache.move_to_end(keys[index])

        # Insert in last-use order so duplicate elimination does not change LRU
        # eviction semantics relative to processing the original row sequence.
        cache_insert_indices = (
            unique_miss_indices
            if miss_positions is None
            else sorted(
                unique_miss_indices,
                key=lambda index: miss_positions[keys[index]][-1],
            )
        )
        for index in cache_insert_indices:
            key = keys[index]
            self.feature_cache[key] = output[index].copy()
            self.feature_cache.move_to_end(key)
        while len(self.feature_cache) > self.cfg.feature_cache_size:
            self.feature_cache.popitem(last=False)
        return output

    def _rank_one_cholesky_update(self, vector: np.ndarray) -> None:
        """Update L in L L^T <- L L^T + vector vector^T."""
        assert self.precision_cholesky is not None
        work = np.asarray(vector, dtype=np.float64).copy()
        lower = self.precision_cholesky
        for index in range(lower.shape[0]):
            diagonal = lower[index, index]
            updated = math.hypot(diagonal, work[index])
            cosine = updated / diagonal
            sine = work[index] / diagonal
            lower[index, index] = updated
            if index + 1 < lower.shape[0]:
                column = (lower[index + 1 :, index] + sine * work[index + 1 :]) / cosine
                work[index + 1 :] = cosine * work[index + 1 :] - sine * column
                lower[index + 1 :, index] = column

    def _update_blr(self, phi: np.ndarray, reward: float) -> None:
        assert (
            self.precision is not None
            and self.precision_cholesky is not None
            and self.phi_y is not None
        )
        noise_variance = self.cfg.gp_noise ** 2 if self.cfg.policy == "gp_ts" else 1.0
        self.precision += np.outer(phi, phi) / noise_variance
        self._rank_one_cholesky_update(phi / math.sqrt(noise_variance))
        self.phi_y += phi * reward / noise_variance
        self.yty += reward * reward
        self.n_obs += 1
        self.last_feature_norm = float(np.linalg.norm(phi))
        self.feature_norm_sum += self.last_feature_norm

    def observe(self, x: np.ndarray, reward: float) -> None:
        self._ensure_model(int(x.shape[0]))
        if (
            self.cfg.freeze_after_step >= 0
            and self.environment_step > self.cfg.freeze_after_step
            and not self.surrogate_frozen
        ):
            self._freeze_surrogate()
        self.history_x.append(x.astype(np.float32))
        self.history_y.append(float(reward))

        n = len(self.history_y)
        enough = n >= self.cfg.cold_start
        early_interval = max(1, self.cfg.early_retrain_interval)
        due = (
            n == self.cfg.cold_start
            or (
                n > self.cfg.cold_start
                and n < self.cfg.early_step
                and (n - self.cfg.cold_start) % early_interval == 0
            )
            or n % self.cfg.retrain_interval == 0
        )
        may_retrain = (
            not self.surrogate_frozen
            and (self.cfg.max_retrains < 0 or self.train_calls < self.cfg.max_retrains)
        )
        if enough and due and may_retrain:
            self.train_surrogate()
            if self.cfg.max_retrains >= 0 and self.train_calls >= self.cfg.max_retrains:
                self._freeze_surrogate()
            self.rebuild_blr()
            self.log_feature_diagnostics()
            return

        # Before the first fit, cold-start decisions do not consult the BLR and
        # that fit will rebuild it under a new representation.  Defer those
        # otherwise wasted QNN forwards.  Fixed/frozen models still update from
        # the very first observation.
        if n < self.cfg.cold_start and not self.surrogate_frozen:
            self.log_feature_diagnostics()
            return
        phi = self._features_np(x)[0]
        self._update_blr(phi, float(reward))
        self.log_feature_diagnostics()

    def train_surrogate(self) -> None:
        assert self.model is not None and self.optimizer is not None
        timing_start = self.timing_start()
        xs = torch.as_tensor(np.stack(self.history_x), dtype=torch.float32, device=self.device)
        ys = torch.as_tensor(np.asarray(self.history_y), dtype=torch.float32, device=self.device)
        n = xs.shape[0]

        effective_epochs = (
            self.cfg.train_epochs
            if self.train_calls == 0 or self.cfg.update_epochs <= 0
            else self.cfg.update_epochs
        )
        self.model.train()
        self.train_calls += 1
        if self.cfg.train_log:
            self._write_train_log(0, "before", self._training_mse(xs, ys))
        for epoch in range(1, effective_epochs + 1):
            order = torch.randperm(n, device=self.device)
            weighted_loss = 0.0
            for start in range(0, n, self.cfg.batch_size):
                idx = order[start:start + self.cfg.batch_size]
                pred = self.model(xs[idx])
                loss = torch.mean((pred - ys[idx]) ** 2)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()
                weighted_loss += float(loss.detach().cpu()) * int(idx.shape[0])
            if self.cfg.train_log:
                self._write_train_log(epoch, "epoch", weighted_loss / max(1, n))
        self.model.eval()
        if isinstance(self.model, QNNSurrogate) and self.cfg.qnn_frozen_fusion:
            # Circuit weights remain fixed between scheduled retrains.  Use the
            # detached static ring during those inference intervals; the next
            # model.train() call automatically returns to the differentiable
            # gate path until this cache is refreshed here.
            self.model.qlayer.prepare_frozen_inference()
        if self.cfg.train_log:
            self._write_train_log(effective_epochs, "after", self._training_mse(xs, ys))
        self._save_checkpoint()
        self.record_timing("gradient_fit", timing_start)

    @torch.no_grad()
    def _training_mse(self, xs: torch.Tensor, ys: torch.Tensor) -> float:
        assert self.model is not None
        was_training = self.model.training
        self.model.eval()
        pred = self.model(xs)
        loss = torch.mean((pred - ys) ** 2)
        if was_training:
            self.model.train()
        return float(loss.detach().cpu())

    def _write_train_log(self, epoch: int, phase: str, loss: float) -> None:
        if not self.cfg.train_log:
            return
        path = self.cfg.train_log
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fields = [
            "unix_time",
            "seed",
            "surrogate",
            "retrain",
            "epoch",
            "phase",
            "n_obs",
            "input_dim",
            "feature_dim",
            "qnn_layers",
            "qnn_n_qubits",
            "qnn_max_z_order",
            "qnn_architecture",
            "early_retrain_interval",
            "max_retrains",
            "freeze_after_step",
            "effective_train_epochs",
            "loaded_checkpoint",
            "train_loss",
            "lr",
            "weight_decay",
            "quantum_lr",
            "quantum_weight_decay",
        ]
        row = {
            "unix_time": f"{time.time():.3f}",
            "seed": self.cfg.seed,
            "surrogate": self.cfg.surrogate,
            "retrain": self.train_calls,
            "epoch": epoch,
            "phase": phase,
            "n_obs": len(self.history_y),
            "input_dim": self.input_dim or "",
            "feature_dim": self.cfg.feature_dim,
            "qnn_layers": self.cfg.qnn_layers,
            "qnn_n_qubits": self.cfg.qnn_n_qubits,
            "qnn_max_z_order": self.cfg.qnn_max_z_order,
            "qnn_architecture": self.cfg.qnn_architecture,
            "early_retrain_interval": self.cfg.early_retrain_interval,
            "max_retrains": self.cfg.max_retrains,
            "freeze_after_step": self.cfg.freeze_after_step,
            "effective_train_epochs": (
                self.cfg.train_epochs
                if self.train_calls == 1 or self.cfg.update_epochs <= 0
                else self.cfg.update_epochs
            ),
            "loaded_checkpoint": self.loaded_checkpoint,
            "train_loss": f"{loss:.10g}",
            "lr": self.cfg.lr,
            "weight_decay": self.cfg.weight_decay,
            "quantum_lr": self.cfg.quantum_lr,
            "quantum_weight_decay": self.cfg.quantum_weight_decay,
        }
        with _TRAIN_LOG_LOCK:
            exists = os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if not exists:
                    writer.writeheader()
                writer.writerow(row)

    def _save_checkpoint(self, final: bool = False) -> None:
        if not self.cfg.checkpoint_dir or self.model is None:
            return
        if final:
            if not self.cfg.save_final_checkpoint or not self.history_y:
                return
        elif (
            self.cfg.checkpoint_every <= 0
            or self.train_calls % self.cfg.checkpoint_every != 0
        ):
            return
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        filename = (
            f"{self.cfg.surrogate}_seed{self.cfg.seed:06d}_final_"
            f"n{len(self.history_y):04d}.pt"
            if final
            else (
                f"{self.cfg.surrogate}_seed{self.cfg.seed:06d}_"
                f"retrain{self.train_calls:04d}_n{len(self.history_y):04d}.pt"
            )
        )
        path = os.path.join(self.cfg.checkpoint_dir, filename)
        temporary = path + ".tmp"
        payload = {
            "checkpoint_kind": "final" if final else "periodic",
            "cfg": asdict(self.cfg),
            "seed": self.cfg.seed,
            "input_dim": self.input_dim,
            "n_obs": self.n_obs,
            "history_count": len(self.history_y),
            "train_calls": self.train_calls,
            "environment_step": self.environment_step,
            "model_state_dict": self.model.state_dict(),
            "model_parameter_count": sum(parameter.numel() for parameter in self.model.parameters()),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer is not None else None
            ),
            "history_x": np.asarray(self.history_x, dtype=np.float32),
            "history_y": np.asarray(self.history_y, dtype=np.float32),
            "precision": self.precision,
            "precision_cholesky": self.precision_cholesky,
            "phi_y": self.phi_y,
            "yty": self.yty,
            "feature_norm_sum": self.feature_norm_sum,
            "last_feature_norm": self.last_feature_norm,
            "numpy_rng_state": self.rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
        }
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def rebuild_blr(self) -> None:
        self._reset_blr()
        if not self.history_x:
            return
        xs = np.stack(self.history_x)
        phis = self._features_np(xs)
        rewards = np.asarray(self.history_y, dtype=np.float64)
        assert self.precision is not None and self.phi_y is not None
        noise_variance = self.cfg.gp_noise ** 2 if self.cfg.policy == "gp_ts" else 1.0
        self.precision += (phis.T @ phis) / noise_variance
        self.precision_cholesky = np.linalg.cholesky(self.precision)
        self.phi_y += (phis.T @ rewards) / noise_variance
        self.yty = float(rewards @ rewards)
        self.n_obs = int(rewards.shape[0])
        norms = np.linalg.norm(phis, axis=1)
        self.feature_norm_sum = float(np.sum(norms))
        self.last_feature_norm = float(norms[-1])

    def _posterior_state(self) -> Tuple[np.ndarray, float, float, np.ndarray]:
        assert self.precision_cholesky is not None and self.phi_y is not None
        intermediate = _solve_triangular(self.precision_cholesky, self.phi_y, lower=True)
        mu = _solve_triangular(self.precision_cholesky.T, intermediate, lower=False)
        quad = float(mu @ self.phi_y)
        a = self.cfg.prior_a + 0.5 * self.n_obs
        b = self.cfg.prior_b + 0.5 * max(1.0e-12, self.yty - quad)
        return mu, a, b, self.precision_cholesky

    def posterior(self) -> Tuple[np.ndarray, float, float]:
        mu, a, b, _ = self._posterior_state()
        return mu, a, b

    def sample_posterior_w(
        self,
        posterior_state: Optional[Tuple[np.ndarray, float, float, np.ndarray]] = None,
    ) -> np.ndarray:
        mu, a, b, precision_cholesky = (
            self._posterior_state() if posterior_state is None else posterior_state
        )
        if self.cfg.posterior_scale <= 0.0:
            return mu
        if self.cfg.policy == "gp_ts":
            # One posterior weight draw scores the whole candidate pool,
            # preserving the spatial correlation required by Thompson sampling.
            deviation = _solve_triangular(
                precision_cholesky.T,
                self.rng.standard_normal(mu.shape[0]),
                lower=False,
            )
            return mu + self.cfg.posterior_scale * deviation
        tau = float(self.rng.gamma(shape=a, scale=1.0 / max(b, 1.0e-12)))
        standard_normal = self.rng.standard_normal(mu.shape[0])
        deviation = _solve_triangular(precision_cholesky.T, standard_normal, lower=False)
        deviation /= math.sqrt(max(tau, 1.0e-12))
        # Preserve the tiny numerical-jitter component from the former
        # covariance construction without forming or factorizing its inverse.
        deviation += math.sqrt(1.0e-9) * self.rng.standard_normal(mu.shape[0])
        return mu + self.cfg.posterior_scale * deviation

    def _ucb_scores(
        self,
        phis: np.ndarray,
        posterior_state: Optional[Tuple[np.ndarray, float, float, np.ndarray]] = None,
    ) -> np.ndarray:
        mu, a, b, precision_cholesky = (
            self._posterior_state() if posterior_state is None else posterior_state
        )
        projected = _solve_triangular(precision_cholesky, phis.T, lower=True)
        predictive_std = np.sqrt(
            (b / max(a, 1.0e-12)) * np.sum(projected * projected, axis=0)
        )
        return phis @ mu + self.cfg.ucb_beta * predictive_std

    def acquisition_scores(self, candidates: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        phis = self._features_np(candidates.astype(np.float32))
        if self.cfg.policy == "neural_ucb":
            return self._ucb_scores(phis)
        if w is None:
            w = self.sample_posterior_w()
        return phis @ w

    def top_observed_configs(
        self,
        k: int,
        current: Optional[np.ndarray] = None,
        best: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        centers: List[np.ndarray] = []
        for candidate in (best, current):
            if candidate is not None:
                centers.append(project_config(candidate))

        if self.history_y:
            order = np.argsort(np.asarray(self.history_y))[-k:][::-1]
            for idx in order:
                centers.append(project_config(self.history_x[int(idx)]))

        unique: List[np.ndarray] = []
        seen = set()
        for center in centers:
            key = tuple(np.round(center, 6).tolist())
            if key in seen:
                continue
            seen.add(key)
            unique.append(center)
            if len(unique) >= k:
                break
        return unique

    def schedule_changed_aps(self) -> int:
        step = len(self.history_y)
        if step < self.cfg.early_step:
            return self.cfg.early_changed_aps
        if step < self.cfg.late_step:
            return self.cfg.mid_changed_aps
        return self.cfg.late_changed_aps

    def schedule_delta_db(self) -> float:
        step = len(self.history_y)
        if step < self.cfg.early_step:
            return self.cfg.early_delta_db
        if step < self.cfg.late_step:
            return self.cfg.mid_delta_db
        return self.cfg.late_delta_db

    def safe_default_config(self, dim: int) -> np.ndarray:
        x = np.zeros(dim, dtype=np.float32)
        x[1::2] = 0.25
        return project_config(x)

    def propose_pair_around(self, center_s: float, center_p: float, sigma: float) -> Tuple[float, float]:
        for _ in range(max(1, self.cfg.local_resample_attempts)):
            s = float(center_s + self.rng.normal(0.0, sigma))
            p = float(center_p + self.rng.normal(0.0, sigma))
            if valid_pair(s, p):
                return s, p
        if valid_pair(float(center_s), float(center_p)):
            return float(center_s), float(center_p)
        return sample_feasible_pair(self.rng)

    def sparse_perturb(self, center: np.ndarray, max_changed_aps: int, max_delta_norm: float) -> np.ndarray:
        x = np.asarray(center, dtype=np.float32).copy()
        n_ap = max(1, x.shape[0] // 2)
        k = int(self.rng.integers(1, min(max_changed_aps, n_ap) + 1))
        aps = self.rng.choice(n_ap, size=k, replace=False)
        for ap in aps:
            j = 2 * int(ap)
            x[j], x[j + 1] = self.propose_pair_around(x[j], x[j + 1], max_delta_norm)
        return x

    def coordinate_pool(
        self,
        centers: List[np.ndarray],
        max_changed_aps: int,
        max_delta_norm: float,
        budget: int,
    ) -> List[np.ndarray]:
        pool: List[np.ndarray] = []
        if not centers or budget <= 0:
            return pool

        steps = np.asarray([-1.0, -0.5, -0.25, 0.25, 0.5, 1.0], dtype=np.float32) * max_delta_norm
        n_ap = centers[0].shape[0] // 2
        ap_order = np.arange(n_ap)
        self.rng.shuffle(ap_order)
        ap_order = ap_order[: min(n_ap, max(1, max_changed_aps * 6))]

        for center in centers:
            if len(pool) >= budget:
                break
            pool.append(project_config(center))
            for ap in ap_order:
                if len(pool) >= budget:
                    break
                base = int(2 * ap)
                for coord in (base, base + 1):
                    for delta in steps:
                        candidate = center.copy()
                        candidate[coord] += float(delta)
                        if config_valid(candidate):
                            pool.append(candidate.astype(np.float32))
                        if len(pool) >= budget:
                            break
                    if len(pool) >= budget:
                        break
        return pool

    def fill_sparse_pool(
        self,
        centers: List[np.ndarray],
        budget: int,
        max_changed_aps: int,
        max_delta_norm: float,
    ) -> np.ndarray:
        pool = self.coordinate_pool(centers, max_changed_aps, max_delta_norm, budget)
        while len(pool) < budget:
            center = centers[int(self.rng.integers(0, len(centers)))]
            pool.append(self.sparse_perturb(center, max_changed_aps, max_delta_norm))
        return np.stack(pool).astype(np.float32)

    def stochastic_score_choice(self, scores: np.ndarray) -> int:
        if scores.shape[0] == 0:
            raise ValueError("cannot choose from an empty score vector")
        top_k = min(max(1, self.cfg.acquisition_top_k), scores.shape[0])
        if top_k == 1 or self.cfg.acquisition_temperature <= 0.0:
            return int(np.argmax(scores))

        top_idx = np.argsort(scores)[-top_k:]
        top_scores = scores[top_idx].astype(np.float64)
        spread = max(float(np.std(top_scores)), 1.0e-6)
        logits = (top_scores - float(np.max(top_scores))) / (spread * self.cfg.acquisition_temperature)
        logits = np.clip(logits, -50.0, 50.0)
        probs = np.exp(logits)
        probs /= np.sum(probs)
        return int(top_idx[int(self.rng.choice(top_idx.shape[0], p=probs))])

    def safe_cold_start_config(
        self,
        dim: int,
        current: Optional[np.ndarray] = None,
        best: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        center = project_config(current) if current is not None else None
        if center is None and best is not None:
            center = project_config(best)
        if center is None:
            center = self.safe_default_config(dim)

        step = len(self.history_y)
        if step == 0:
            return center
        max_changed_aps = 1 if step < self.cfg.cold_start // 2 else min(4, max(1, dim // 2))
        max_delta_norm = min(4.0, self.schedule_delta_db()) / 20.0
        return self.sparse_perturb(center, max_changed_aps, max_delta_norm)

    def ask_config(
        self,
        dim: int,
        current: Optional[np.ndarray] = None,
        best: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._ensure_model(dim)

        if current is not None:
            current = project_config(current)
        if best is not None:
            best = project_config(best)

        if len(self.history_y) < self.cfg.cold_start:
            return self.safe_cold_start_config(dim, current=current, best=best)

        centers = self.top_observed_configs(self.cfg.top_centers, current=current, best=best)
        if not centers:
            centers = [self.safe_default_config(dim)]

        w = None if self.cfg.policy == "neural_ucb" else self.sample_posterior_w()
        max_changed_aps = self.schedule_changed_aps()
        max_delta_norm = self.schedule_delta_db() / 20.0
        budget = max(16, self.cfg.virtual_budget)

        pool = self.fill_sparse_pool(centers, budget, max_changed_aps, max_delta_norm)

        for _ in range(max(1, self.cfg.cem_iters)):
            scores = self.acquisition_scores(pool, w)

            elite_n = max(8, int(self.cfg.elite_frac * pool.shape[0]))
            elite_idx = np.argsort(scores)[-elite_n:]
            elite_scores = scores[elite_idx]
            elites = pool[elite_idx]
            top_idx = np.argsort(elite_scores)[-self.cfg.top_centers:]
            centers = [project_config(x) for x in elites[top_idx]]
            pool = self.fill_sparse_pool(centers, budget, max_changed_aps, max_delta_norm)

        scores = self.acquisition_scores(pool, w)
        idx = self.stochastic_score_choice(scores)
        chosen = project_config(pool[idx])
        self.last_stats = {
            "candidate_boundary_rate": boundary_rate(pool),
            "chosen_boundary_rate": boundary_rate(chosen.reshape(1, -1)),
        }
        return chosen

    def choose_candidate(
        self,
        dim: int,
        candidates: np.ndarray,
        current: Optional[np.ndarray] = None,
        best: Optional[np.ndarray] = None,
        collect_diagnostics: bool = True,
    ) -> int:
        self._ensure_model(dim)
        if candidates.ndim != 2 or candidates.shape[1] != dim:
            raise ValueError(f"candidate matrix must have shape (*, {dim})")
        if candidates.shape[0] == 0:
            raise ValueError("ASK_CHOICE requires at least one candidate")

        if len(self.history_y) < self.cfg.cold_start:
            chosen = 0
            self.last_stats = (
                {
                    "candidate_boundary_rate": boundary_rate(candidates),
                    "chosen_boundary_rate": boundary_rate(
                        candidates[chosen].reshape(1, -1)
                    ),
                }
                if collect_diagnostics
                else {}
            )
            return chosen

        # Keep an explicit exploration floor for finite sampler pools.  This is
        # separate from posterior sampling: once a surrogate becomes confident,
        # Thompson scores can rank the same incumbent first indefinitely.  The
        # floor selects only configurations not observed before, preserving the
        # HCM proposal distribution instead of inventing off-sampler actions.
        epsilon = float(np.clip(self.cfg.acquisition_epsilon, 0.0, 1.0))
        # The unseen pool changes the action only when the explicit exploration
        # floor is enabled.  Avoid constructing hundreds of rounded Python
        # tuples on the production epsilon=0 path merely for verbose metrics.
        unseen = None
        if epsilon > 0.0 or collect_diagnostics:
            seen = {
                tuple(np.round(np.asarray(x, dtype=np.float32), 6).tolist())
                for x in self.history_x
            }
            unseen = np.asarray(
                [
                    idx
                    for idx, candidate in enumerate(candidates)
                    if tuple(np.round(candidate, 6).tolist()) not in seen
                ],
                dtype=np.int64,
            )
        if (
            unseen is not None
            and unseen.size > 0
            and epsilon > 0.0
            and float(self.rng.random()) < epsilon
        ):
            chosen = int(self.rng.choice(unseen))
            self.last_stats = (
                {
                    "candidate_boundary_rate": boundary_rate(candidates),
                    "chosen_boundary_rate": boundary_rate(
                        candidates[chosen].reshape(1, -1)
                    ),
                    "forced_exploration": 1.0,
                    "unseen_candidates": float(unseen.size),
                }
                if collect_diagnostics
                else {}
            )
            return chosen

        posterior_state = self._posterior_state()
        phis = self._features_np(candidates.astype(np.float32))
        if self.cfg.policy == "neural_ucb":
            scores = self._ucb_scores(phis, posterior_state)
        else:
            w = self.sample_posterior_w(posterior_state)
            scores = phis @ w
        chosen = self.stochastic_score_choice(scores)
        if collect_diagnostics:
            assert unseen is not None
            posterior_mu = posterior_state[0]
            mean_scores = phis @ posterior_mu
            self.last_stats = {
                "candidate_boundary_rate": boundary_rate(candidates),
                "chosen_boundary_rate": boundary_rate(candidates[chosen].reshape(1, -1)),
                "score_std": float(np.std(scores)),
                "score_gap": float(np.max(scores) - np.median(scores)),
                "mean_score_gap": float(np.max(mean_scores) - np.median(mean_scores)),
                "chosen_mean_advantage": float(mean_scores[chosen] - mean_scores[0]),
                "chosen_seen": float(chosen not in unseen),
                "forced_exploration": 0.0,
                "unseen_candidates": float(unseen.size),
            }
        else:
            self.last_stats = {}
        return chosen


class NeuralBanditHandler(socketserver.StreamRequestHandler):
    def _new_bandit(self, seed: int):
        cfg = self.server.bandit_config(seed)
        return PoolArmBandit(cfg) if cfg.policy == "per_arm_ts" else NeuralLinearBandit(cfg)

    def setup(self) -> None:
        super().setup()
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.seed = self.server.next_seed()
        self.bandit = self._new_bandit(self.seed)

    def handle(self) -> None:
        peer = self.client_address[0] if self.client_address else "client"
        if self.server.verbose:
            print(f"nb-bridge: connection from {peer}", flush=True)
        try:
            while True:
                raw = self.rfile.readline()
                if not raw:
                    break
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    if parts[0] == "HELLO":
                        self._handle_hello(parts)
                    elif parts[0] == "STEP":
                        self._handle_step(parts)
                    elif parts[0] == "OBSERVE":
                        self._handle_observe(parts)
                    elif parts[0] == "ASK_CHOICE":
                        self._handle_ask_choice(parts)
                    elif parts[0] == "ASK_CONFIG":
                        self._handle_ask_config(parts)
                    else:
                        raise ValueError(f"unknown command {parts[0]}")
                except Exception as exc:
                    print(f"nb-bridge: protocol error: {exc}", flush=True)
                    break
        finally:
            self.bandit.flush_timing()
            self.bandit._save_checkpoint(final=True)
        if self.server.verbose:
            print(f"nb-bridge: connection from {peer} closed", flush=True)

    def _handle_hello(self, parts: List[str]) -> None:
        if len(parts) not in (2, 3):
            raise ValueError("HELLO requires seed and optional input dimension")
        seed = int(parts[1])
        if self.bandit.history_y:
            print(
                "nb-bridge: HELLO received after observations; keeping current bandit state",
                flush=True,
            )
            return
        self.seed = seed
        self.bandit = self._new_bandit(seed)
        if len(parts) == 3:
            input_dim = int(parts[2])
            if input_dim <= 0 or input_dim % 2:
                raise ValueError("HELLO input dimension must be positive and even")
            # Model/optimizer construction and one inference-kernel warm-up are
            # deployment setup, not online computation after a reward arrives.
            self.bandit._ensure_model(input_dim)
            self.bandit._features_np(np.zeros(input_dim, dtype=np.float32))
        if self.server.verbose:
            print(
                f"nb-bridge: hello seed={seed} input_dim={self.bandit.input_dim}",
                flush=True,
            )

    def _handle_step(self, parts: List[str]) -> None:
        if len(parts) != 2:
            raise ValueError("STEP requires an environment-step index")
        step = int(parts[1])
        if step < 0:
            raise ValueError("STEP index must be non-negative")
        self.bandit.environment_step = step

    def _handle_observe(self, parts: List[str]) -> None:
        if len(parts) < 4:
            raise ValueError("OBSERVE requires dim, reward, and features")
        dim = int(parts[1])
        reward = float(parts[2])
        values = [float(v) for v in parts[3:]]
        if len(values) != dim:
            raise ValueError(f"OBSERVE dim={dim} but got {len(values)} values")
        timing_start = self.bandit.timing_start()
        self.bandit.observe(np.asarray(values, dtype=np.float32), reward)
        self.bandit.record_timing("training_update", timing_start)
        if self.server.verbose:
            print(
                f"nb-bridge: observe n={len(self.bandit.history_y)} reward={reward:.5f}",
                flush=True,
            )

    def _handle_ask_config(self, parts: List[str]) -> None:
        if len(parts) != 2:
            raise ValueError("ASK_CONFIG requires dim")
        dim = int(parts[1])
        current = None
        best = None

        while True:
            raw = self.rfile.readline()
            if not raw:
                raise ValueError("EOF while reading ASK_CONFIG payload")
            line = raw.decode("utf-8").strip()
            if line == "END":
                break
            payload = line.split()
            if not payload:
                continue
            key = payload[0]
            values = [float(v) for v in payload[1:]]
            if len(values) != dim:
                raise ValueError(f"{key} dim={dim} but got {len(values)} values")
            if key == "CURRENT":
                current = np.asarray(values, dtype=np.float32)
            elif key == "BEST":
                best = np.asarray(values, dtype=np.float32)
            else:
                raise ValueError(f"unknown ASK_CONFIG field {key}")

        timing_start = self.bandit.timing_start()
        config = self.bandit.ask_config(dim, current=current, best=best)
        self.bandit.record_timing("inference_model", timing_start)
        values = " ".join(f"{float(v):.12g}" for v in config)
        self.wfile.write(f"CONFIG {dim} {values}\n".encode("utf-8"))
        self.wfile.flush()
        if self.server.verbose:
            stats = self.bandit.last_stats
            print(
                f"nb-bridge: ask_config n={len(self.bandit.history_y)} dim={dim} "
                f"candidate_boundary={stats.get('candidate_boundary_rate', 0.0):.4f} "
                f"chosen_boundary={stats.get('chosen_boundary_rate', 0.0):.4f} "
                f"score_std={stats.get('score_std', 0.0):.6f} "
                f"score_gap={stats.get('score_gap', 0.0):.6f} "
                f"mean_gap={stats.get('mean_score_gap', 0.0):.6f} "
                f"chosen_mean_adv={stats.get('chosen_mean_advantage', 0.0):.6f} "
                f"chosen_seen={int(stats.get('chosen_seen', 0.0))} "
                f"forced_explore={int(stats.get('forced_exploration', 0.0))} "
                f"unseen={int(stats.get('unseen_candidates', 0.0))}",
                flush=True,
            )

    def _handle_ask_choice(self, parts: List[str]) -> None:
        if len(parts) != 3:
            raise ValueError("ASK_CHOICE requires dim and candidate count")
        dim = int(parts[1])
        expected_candidates = int(parts[2])
        current = None
        best = None
        candidates: List[np.ndarray] = []

        while True:
            raw = self.rfile.readline()
            if not raw:
                raise ValueError("EOF while reading ASK_CHOICE payload")
            line = raw.decode("utf-8").strip()
            if line == "END":
                break
            payload = line.split()
            if not payload:
                continue
            key = payload[0]
            values = [float(v) for v in payload[1:]]
            if len(values) != dim:
                raise ValueError(f"{key} dim={dim} but got {len(values)} values")
            array = np.asarray(values, dtype=np.float32)
            if key == "CURRENT":
                current = array
            elif key == "BEST":
                best = array
            elif key == "CANDIDATE":
                candidates.append(array)
            else:
                raise ValueError(f"unknown ASK_CHOICE field {key}")

        if len(candidates) != expected_candidates:
            raise ValueError(
                f"ASK_CHOICE declared {expected_candidates} candidates but got {len(candidates)}"
            )

        candidate_matrix = np.stack(candidates).astype(np.float32)
        timing_start = self.bandit.timing_start()
        chosen = self.bandit.choose_candidate(
            dim,
            candidate_matrix,
            current=current,
            best=best,
            collect_diagnostics=self.server.verbose,
        )
        self.bandit.record_timing(
            "inference_model",
            timing_start,
            n_candidates=expected_candidates,
        )
        self.wfile.write(f"CHOICE {chosen}\n".encode("utf-8"))
        self.wfile.flush()
        if self.server.verbose:
            stats = self.bandit.last_stats
            print(
                f"nb-bridge: ask_choice n={len(self.bandit.history_y)} "
                f"dim={dim} candidates={len(candidates)} choice={chosen} "
                f"candidate_boundary={stats.get('candidate_boundary_rate', 0.0):.4f} "
                f"chosen_boundary={stats.get('chosen_boundary_rate', 0.0):.4f} "
                f"score_std={stats.get('score_std', 0.0):.6f} "
                f"score_gap={stats.get('score_gap', 0.0):.6f} "
                f"mean_gap={stats.get('mean_score_gap', 0.0):.6f} "
                f"chosen_mean_adv={stats.get('chosen_mean_advantage', 0.0):.6f} "
                f"chosen_seen={int(stats.get('chosen_seen', 0.0))} "
                f"forced_explore={int(stats.get('forced_exploration', 0.0))} "
                f"unseen={int(stats.get('unseen_candidates', 0.0))}",
                flush=True,
            )


class ThreadedBanditServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, handler_cls, args):
        super().__init__(server_address, handler_cls)
        self.args = args
        self.verbose = args.verbose
        self._seed_lock = threading.Lock()
        self._connections = 0

    def next_seed(self) -> int:
        with self._seed_lock:
            seed = self.args.seed + self._connections
            self._connections += 1
            return seed

    def bandit_config(self, seed: int) -> BanditConfig:
        return BanditConfig(
            hidden=self.args.hidden,
            feature_dim=self.args.feature_dim,
            surrogate=self.args.surrogate,
            policy=self.args.policy,
            matched_params=self.args.matched_params,
            ucb_beta=self.args.ucb_beta,
            gp_features=self.args.gp_features,
            gp_length_scale=self.args.gp_length_scale,
            gp_noise=self.args.gp_noise,
            qnn_layers=self.args.qnn_layers,
            qnn_n_qubits=self.args.qnn_n_qubits,
            qnn_max_z_order=self.args.qnn_max_z_order,
            qnn_unitary_fusion=self.args.qnn_unitary_fusion,
            qnn_edge_specific=self.args.qnn_edge_specific,
            qnn_input_tanh=self.args.qnn_input_tanh,
            qnn_architecture=self.args.qnn_architecture,
            retrain_interval=self.args.retrain_interval,
            early_retrain_interval=self.args.early_retrain_interval,
            max_retrains=self.args.max_retrains,
            freeze_after_step=self.args.freeze_after_step,
            cold_start=self.args.cold_start,
            train_epochs=self.args.train_epochs,
            update_epochs=self.args.update_epochs,
            batch_size=self.args.batch_size,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            quantum_lr=self.args.quantum_lr,
            quantum_weight_decay=self.args.quantum_weight_decay,
            train_log=self.args.train_log,
            timing_log=self.args.timing_log,
            feature_diagnostics_log=self.args.feature_diagnostics_log,
            timing_method=self.args.timing_method,
            timing_topology=self.args.timing_topology,
            checkpoint_dir=self.args.checkpoint_dir,
            checkpoint_every=self.args.checkpoint_every,
            save_final_checkpoint=self.args.save_final_checkpoint,
            load_checkpoint=self.args.load_checkpoint,
            restore_checkpoint_state=self.args.restore_checkpoint_state,
            freeze_loaded_model=self.args.freeze_loaded_model,
            qnn_frozen_fusion=self.args.qnn_frozen_fusion,
            qnn_frozen_torchscript=self.args.qnn_frozen_torchscript,
            feature_cache_size=self.args.feature_cache_size,
            prior_lambda=self.args.prior_lambda,
            prior_a=self.args.prior_a,
            prior_b=self.args.prior_b,
            virtual_budget=self.args.virtual_budget,
            cem_iters=self.args.cem_iters,
            elite_frac=self.args.elite_frac,
            top_centers=self.args.top_centers,
            early_step=self.args.early_step,
            late_step=self.args.late_step,
            early_changed_aps=self.args.early_changed_aps,
            mid_changed_aps=self.args.mid_changed_aps,
            late_changed_aps=self.args.late_changed_aps,
            early_delta_db=self.args.early_delta_db,
            mid_delta_db=self.args.mid_delta_db,
            late_delta_db=self.args.late_delta_db,
            acquisition_top_k=self.args.acquisition_top_k,
            acquisition_temperature=self.args.acquisition_temperature,
            acquisition_epsilon=self.args.acquisition_epsilon,
            posterior_scale=self.args.posterior_scale,
            local_resample_attempts=self.args.local_resample_attempts,
            seed=seed,
            device=self.args.device,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Neural-Linear bandit choices to ns-3 over TCP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9877)
    parser.add_argument(
        "--surrogate",
        choices=(
            "mlp_norm", "qnn",
        ),
        default="mlp_norm",
    )
    parser.add_argument(
        "--policy", choices=("neural_ts",), default="neural_ts",
        help="Acquisition policy. GP-TS scores raw configurations with an RBF random-feature GP.",
    )
    parser.add_argument("--matched-params", type=int, default=0,
                        help="Exact MLP parameter target; 0 counts the QNN specified by the QNN flags.")
    parser.add_argument("--ucb-beta", type=float, default=1.0)
    parser.add_argument("--gp-features", type=int, default=128)
    parser.add_argument("--gp-length-scale", type=float, default=0.5)
    parser.add_argument("--gp-noise", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, choices=(64,), default=64)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--qnn-layers", type=int, choices=(3,), default=3)
    parser.add_argument("--qnn-n-qubits", type=int, choices=(5,), default=5)
    parser.add_argument("--qnn-max-z-order", type=int, choices=(5,), default=5)
    parser.add_argument(
        "--qnn-architecture",
        choices=("qcnn_shared_reupload",),
        default="qcnn_shared_reupload",
        help=(
            "Circuit layout: legacy ring, dual-path, direct-angle, or shared-latent "
            "amplitude/angle reupload."
        ),
    )
    parser.add_argument(
        "--qnn-unitary-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evolve the computational basis once per forward, then apply the fused unitary.",
    )
    parser.add_argument(
        "--qnn-edge-specific",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use an independent 15-angle U_SU4 block on each ring edge.",
    )
    parser.add_argument(
        "--qnn-input-tanh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply tanh before L2 amplitude normalization.",
    )
    parser.add_argument("--retrain-interval", type=int, default=50)
    parser.add_argument("--early-retrain-interval", type=int, default=10)
    parser.add_argument(
        "--max-retrains",
        type=int,
        default=-1,
        help=(
            "Maximum online surrogate retrains: -1 keeps periodic retraining, "
            "0 freezes random features, and 1 trains once then freezes."
        ),
    )
    parser.add_argument(
        "--freeze-after-step",
        type=int,
        default=-1,
        help=(
            "Freeze the surrogate after this environment-step budget; -1 disables. "
            "Unlike max-retrains, this is sampler-independent."
        ),
    )
    parser.add_argument("--cold-start", type=int, default=30)
    parser.add_argument("--train-epochs", type=int, default=80)
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=0,
        help=(
            "Epochs for warm-start retrains after the first fit; 0 reuses "
            "--train-epochs. A small value keeps online adaptation inexpensive."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--quantum-lr", type=float, default=3.0e-3)
    parser.add_argument("--quantum-weight-decay", type=float, default=0.0)
    parser.add_argument("--train-log", default="")
    parser.add_argument("--timing-log", default="")
    parser.add_argument(
        "--feature-diagnostics-log",
        default="",
        help="Append one CSV row per observation with precision conditioning and feature norms.",
    )
    parser.add_argument("--timing-method", default="")
    parser.add_argument("--timing-topology", default="")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument(
        "--save-final-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Atomically save one complete model and BLR state when the client disconnects.",
    )
    parser.add_argument(
        "--load-checkpoint",
        default="",
        help="Load a separately pretrained surrogate checkpoint before the episode.",
    )
    parser.add_argument(
        "--restore-checkpoint-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restore the saved observation history and Bayesian posterior for exact policy inference.",
    )
    parser.add_argument(
        "--freeze-loaded-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Freeze a loaded checkpoint and use it only for feature inference; "
            "the Bayesian linear posterior still updates online."
        ),
    )
    parser.add_argument(
        "--qnn-frozen-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuse the input-independent U_SU4 ring between QNN weight updates.",
    )
    parser.add_argument(
        "--qnn-frozen-torchscript",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trace and freeze the accelerated QNN feature path for inference.",
    )
    parser.add_argument(
        "--feature-cache-size",
        type=int,
        default=65536,
        help="Maximum exact frozen feature vectors cached in LRU order; 0 disables.",
    )
    parser.add_argument("--prior-lambda", type=float, default=1.0)
    parser.add_argument("--prior-a", type=float, default=0.5)
    parser.add_argument("--prior-b", type=float, default=0.025)
    parser.add_argument("--virtual-budget", type=int, default=2048)
    parser.add_argument("--cem-iters", type=int, default=3)
    parser.add_argument("--elite-frac", type=float, default=0.05)
    parser.add_argument("--top-centers", type=int, default=8)
    parser.add_argument("--early-step", type=int, default=300)
    parser.add_argument("--late-step", type=int, default=800)
    parser.add_argument("--early-changed-aps", type=int, default=4)
    parser.add_argument("--mid-changed-aps", type=int, default=8)
    parser.add_argument("--late-changed-aps", type=int, default=16)
    parser.add_argument("--early-delta-db", type=float, default=4.0)
    parser.add_argument("--mid-delta-db", type=float, default=6.0)
    parser.add_argument("--late-delta-db", type=float, default=8.0)
    parser.add_argument("--acquisition-top-k", type=int, default=1)
    parser.add_argument("--acquisition-temperature", type=float, default=0.0)
    parser.add_argument(
        "--acquisition-epsilon",
        type=float,
        default=0.0,
        help="Probability of selecting an unseen sampler candidate (anti-collapse floor).",
    )
    parser.add_argument(
        "--posterior-scale",
        type=float,
        default=1.0,
        help=(
            "Scale of Thompson posterior deviations around the posterior mean; "
            "1 preserves NeuralTS and 0 uses the posterior mean."
        ),
    )
    parser.add_argument("--local-resample-attempts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.acquisition_epsilon <= 1.0:
        raise SystemExit("--acquisition-epsilon must be in [0, 1]")
    if args.posterior_scale < 0.0:
        raise SystemExit("--posterior-scale must be >= 0")
    if args.ucb_beta < 0.0:
        raise SystemExit("--ucb-beta must be >= 0")
    if args.matched_params < 0:
        raise SystemExit("--matched-params must be >= 0")
    if args.gp_features < 1 or args.gp_length_scale <= 0 or args.gp_noise <= 0:
        raise SystemExit("GP feature count, length scale, and noise must be positive")
    if args.max_retrains < -1:
        raise SystemExit("--max-retrains must be -1, 0, or a positive integer")
    if args.freeze_after_step < -1:
        raise SystemExit("--freeze-after-step must be -1 or a non-negative integer")
    if args.update_epochs < 0:
        raise SystemExit("--update-epochs must be >= 0")
    if args.feature_cache_size < 0:
        raise SystemExit("--feature-cache-size must be >= 0")
    if args.prior_lambda <= 0.0:
        raise SystemExit("--prior-lambda must be > 0 for Cholesky posterior updates")
    if args.load_checkpoint and not os.path.isfile(os.path.expanduser(args.load_checkpoint)):
        raise SystemExit(f"--load-checkpoint does not exist: {args.load_checkpoint}")
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    with ThreadedBanditServer((args.host, args.port), NeuralBanditHandler, args) as server:
        print(f"nb-bridge: listening on {args.host}:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("nb-bridge: interrupted", flush=True)


if __name__ == "__main__":
    main()
