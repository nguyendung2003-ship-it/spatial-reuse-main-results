"""Immutable, numerically equivalent feature inference for trained paper models.

Compilation, model copies and static circuit fusion happen during construction,
never during feature inference. No candidate features are memoized. The native
kernel supports the evaluated five-qubit, three-layer shared-reupload QNN and
performs the same amplitude evolution, Z observables and normalized readout.
"""
from __future__ import annotations

import copy
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import torch
from torch import nn


class _FeaturesOnly(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.model.features(xs)


def _immutable_copy(model: nn.Module) -> nn.Module:
    copied = copy.deepcopy(model).cpu().eval()
    for parameter in copied.parameters():
        parameter.requires_grad_(False)
    if hasattr(copied, "qlayer"):
        copied.qlayer.prepare_frozen_inference()
    return copied


class FrozenTorchFeatures:
    """TorchScript with constant weights, static entangler cache and no gradients."""
    backend = "torchscript_frozen"

    def __init__(self, model: nn.Module, example_batch: int = 512):
        copied = _immutable_copy(model)
        input_dim = getattr(copied, "input_dim", None)
        if input_dim is None:
            input_dim = next(module.in_features for module in copied.modules()
                             if isinstance(module, nn.Linear))
        wrapper = _FeaturesOnly(copied).eval()
        with torch.inference_mode():
            traced = torch.jit.trace(wrapper, torch.zeros(example_batch, input_dim),
                                     check_trace=False)
            self.module = torch.jit.freeze(traced.eval())
            self.module(torch.zeros(example_batch, input_dim))
        self.input_dim = input_dim

    def features_np(self, xs: np.ndarray) -> np.ndarray:
        values = np.ascontiguousarray(xs, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError("Expected a two-dimensional input of the model input width")
        with torch.inference_mode():
            return self.module(torch.from_numpy(values)).numpy()

    __call__ = features_np


class FrozenNumpyQNN:
    """Vectorized NumPy CPU QNN with fused observable/readout linear transform."""
    backend = "numpy_frozen_fused"

    def __init__(self, model: nn.Module):
        copied = _immutable_copy(model)
        if getattr(copied, "architecture", None) != "qcnn_shared_reupload":
            raise ValueError("Frozen QNN backend requires shared-reupload architecture")
        self.input_dim = copied.input_dim
        self.q = copied.n_qubits
        self.layers = copied.qnn_layers
        self.dim = 1 << self.q
        self.feature_dim = copied.measure_to_feature.out_features
        self.normalized = copied.normalize_readout
        self.encoder = np.ascontiguousarray(copied.input_layer.weight.detach().numpy().T)
        self.encoder_bias = np.ascontiguousarray(copied.input_layer.bias.detach().numpy())
        self.unitaries = np.ascontiguousarray(copied.qlayer._frozen_layer_unitaries.numpy())
        signs = copied.qlayer._measurement_signs.detach().numpy().T
        weights = copied.measure_to_feature.weight.detach().numpy().T
        self.projection = np.ascontiguousarray(signs @ weights)
        self.bias = np.ascontiguousarray(copied.measure_to_feature.bias.detach().numpy())

    def features_np(self, xs: np.ndarray) -> np.ndarray:
        xs = np.ascontiguousarray(xs, dtype=np.float32)
        if xs.ndim != 2 or xs.shape[1] != self.input_dim:
            raise ValueError("Expected a two-dimensional input of the model input width")
        z = np.tanh(xs @ self.encoder + self.encoder_bias)
        norm = np.sqrt(np.sum(z*z, axis=1, keepdims=True))
        state = (z / np.maximum(norm, np.float32(1e-12))).astype(np.complex64)
        for layer in range(self.layers):
            for wire in range(self.q):
                k = (layer * self.q + wire) * 2
                y = np.float32(np.pi*.5) * z[:, k % self.dim]
                p = np.float32(np.pi*.5) * z[:, (k+1) % self.dim]
                c, s = np.cos(y)[:, None, None], np.sin(y)[:, None, None]
                phase = (np.cos(p) + np.complex64(1j)*np.sin(p))[:, None, None]
                view = state.reshape(xs.shape[0], 1 << wire, 2, 1 << (self.q-wire-1))
                a, b = view[:, :, 0, :].copy(), view[:, :, 1, :].copy()
                view[:, :, 0, :] = (c*a-s*b) * phase.conj()
                view[:, :, 1, :] = (s*a+c*b) * phase
            if self.unitaries.size:
                state = state @ self.unitaries[layer]
        probabilities = state.real*state.real + state.imag*state.imag
        projected = probabilities @ self.projection + self.bias
        if not self.normalized:
            return projected
        values = np.tanh(projected)
        norms = np.maximum(np.sqrt(np.sum(values*values, axis=1, keepdims=True)), np.float32(1e-6))
        return values / norms * np.float32(self.feature_dim**.5)

    __call__ = features_np


_NATIVE_SOURCE = r'''
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>
extern "C" {
void cblas_sgemm(const int, const int, const int, const int, const int, const int,
                const float, const float*, const int, const float*, const int,
                const float, float*, const int);
void cblas_cgemm(const int, const int, const int, const int, const int, const int,
                const void*, const void*, const int, const void*, const int,
                const void*, void*, const int);
void qnn_features(const int batch, const int input_dim, const int feature_dim,
                  const int normalized, const float* x, const float* enc,
                  const float* enc_bias, const float* unitary, const float* projection,
                  const float* bias, float* z, float* workspace_state,
                  float* workspace_next, float* probabilities, float* output) {
    constexpr int D = 32, Q = 5, L = 3;
    constexpr float PI = 3.14159265358979323846f;
    constexpr int RowMajor = 101, NoTrans = 111;
    cblas_sgemm(RowMajor, NoTrans, NoTrans, batch, D, input_dim,
                1.f, x, input_dim, enc, D, 0.f, z, D);
    auto* state = reinterpret_cast<std::complex<float>*>(workspace_state);
    auto* next = reinterpret_cast<std::complex<float>*>(workspace_next);
    for (int b = 0; b < batch; ++b) {
        float sum = 0.f;
        for (int i = 0; i < D; ++i) {
            float v = std::tanh(z[b*D+i] + enc_bias[i]);
            z[b*D+i] = v;
            sum += v*v;
        }
        float norm = std::max(std::sqrt(sum), 1.e-12f);
        for (int i = 0; i < D; ++i) state[b*D+i] = {z[b*D+i]/norm, 0.f};
    }
    const std::complex<float> one(1.f,0.f), zero(0.f,0.f);
    for (int layer = 0; layer < L; ++layer) {
        for (int b = 0; b < batch; ++b) {
            std::complex<float>* row = state + b*D;
            for (int wire = 0; wire < Q; ++wire) {
                int k = (layer*Q+wire)*2;
                float y = (PI*z[b*D+k])*.5f;
                float p = (PI*z[b*D+k+1])*.5f;
                float c = std::cos(y), s = std::sin(y);
                float cp = std::cos(p), sp = std::sin(p);
                int stride = 1 << (Q-wire-1);
                for (int outer = 0; outer < D; outer += stride*2) {
                    for (int inner = 0; inner < stride; ++inner) {
                        int ai = outer+inner, bi = ai+stride;
                        auto a = row[ai], d = row[bi];
                        auto ra = c*a-s*d, rb = s*a+c*d;
                        row[ai] = {cp*ra.real()+sp*ra.imag(), cp*ra.imag()-sp*ra.real()};
                        row[bi] = {cp*rb.real()-sp*rb.imag(), cp*rb.imag()+sp*rb.real()};
                    }
                }
            }
        }
        cblas_cgemm(RowMajor, NoTrans, NoTrans, batch, D, D, &one,
                    state, D, unitary+layer*D*D*2, D, &zero, next, D);
        std::swap(state, next);
    }
    for (int i = 0; i < batch*D; ++i) {
        float re = state[i].real(), im = state[i].imag();
        probabilities[i] = re*re+im*im;
    }
    cblas_sgemm(RowMajor, NoTrans, NoTrans, batch, feature_dim, D,
                1.f, probabilities, D, projection, feature_dim, 0.f, output, feature_dim);
    for (int b = 0; b < batch; ++b) {
        float sum = 0.f;
        for (int i = 0; i < feature_dim; ++i) {
            float v = output[b*feature_dim+i]+bias[i];
            if (normalized) v = std::tanh(v);
            output[b*feature_dim+i] = v;
            sum += v*v;
        }
        if (normalized) {
            float scale = std::sqrt(float(feature_dim))/std::max(std::sqrt(sum),1.e-6f);
            for (int i = 0; i < feature_dim; ++i) output[b*feature_dim+i] *= scale;
        }
    }
}
void qnn_evolve_precomputed(const int batch, const float* amplitudes,
                           const float* trig, const float* unitary,
                           float* workspace_state, float* workspace_next,
                           float* probabilities) {
    constexpr int D = 32, Q = 5, L = 3;
    constexpr int RowMajor = 101, NoTrans = 111;
    auto* state = reinterpret_cast<std::complex<float>*>(workspace_state);
    auto* next = reinterpret_cast<std::complex<float>*>(workspace_next);
    for (int i = 0; i < batch*D; ++i) state[i] = {amplitudes[i], 0.f};
    const std::complex<float> one(1.f,0.f), zero(0.f,0.f);
    for (int layer = 0; layer < L; ++layer) {
        for (int b = 0; b < batch; ++b) {
            std::complex<float>* row = state + b*D;
            for (int wire = 0; wire < Q; ++wire) {
                const float* slot = trig+(b*L*Q+layer*Q+wire)*4;
                float c = slot[0], s = slot[1], cp = slot[2], sp = slot[3];
                int stride = 1 << (Q-wire-1);
                for (int outer = 0; outer < D; outer += stride*2) {
                    for (int inner = 0; inner < stride; ++inner) {
                        int ai = outer+inner, bi = ai+stride;
                        auto a = row[ai], d = row[bi];
                        auto ra = c*a-s*d, rb = s*a+c*d;
                        row[ai] = {cp*ra.real()+sp*ra.imag(), cp*ra.imag()-sp*ra.real()};
                        row[bi] = {cp*rb.real()-sp*rb.imag(), cp*rb.imag()+sp*rb.real()};
                    }
                }
            }
        }
        cblas_cgemm(RowMajor, NoTrans, NoTrans, batch, D, D, &one,
                    state, D, unitary+layer*D*D*2, D, &zero, next, D);
        std::swap(state, next);
    }
    for (int i = 0; i < batch*D; ++i) {
        float re = state[i].real(), im = state[i].imag();
        probabilities[i] = re*re+im*im;
    }
}
}
'''


def _native_library() -> tuple[Any, dict]:
    prefix = Path(sys.prefix)
    compiler = shutil.which("g++")
    if compiler is None:
        matches = list((prefix / "bin").glob("*-linux-gnu-c++"))
        compiler = str(matches[0]) if matches else None
    library = prefix / "lib/libcblas.so"
    if compiler is None or not library.exists():
        raise RuntimeError("The native QNN kernel requires a C++ compiler and libCBLAS")
    flags = ["-O3", "-march=native", "-ffp-contract=off", "-fPIC", "-shared", "-std=c++17"]
    source_sha = hashlib.sha256(_NATIVE_SOURCE.encode()).hexdigest()
    identity = source_sha + compiler + str(library) + str(flags)
    cache_dir = Path(tempfile.gettempdir()) / "ns3_frozen_qnn_cache" / hashlib.sha256(identity.encode()).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    source = cache_dir / "frozen_qnn.cc"
    binary = cache_dir / "frozen_qnn.so"
    if not binary.exists():
        source.write_text(_NATIVE_SOURCE)
        command = [compiler, *flags, str(source), str(library), "-Wl,-rpath,"+str(library.parent), "-o", str(binary)]
        subprocess.run(command, check=True, capture_output=True, text=True)
    lib = ctypes.CDLL(str(binary))
    pointer = ctypes.POINTER(ctypes.c_float)
    lib.qnn_features.argtypes = [ctypes.c_int]*4 + [pointer]*11
    lib.qnn_features.restype = None
    lib.qnn_evolve_precomputed.argtypes = [ctypes.c_int] + [pointer]*6
    lib.qnn_evolve_precomputed.restype = None
    info = {"compiler": compiler, "compile_flags": flags, "source_sha256": source_sha,
            "binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "blas_library": str(library), "compiled_at_initialization": True,
            "candidate_feature_memoization": False, "fast_math": False}
    return lib, info


class FrozenNativeQNN(FrozenNumpyQNN):
    """Compiled float32 statevector inference; preallocated workspaces, one C call.

    This object has reusable workspaces and is intentionally not thread-safe.
    Construct a separate instance per inference worker. Returned outputs are
    newly allocated and safe to retain across subsequent calls.
    """
    backend = "native_frozen_fused"

    def __init__(self, model: nn.Module):
        super().__init__(model)
        if self.q != 5 or self.layers != 3 or not self.unitaries.size:
            raise ValueError("Native QNN kernel supports the evaluated q=5,l=3 entangling QNN")
        self.lib, self.compilation = _native_library()
        self._capacity = 0
        self._constant_arrays = (self.encoder, self.encoder_bias, self.unitaries.view(np.float32),
                                 self.projection, self.bias)
        self._constant_pointers = [self._pointer(v) for v in self._constant_arrays]

    @staticmethod
    def _pointer(values: np.ndarray):
        return values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def _ensure_capacity(self, batch: int) -> None:
        if batch <= self._capacity:
            return
        self._capacity = batch
        self._z = np.empty((batch, self.dim), dtype=np.float32)
        self._state = np.empty((batch, self.dim), dtype=np.complex64)
        self._next = np.empty((batch, self.dim), dtype=np.complex64)
        self._probabilities = np.empty((batch, self.dim), dtype=np.float32)
        self._workspace_pointers = [self._pointer(v) for v in
                                   (self._z, self._state.view(np.float32), self._next.view(np.float32), self._probabilities)]

    def features_np(self, xs: np.ndarray) -> np.ndarray:
        xs = np.ascontiguousarray(xs, dtype=np.float32)
        if xs.ndim != 2 or xs.shape[1] != self.input_dim:
            raise ValueError("Expected a two-dimensional input of the model input width")
        self._ensure_capacity(xs.shape[0])
        output = np.empty((xs.shape[0], self.feature_dim), dtype=np.float32)
        self.lib.qnn_features(xs.shape[0], self.input_dim, self.feature_dim, int(self.normalized),
                              self._pointer(xs), *self._constant_pointers,
                              *self._workspace_pointers, self._pointer(output))
        return output

    __call__ = features_np


class FrozenHybridNativeQNN(FrozenNativeQNN):
    """SIMD NumPy nonlinearities/BLAS with compiled batched quantum evolution."""
    backend = "hybrid_native_frozen_fused"

    def features_np(self, xs: np.ndarray) -> np.ndarray:
        xs = np.ascontiguousarray(xs, dtype=np.float32)
        if xs.ndim != 2 or xs.shape[1] != self.input_dim:
            raise ValueError("Expected a two-dimensional input of the model input width")
        batch = xs.shape[0]
        self._ensure_capacity(batch)
        z = np.tanh(xs @ self.encoder + self.encoder_bias)
        norm = np.maximum(np.sqrt(np.sum(z*z, axis=1, keepdims=True)), np.float32(1e-12))
        amplitudes = np.ascontiguousarray(z / norm)
        half_angles = z[:, :self.layers*self.q*2] * np.float32(np.pi*.5)
        cosines, sines = np.cos(half_angles), np.sin(half_angles)
        trig = np.ascontiguousarray(np.stack((cosines[:, 0::2], sines[:, 0::2],
                                             cosines[:, 1::2], sines[:, 1::2]), axis=-1))
        probabilities = self._probabilities[:batch]
        self.lib.qnn_evolve_precomputed(batch, self._pointer(amplitudes), self._pointer(trig),
                                        self._constant_pointers[2], self._workspace_pointers[1],
                                        self._workspace_pointers[2], self._pointer(probabilities))
        projected = probabilities @ self.projection + self.bias
        if not self.normalized:
            return projected
        values = np.tanh(projected)
        norms = np.maximum(np.sqrt(np.sum(values*values, axis=1, keepdims=True)), np.float32(1e-6))
        return values / norms * np.float32(self.feature_dim**.5)

    __call__ = features_np


def frozen_backends(model: nn.Module) -> dict[str, Any]:
    """Construct all applicable, independently measurable immutable backends."""
    options = {"torchscript_frozen": FrozenTorchFeatures(model)}
    if getattr(model, "architecture", None) == "qcnn_shared_reupload":
        options["numpy_frozen_fused"] = FrozenNumpyQNN(model)
        try:
            options["native_frozen_fused"] = FrozenNativeQNN(model)
            options["hybrid_native_frozen_fused"] = FrozenHybridNativeQNN(model)
        except RuntimeError:
            pass
    return options
