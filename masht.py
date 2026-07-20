"""Masht: HYDRA + MultiRocket features classified by TabPFN.

SPDX-License-Identifier: GPL-3.0-only

The classifier exposes a reusable, scikit-learn-compatible estimator interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from tabpfn import TabPFNClassifier

from masht_vendor.hydra.multivariate import HydraMultivariate
from masht_vendor.hydra.univariate import Hydra, SparseScaler
from masht_vendor.multirocket import multivariate as multirocket_multivariate
from masht_vendor.multirocket import univariate as multirocket_univariate


def _adaptive_feature_budget(num_samples: int) -> int:
    if num_samples < 1_000:
        return 10_000
    if num_samples < 100_000:
        return 2_000
    return 200


class Masht(ClassifierMixin, BaseEstimator):
    """Time-series classifier combining HYDRA, MultiRocket, and TabPFN.

    Parameters
    ----------
    multivariate:
        If false, accept ``(samples, timepoints)`` (or a singleton channel
        dimension). If true, expect ``(samples, channels, timepoints)`` and use
        the experimental multivariate HYDRA and MultiRocket implementations.
    max_features:
        Total feature budget before the budget is split equally between HYDRA
        and MultiRocket. If omitted, use the adaptive budget from the presented
        evaluation method.
    total_samples:
        Sample count used by the adaptive budget. Set this to
        ``len(X_train) + len(X_test)`` to reproduce the evaluation protocol.
        By default only the training count is used, as required by a normal
        fitted estimator.
    device:
        Device passed to TabPFN. ``"auto"`` is portable; use ``"cuda"`` to
        reproduce the presented GPU configuration.
    n_estimators, n_preprocessing_jobs:
        TabPFN inference settings used by the presented method.
    tabpfn_kwargs:
        Additional keyword arguments for ``TabPFNClassifier``. These override
        Masht's defaults.
    """

    def __init__(
        self,
        multivariate: bool = False,
        max_features: int | None = None,
        total_samples: int | None = None,
        device: str = "auto",
        n_estimators: int = 8,
        n_preprocessing_jobs: int = 8,
        tabpfn_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.multivariate = multivariate
        self.max_features = max_features
        self.total_samples = total_samples
        self.device = device
        self.n_estimators = n_estimators
        self.n_preprocessing_jobs = n_preprocessing_jobs
        self.tabpfn_kwargs = tabpfn_kwargs

    def _validate_X(self, X: Any) -> np.ndarray:
        X_array = np.asarray(X)
        if self.multivariate:
            if X_array.ndim != 3:
                raise ValueError(
                    "Multivariate input must have shape "
                    "(samples, channels, timepoints)."
                )
        else:
            if X_array.ndim == 3 and X_array.shape[1] == 1:
                X_array = X_array[:, 0, :]
            if X_array.ndim != 2:
                raise ValueError(
                    "Univariate input must have shape (samples, timepoints) "
                    "or (samples, 1, timepoints)."
                )

        if X_array.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if X_array.shape[-1] < 10:
            raise ValueError("Masht requires time series with at least 10 timepoints.")
        if not np.isfinite(X_array).all():
            raise ValueError("Masht does not accept NaN or infinite values.")
        return X_array

    def _resolve_budget(self, num_training_samples: int) -> int:
        sample_count = (
            self.total_samples
            if self.total_samples is not None
            else num_training_samples
        )
        budget = (
            self.max_features
            if self.max_features is not None
            else _adaptive_feature_budget(sample_count)
        )
        return int(budget)

    @staticmethod
    def _hydra_groups(input_length: int, component_budget: int) -> int:
        num_dilations = np.ceil(np.log2((input_length - 1) / (9 - 1)))
        groups = int(np.floor(component_budget / (num_dilations * 8)) / 2)
        if groups < 1:
            raise ValueError(
                "The selected feature budget is too small for this series length."
            )
        return groups

    def _fit_hydra(self, X: np.ndarray) -> np.ndarray:
        X_tensor = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        if not self.multivariate:
            X_tensor = X_tensor.unsqueeze(1)

        groups = self._hydra_groups(X.shape[-1], self.component_budget_)
        if self.multivariate:
            self.hydra_ = HydraMultivariate(
                X.shape[-1], X.shape[1], k=8, g=groups
            )
        else:
            self.hydra_ = Hydra(X.shape[-1], k=8, g=groups)

        with torch.no_grad():
            features = self.hydra_.batch(X_tensor)
        self.hydra_scaler_ = SparseScaler()
        return self.hydra_scaler_.fit_transform(features).numpy()

    def _transform_hydra(self, X: np.ndarray) -> np.ndarray:
        X_tensor = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        if not self.multivariate:
            X_tensor = X_tensor.unsqueeze(1)
        with torch.no_grad():
            features = self.hydra_.batch(X_tensor)
        return self.hydra_scaler_.transform(features).numpy()

    def _fit_multirocket(self, X: np.ndarray) -> np.ndarray:
        module = (
            multirocket_multivariate if self.multivariate else multirocket_univariate
        )
        X64 = np.ascontiguousarray(X, dtype=np.float64)
        diff = np.ascontiguousarray(np.diff(X64, axis=-1))
        num_kernels = int((self.component_budget_ / 2) / 4)
        self.multirocket_base_parameters_ = module.fit(
            X64, num_features=num_kernels
        )
        self.multirocket_diff_parameters_ = module.fit(
            diff, num_features=num_kernels
        )
        return module.transform(
            X64,
            diff,
            self.multirocket_base_parameters_,
            self.multirocket_diff_parameters_,
            4,
        )

    def _transform_multirocket(self, X: np.ndarray) -> np.ndarray:
        module = (
            multirocket_multivariate if self.multivariate else multirocket_univariate
        )
        X64 = np.ascontiguousarray(X, dtype=np.float64)
        diff = np.ascontiguousarray(np.diff(X64, axis=-1))
        return module.transform(
            X64,
            diff,
            self.multirocket_base_parameters_,
            self.multirocket_diff_parameters_,
            4,
        )

    def fit(self, X: Any, y: Any) -> "Masht":
        """Fit the feature transforms and TabPFN classifier."""
        X_array = self._validate_X(X)
        y_array = np.asarray(y)
        if y_array.ndim != 1 or len(y_array) != len(X_array):
            raise ValueError("y must be one-dimensional with one label per sample.")

        self.n_features_in_ = X_array.shape[-1]
        self.input_shape_ = X_array.shape[1:]
        self.feature_budget_ = self._resolve_budget(len(X_array))
        self.component_budget_ = self.feature_budget_ // 2

        hydra_features = self._fit_hydra(X_array)
        multirocket_features = self._fit_multirocket(X_array)
        features = np.concatenate((hydra_features, multirocket_features), axis=1)

        self.label_encoder_ = LabelEncoder()
        encoded_y = self.label_encoder_.fit_transform(y_array)
        self.classes_ = self.label_encoder_.classes_

        classifier_options: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "auto_scale_n_estimators": True,
            "fit_mode": "low_memory",
            "device": self.device,
            "inference_precision": "auto",
            "n_preprocessing_jobs": self.n_preprocessing_jobs,
            "tuning_config": None,
            "show_progress_bar": False,
            "ignore_pretraining_limits": True,
        }
        if self.tabpfn_kwargs:
            classifier_options.update(self.tabpfn_kwargs)
        self.classifier_ = TabPFNClassifier(**classifier_options)
        self.classifier_.fit(features, encoded_y)
        self.n_transformed_features_ = features.shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Return the concatenated HYDRA and MultiRocket representation."""
        check_is_fitted(self, ("hydra_", "classifier_"))
        X_array = self._validate_X(X)
        if X_array.shape[1:] != self.input_shape_:
            raise ValueError(
                f"Expected per-sample shape {self.input_shape_}, got "
                f"{X_array.shape[1:]}."
            )
        return np.concatenate(
            (self._transform_hydra(X_array), self._transform_multirocket(X_array)),
            axis=1,
        )

    def predict(self, X: Any) -> np.ndarray:
        """Predict labels in the label space supplied to ``fit``."""
        encoded = self.classifier_.predict(self.transform(X)).astype(int)
        return self.label_encoder_.inverse_transform(encoded)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities ordered according to ``classes_``."""
        return self.classifier_.predict_proba(self.transform(X))
