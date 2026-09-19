"""Turning a window of EEG into numbers.

Three views, following the MIND club's pipeline:

  covariance   how the 8 channels vary together in 8-30 Hz -> tangent space
  CSP          spatial filters per sub-band whose log-power splits the classes
  MRCP         slow-potential shape in 0.05-5 Hz, plus a C3/C4 laterality index

The first two come from pyriemann (their hand-rolled lw_cov / AIRM mean /
tangent_space are the same maths, and pyriemann is already a dependency, so
there is no reason to carry our own eigenvalue clipping). The MRCP features are
theirs, ported.
"""

from __future__ import annotations

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
from scipy.signal import hilbert
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

from . import config as cfg

MRCP_TAIL_SEC = 0.6      # their window: the last 0.6 s of the trial
EPS = 1e-12


def tangent_pipeline():
    """Ledoit-Wolf covariance -> tangent space at the Riemannian mean (36 features)."""
    return make_pipeline(Covariances(estimator="lwf"), TangentSpace(metric="riemann"))


def csp_pipeline(n_components: int = 4):
    """Their BandTaskCSP, minus the bug: one CSP per band, not three copies of it.

    The tutorial makes three "tasks" (L_vs_R, B_vs_LR, Act_vs_Rest) but passes
    the same four-class labels to each, so the three are identical. pyriemann's
    CSP handles the classes jointly, which is what those three were reaching for.
    """
    return make_pipeline(Covariances(estimator="lwf"), CSP(nfilter=n_components, log=True))


class LogVar(BaseEstimator, TransformerMixin):
    """Log band power per channel: the oldest motor-imagery feature there is,
    kept as a sanity check. If this beats everything else, the fancy models are
    fitting noise."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log(np.var(X, axis=-1) + EPS)


def mrcp_features(trials: np.ndarray, rate: float) -> tuple[np.ndarray, list[str]]:
    """Slow-potential features from the 0.05-5 Hz band (their feature_extraction).

    trials: (n_trials, n_channels, n_samples) -> (n_trials, 4 * n_channels + 1)

    Note these describe EXECUTED movement: mean level, drift and depth of a slow
    shift. On imagined trials they mostly describe electrode drift, which is why
    the stack defaults off in imagine mode.
    """
    n_trials, n_channels, n_samples = trials.shape
    tail = min(int(round(MRCP_TAIL_SEC * rate)), n_samples)
    start = n_samples - tail

    names = cfg.CHANNEL_NAMES[:n_channels]
    c3 = names.index("C3") if "C3" in names else None
    c4 = names.index("C4") if "C4" in names else None

    t = np.linspace(0, 1, tail)
    t = t - t.mean()
    denom = float(np.sum(t * t)) + EPS

    out = np.empty((n_trials, 4 * n_channels + 1))
    for i in range(n_trials):
        segment = trials[i, :, start:]
        laterality = 0.0
        if c3 is not None and c4 is not None:
            env_c3 = np.abs(hilbert(trials[i, c3]))[start:].mean()
            env_c4 = np.abs(hilbert(trials[i, c4]))[start:].mean()
            laterality = (env_c3 - env_c4) / (env_c3 + env_c4 + EPS)
        out[i] = np.concatenate([segment.mean(axis=1),          # level
                                 (segment @ t) / denom,          # slope
                                 segment.min(axis=1),            # depth
                                 # Their "area" is sum/n, i.e. the mean again. Kept so the
                                 # ported branch matches theirs; it adds nothing.
                                 segment.mean(axis=1),
                                 [laterality]])
    labels = ([f"mrcp_mean_{n}" for n in names] + [f"mrcp_slope_{n}" for n in names]
              + [f"mrcp_min_{n}" for n in names] + [f"mrcp_area_{n}" for n in names]
              + ["mrcp_LI_C3C4"])
    return out, labels


class MRCP(BaseEstimator, TransformerMixin):
    """mrcp_features as a transformer, so it can sit in a pipeline."""

    def __init__(self, rate: float = 125.0):
        self.rate = rate

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return mrcp_features(X, self.rate)[0]
