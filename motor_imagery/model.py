"""The decoders, and the rule that turns probabilities into a pick.

Five models, all with the same interface - fit(trials) / predict_proba(bands) -
so evaluate.py can score them on identical folds and the speller can load
whichever won:

  tangent      8-30 Hz covariance -> tangent space -> logistic regression  (default)
  fb-tangent   the same, per sub-band, concatenated
  fbcsp        CSP log-power per sub-band -> shrinkage LDA        (their branch A)
  stack        all three branches + MRCP, stacked                 (their full ensemble)
  logvar       log band power per channel -> shrinkage LDA        (sanity check)

Why `tangent` is the default and not `stack`: with ~24 trials per class, the
stack fits five sets of spatial filters, a covariance mean, three classifiers
and a meta-learner. Tangent-space logistic regression fits 36 features and one
regularisation constant. The stack is here to be measured, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as cfg
from .features import MRCP, LogVar, csp_pipeline, tangent_pipeline

MODELS = ["tangent", "fb-tangent", "fbcsp", "stack", "logvar"]


def _logistic():
    return LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)


def _lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


class BandModel(BaseEstimator, ClassifierMixin):
    """One transformer per band, features concatenated, then one classifier.

    Each band's transformer is fitted on the training trials only, which is what
    makes fold-wise scoring honest: CSP filters and the Riemannian mean never see
    the trials they are scored on.
    """

    def __init__(self, bands: list[str], make_transformer, classifier, scale: bool = True):
        self.bands = bands
        self.make_transformer = make_transformer
        self.classifier = classifier
        self.scale = scale

    def fit(self, bands: dict[str, np.ndarray], y: np.ndarray) -> "BandModel":
        self.transformers_ = {b: self.make_transformer() for b in self.bands}
        features = [self.transformers_[b].fit_transform(bands[b], y) for b in self.bands]
        self.pipeline_ = (make_pipeline(StandardScaler(), self.classifier) if self.scale
                          else self.classifier)
        self.pipeline_.fit(np.hstack(features), y)
        self.classes_ = self.pipeline_.classes_
        return self

    def transform(self, bands: dict[str, np.ndarray]) -> np.ndarray:
        return np.hstack([self.transformers_[b].transform(bands[b]) for b in self.bands])

    def predict_proba(self, bands: dict[str, np.ndarray]) -> np.ndarray:
        return self.pipeline_.predict_proba(self.transform(bands))


class Stack(BaseEstimator, ClassifierMixin):
    """The tutorial's ensemble: three branches, then a meta-learner on their
    out-of-fold probabilities.

    Branch C only ever predicts active vs rest, so it contributes one number.
    With no rest class in the session it is dropped entirely.
    """

    def __init__(self, rate: float, folds: int = 5, use_mrcp: bool = True):
        self.rate = rate
        self.folds = folds
        self.use_mrcp = use_mrcp

    def _branches(self):
        branches = {
            "fbcsp": BandModel(cfg.SUB_BANDS, csp_pipeline, _lda()),
            "tangent": BandModel([cfg.BROADBAND], tangent_pipeline, _logistic()),
        }
        if self.use_mrcp:
            branches["mrcp"] = BandModel([cfg.MRCP_BAND], lambda: MRCP(self.rate), _logistic())
        return branches

    def _meta_row(self, branches, bands) -> np.ndarray:
        parts = [branches["fbcsp"].predict_proba(bands), branches["tangent"].predict_proba(bands)]
        if "mrcp" in branches:                      # one number: P(active) = 1 - P(rest)
            rest = list(branches["mrcp"].classes_).index(cfg.REST)
            parts.append(1.0 - branches["mrcp"].predict_proba(bands)[:, rest][:, None])
        return np.hstack(parts)

    def fit(self, bands: dict[str, np.ndarray], y: np.ndarray) -> "Stack":
        self.classes_, counts = np.unique(y, return_counts=True)
        self.use_mrcp = self.use_mrcp and cfg.REST in self.classes_
        n = len(y)

        # Out-of-fold meta rows: each branch scores only trials it never saw.
        folds = min(self.folds, int(counts.min()))
        meta_rows = None
        if folds >= 2:
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
            for train, test in splitter.split(np.zeros(n), y):
                branches = self._branches()
                for branch in branches.values():
                    branch.fit({b: x[train] for b, x in bands.items()}, y[train])
                row = self._meta_row(branches, {b: x[test] for b, x in bands.items()})
                if meta_rows is None:
                    meta_rows = np.zeros((n, row.shape[1]))
                meta_rows[test] = row

        self.branches_ = self._branches()
        for branch in self.branches_.values():
            branch.fit(bands, y)
        if meta_rows is None:                       # too few trials to hold any out
            meta_rows = self._meta_row(self.branches_, bands)
        self.meta_ = make_pipeline(StandardScaler(), _logistic()).fit(meta_rows, y)
        return self

    def predict_proba(self, bands: dict[str, np.ndarray]) -> np.ndarray:
        row = self._meta_row(self.branches_, bands)
        return self.meta_.predict_proba(row)


def build(name: str, rate: float, mode: str = "clench"):
    """One of MODELS, ready to fit."""
    if name == "tangent":
        return BandModel([cfg.BROADBAND], tangent_pipeline, _logistic())
    if name == "fb-tangent":
        return BandModel(cfg.SUB_BANDS, tangent_pipeline, _logistic())
    if name == "fbcsp":
        return BandModel(cfg.SUB_BANDS, csp_pipeline, _lda())
    if name == "logvar":
        return BandModel([cfg.BROADBAND], LogVar, _lda())
    if name == "stack":
        # MRCP measures executed movement, so it comes off for imagined runs.
        return Stack(rate, use_mrcp=(mode == "clench"))
    raise ValueError(f"unknown model {name!r} - pick one of {', '.join(MODELS)}")


# --------------------------------------------------------------------------- #
# Deciding what to type
# --------------------------------------------------------------------------- #

def decide(proba: np.ndarray, classes: np.ndarray, margin: float) -> int:
    """Class id to type, or -1 for "neither".

    A wrong letter has to be deleted; a missed pick just means trying again. So
    rest wins ties: the best hand has to beat rest by `margin` before anything is
    typed. Without a rest class the margin acts as a plain confidence floor.
    """
    classes = list(classes)
    active = [i for i, c in enumerate(classes) if c != cfg.REST]
    if not active:
        return -1
    best = max(active, key=lambda i: proba[i])
    rest = proba[classes.index(cfg.REST)] if cfg.REST in classes else 0.0
    if cfg.REST in classes:
        return classes[best] if proba[best] - rest >= margin else -1
    return classes[best] if proba[best] >= 0.5 + margin else -1


@dataclass
class MarginResult:
    margin: float
    false_pick: float        # rest trials that typed something
    accuracy: float          # left/right trials typed correctly
    missed: float            # left/right trials that typed nothing

    def __str__(self) -> str:
        return (f"margin {self.margin:.2f}: {100 * self.accuracy:.0f}% of hand trials right, "
                f"{100 * self.missed:.0f}% missed, {100 * self.false_pick:.0f}% of rest typed")


def score_margin(proba: np.ndarray, y: np.ndarray, classes: np.ndarray,
                 margin: float) -> MarginResult:
    picks = np.array([decide(p, classes, margin) for p in proba])
    is_rest = y == cfg.REST
    hands = ~is_rest
    false_pick = float((picks[is_rest] >= 0).mean()) if is_rest.any() else 0.0
    accuracy = float((picks[hands] == y[hands]).mean()) if hands.any() else 0.0
    missed = float((picks[hands] < 0).mean()) if hands.any() else 0.0
    return MarginResult(margin, false_pick, accuracy, missed)


def tune_margin(proba: np.ndarray, y: np.ndarray, classes: np.ndarray,
                max_false_pick: float = cfg.MAX_FALSE_PICK) -> MarginResult:
    """Smallest margin whose false-pick rate on rest is acceptable.

    Same idea as trca_model.fit_idle, but on probabilities instead of a TRCA
    score. Tuned on held-out predictions, never on the trials the model was fitted on.
    """
    candidates = [round(m, 3) for m in np.arange(0.0, 0.95, 0.025)]
    scored = [score_margin(proba, y, classes, m) for m in candidates]
    ok = [s for s in scored if s.false_pick <= max_false_pick]
    if ok:
        return max(ok, key=lambda s: (s.accuracy, -s.margin))
    return min(scored, key=lambda s: (s.false_pick, -s.accuracy))
