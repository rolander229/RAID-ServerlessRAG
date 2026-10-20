"""The ``<X, s, p> -> <m, t>`` predictors of Sec. II-B, scored by RMSLE.

What counts as one prediction model decides the score completely, so this module
says which one it is using. "MOPAR establishes a prediction model to characterize
the relationship between <X,s,p> and <mi,ti> **for each dominant operator**" is
read here the way it has to be for the paper's errors to make sense: 118 operators
called 1600 times is 13.6 invocations each, i.e. one operator seen at several
input sizes, and an RMSLE of 0.0319 for Conv2D is what that looks like when a
convolution's footprint is nearly proportional to the pixel count. Within such a
group ``X`` and ``p`` are constants, so the predictor only ever sees ``s``.

:func:`cross_fit` answers the other question -- one model for a whole population
of operators, with entire (service, layer) groups held out so nothing is predicted
from its own measurement -- and it is an order of magnitude worse on the same
rows. That is not a contradiction with the paper but the cost of its thin feature
set: ``<X, s, p>`` cannot tell a 3x3/64 convolution from a 3x3/128 one at the same
input size, since they differ only in ``p``.

RMSLE is the metric the paper chooses because it "imposes a heavier penalty on
underestimation than overestimation": a shortfall is further from the target in
log terms than the same proportional surplus, and OOM is a hard failure while a
wide quota is only wasted money. Three details of the fit are ours, not the
paper's, and all are reported rather than buried: rows whose target is zero are
dropped (an in-place operator has no bytes to be wrong about, and scoring it as a
free hit would pad every column), :func:`per_operator` offers both the paper's
random row split, where a validation size is also a training size, and a stricter
split that holds whole sizes out -- the difference between the two is how much of
a tree predictor's score is interpolation -- and the memory tables are fitted
twice, once on the byte model of Eq. (1)-(3) and once on the allocator's own
reading of the same layer. Only the second is a measurement, and :func:`rmsle`
against the first is 0 for a linear fit by construction: a convolution's footprint
is width times height times channels, so a predictor given the request size is
being asked to reproduce an arithmetic identity.
"""

from __future__ import annotations

import zlib
from collections import defaultdict
from typing import Any, Callable, Sequence

import numpy as np

KINDS = ("lr", "xgboost", "rf")
MIN_TRAIN = 3


def _regressor(kind: str, workers: int = 1):
    """A fresh predictor.

    ``workers`` defaults to one because :func:`per_operator` builds hundreds of
    models over a handful of rows each, and the thread pool then costs more than
    the fit -- a per-group RandomForest at n_jobs=8 measured 30x slower than the
    same forest serialised.
    """
    if kind == "lr":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, min_samples_leaf=1,
                                     n_jobs=workers, random_state=0)
    if kind == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1,
                            subsample=0.9, colsample_bytree=0.9, n_jobs=workers,
                            random_state=0, tree_method="hist")
    raise ValueError("unknown predictor %r; choose from %s" % (kind, ", ".join(KINDS)))


def rmsle(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Root mean square log error; a negative prediction is clipped to zero."""
    if len(actual) == 0:
        return float("nan")
    terms = (np.log1p(np.clip(predicted, 0.0, None))
             - np.log1p(np.clip(actual, 0.0, None))) ** 2
    return float(np.sqrt(np.mean(terms)))


def accuracy(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """1 - MAPE over the nonzero targets; the paper reports ">95%" with no formula."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.clip(np.asarray(predicted, dtype=np.float64), 0.0, None)
    mask = actual > 0
    if not mask.any():
        return float("nan")
    return float(max(0.0, 1.0 - np.mean(np.abs(predicted[mask] - actual[mask])
                                        / actual[mask])))


def target(rows: Sequence[Any], name: str) -> np.ndarray:
    return np.asarray([float(getattr(row, name)) for row in rows], dtype=np.float64)


def design(rows: Sequence[Any], columns: Sequence[str],
           models: Sequence[str] = ()) -> np.ndarray:
    """Numeric features, then ``X`` as a one-hot block where a fit needs one.

    One-hot rather than the model's name as an integer: all three predictors take
    numbers, and an ordinal encoding would let a tree conclude that BERT-3.0B
    lies between BERT-1.3B and DisBERT.
    """
    index = {name: position for position, name in enumerate(models)}
    out = []
    for row in rows:
        vector = [float(getattr(row, name, 0.0)) for name in columns]
        vector += [1.0 if position == index[row.model] else 0.0
                   for position in range(len(models))]
        out.append(vector)
    return np.asarray(out, dtype=np.float64)


def group(rows: Sequence[Any], by: Callable[[Any], tuple]) -> dict[tuple, list[Any]]:
    grouped: dict[tuple, list[Any]] = defaultdict(list)
    for row in rows:
        grouped[by(row)].append(row)
    return grouped


def hold_out_sizes(rows: Sequence[Any], fraction: float = 0.75,
                   seed: int = 0) -> tuple[list, list]:
    """Split on input size, so a held-out size is never also a training size."""
    rng = np.random.default_rng(seed)
    sizes = sorted({row.size for row in rows})
    rng.shuffle(sizes)
    held_out = set(sizes[int(len(sizes) * fraction):])
    return ([r for r in rows if r.size not in held_out],
            [r for r in rows if r.size in held_out])


def hold_out_rows(rows: Sequence[Any], fraction: float = 0.75,
                  seed: int = 0) -> tuple[list, list]:
    """The paper's split: 1200 of 1600 calls chosen at random.

    Every input size then appears on both sides, which is interpolation and is
    what a caller who re-measures the same operator at a known batch can expect.
    """
    rng = np.random.default_rng(seed)
    order = np.arange(len(rows))
    rng.shuffle(order)
    cut = int(round(len(rows) * fraction))
    train = [rows[i] for i in order[:cut]]
    test = [rows[i] for i in order[cut:]]
    return train, test


def _seed(key: tuple) -> int:
    """A per-group seed.

    One seed for the whole table would make every group hold out the same sizes,
    since all groups are measured at the same batches; the trees would then always
    be extrapolating past the largest training size, and the split would be one
    global decision wearing a per-operator costume.
    """
    return zlib.crc32(repr(key).encode()) % (2 ** 31)


def _empty(kinds: Sequence[str]) -> dict[str, dict[str, list]]:
    return {kind: {"actual": [], "predicted": []} for kind in kinds}


def _summarise(scored: dict, groups: int, unused: int, zero_rows: int) -> dict[str, Any]:
    out: dict[str, Any] = {"groups": groups, "unused_groups": unused,
                           "zero_rows": zero_rows}
    for kind, collected in scored.items():
        actual, predicted = collected["actual"], collected["predicted"]
        if not actual:
            out[kind] = {"insufficient": True}
            continue
        out[kind] = {"n": len(actual), "rmsle": rmsle(actual, predicted),
                     "accuracy": accuracy(actual, predicted),
                     "underestimated": float(np.mean(np.asarray(predicted)
                                                     < np.asarray(actual)))}
    return out


def _usable(rows: Sequence[Any], name: str) -> list:
    """Drop rows the target cannot score; see the module docstring.

    Two kinds go: a row with no reading at all, which is every peak row of a CPU
    profile, and a row whose target is zero.
    """
    kept = []
    for row in rows:
        value = getattr(row, name, None)
        if value is not None and value > 0:
            kept.append(row)
    return kept


def model_error(rows: Sequence[Any], predicted: str, measured: str) -> dict[str, Any]:
    """One column read as a prediction of another, with nothing fitted.

    ``memory`` is what Eq. (1)-(3) derive from shapes; ``peak`` is what the
    allocator was made to provide while the same layer ran. Scoring one against
    the other is the statement the reproduction can actually make about its memory
    vector, and it is the reason the two are not interchangeable as a regression
    target: a predictor that is handed ``s`` and asked for a byte count that is a
    formula in ``s`` scores zero error, which is arithmetic rather than learning.
    """
    kept = [row for row in rows
            if getattr(row, measured, None) is not None and getattr(row, measured) > 0]
    out = {"n": len(kept), "rmsle": rmsle([getattr(r, measured) for r in kept],
                                          [getattr(r, predicted) for r in kept])}
    if kept:
        out["underestimated"] = float(np.mean(
            [getattr(r, predicted) < getattr(r, measured) for r in kept]))
    return out


def per_operator(rows: Sequence[Any], name: str, by: Callable[[Any], tuple],
                 kinds: Sequence[str] = KINDS,
                 columns: Sequence[str] = ("size",),
                 split: str = "rows") -> dict[str, Any]:
    """The paper's protocol: one predictor per dominant operator, varied by ``s``."""
    held_out = {"rows": hold_out_rows, "size": hold_out_sizes}[split]
    scored = _empty(kinds)
    used = unused = zero_rows = 0
    for key, members in group(rows, by).items():
        kept = _usable(members, name)
        zero_rows += len(members) - len(kept)
        train, test = held_out(kept, seed=_seed(key))
        if len(train) < MIN_TRAIN or not test:
            unused += 1
            continue
        used += 1
        x_train, x_test = design(train, columns), design(test, columns)
        y_train = target(train, name)
        for kind in kinds:
            predictor = _regressor(kind)
            predictor.fit(x_train, y_train)
            scored[kind]["actual"] += list(target(test, name))
            scored[kind]["predicted"] += list(predictor.predict(x_test))
    return _summarise(scored, used, unused, zero_rows)


def cross_fit(rows: Sequence[Any], name: str, by: Callable[[Any], tuple],
              kinds: Sequence[str] = KINDS,
              columns: Sequence[str] = ("size", "params"),
              fraction: float = 0.75) -> dict[str, Any]:
    """One model over a whole population, holding out entire groups."""
    rows = _usable(rows, name)
    groups = group(rows, by)
    keys = sorted(groups)
    cut = int(len(keys) * fraction)
    train = [row for key in keys[:cut] for row in groups[key]]
    test = [row for key in keys[cut:] for row in groups[key]]
    scored = _empty(kinds)
    if len(train) < MIN_TRAIN or not test:
        return _summarise(scored, cut, len(keys) - cut, 0)
    models = sorted({row.model for row in rows})
    y_train, y_test = target(train, name), target(test, name)
    x_train, x_test = design(train, columns, models), design(test, columns, models)
    for kind in kinds:
        predictor = _regressor(kind, workers=8)
        predictor.fit(x_train, y_train)
        scored[kind]["actual"] += list(y_test)
        scored[kind]["predicted"] += list(predictor.predict(x_test))
    return _summarise(scored, cut, len(keys) - cut, 0)


def operator_key(row: Any) -> tuple:
    """A dominant operator: one service, one layer, one operator family."""
    return (row.model, row.layer, row.family)


def layer_key(row: Any) -> tuple:
    return (row.model, row.layer)


def evaluate(profiles: Sequence[Any], min_rows: int = 12) -> dict[str, Any]:
    """Table I (layer level) and Fig. 5a/5b (operator level) from measured profiles.

    Every memory figure comes out twice, and the difference between the two
    columns is a result rather than a detail: ``memory`` fits the byte model,
    which is an exact function of the request size and so learns nothing, and
    ``memory_measured`` fits what the allocator actually rose to, which is what
    the paper's 0.105 was measuring.
    """
    layer_rows = [row for profile in profiles for row in profile.layers]
    op_rows = [row for profile in profiles for row in profile.operators]
    families: dict[str, list] = defaultdict(list)
    for row in op_rows:
        families[row.family].append(row)

    result: dict[str, Any] = {
        "layer_rows": len(layer_rows),
        "operator_rows": len(op_rows),
        "layer": {},
        "by_family": {},
    }
    for table, key in (("memory", "memory"), ("peak", "memory_measured"),
                       ("ns", "time")):
        result["layer"][key] = per_operator(layer_rows, table, layer_key)
        result["layer"][key + "_size"] = per_operator(layer_rows, table, layer_key,
                                                     split="size")
        result["layer"][key + "_cross"] = cross_fit(layer_rows, table, layer_key)
    result["layer"]["byte_model"] = {
        "layer": model_error(layer_rows, "memory", "peak"),
        "operators": model_error(op_rows, "memory", "peak"),
    }
    for family, rows in sorted(families.items()):
        entry = {"n": len(rows)}
        if len(rows) >= min_rows:
            entry["memory"] = per_operator(rows, "memory", operator_key)
            entry["memory_size"] = per_operator(rows, "memory", operator_key,
                                               split="size")
            entry["memory_cross"] = cross_fit(rows, "memory", operator_key)
            entry["memory_measured"] = per_operator(rows, "peak", operator_key)
            entry["byte_model"] = model_error(rows, "memory", "peak")
            entry["time"] = per_operator(rows, "ns", operator_key)
        else:
            entry["insufficient"] = True
        result["by_family"][family] = entry
    return result

