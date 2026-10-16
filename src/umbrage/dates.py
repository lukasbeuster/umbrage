"""Representative dates for shade simulations: choose a few days that stand for a whole year.

For a fixed site the sun's path on a day depends only on the solar declination; the equation of time shifts
that path in clock time by up to about a quarter of an hour. Any per-pixel quantity that depends only on the sun
path (shaded hours over a day, say) is therefore a function f(declination), and a yearly mean is an integral of
f over the distribution of declination across the year. Declination dwells near the solstices, so its
distribution is close to the arcsine law, and the natural quadratures are Chebyshev ones:

- Gauss-Chebyshev (``gauss_nodes``): n declinations amp * cos((2k - 1) pi / 2n), equal weights 1 / n.
- Chebyshev-Lobatto (``lobatto_nodes``): n declinations amp * cos(k pi / (n - 1)), including both solstices
  (and the equinox for odd n), weights 1 / (2(n - 1)) at the solstices and 1 / (n - 1) elsewhere.

The textbook weights assume a symmetric arcsine law. The real year is not symmetric: the sun spends about
186 days north of the equator and 179 south (orbital eccentricity), so the yearly mean declination is about
+0.4 deg. Two data-driven weightings use the actual days of a year instead:

- ``poly_weights``: interpolatory weights on the actual days (integrate the polynomial through the nodes);
  exact for any f that is a polynomial of degree < n in declination. Use these for yearly means.
- ``interp_weights``: each day is shared between the two nodes that bracket its declination (linear in
  declination). Less accurate, but every share is in [0, 1], so a day "belongs" to its nearest nodes.

Both also return the node-by-day share matrix, so any per-day weight (heat, footfall) can be carried onto the
nodes: W_node(t) = sum over days of share(node, day) x w(day, t) / n_days.

These are geometry-only statements; whether a few dates reproduce a dense year for a given surface model is an
empirical question that should be tested on real tiles.
"""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone

import numpy as np

from .sun import SGT, declination_eot


def year_table(year: int, tz: timezone = SGT, at: time = time(12, 0)) -> list[tuple[date, float, float]]:
    """(day, declination deg, equation of time min) for every day of `year`, evaluated at local time `at`."""
    out, d = [], date(year, 1, 1)
    while d.year == year:
        decl, eot = declination_eot(datetime.combine(d, at, tzinfo=tz))
        out.append((d, decl, eot))
        d += timedelta(days=1)
    return out


def zenith_dates(year: int, lat: float, tz: timezone = SGT) -> list[date]:
    """Days on which the noon sun passes closest to the zenith (declination crosses the latitude). Empty
    outside the tropics."""
    t = year_table(year, tz)
    out = []
    for (d0, a, _), (d1, b, _) in zip(t, t[1:]):
        if (a - lat) * (b - lat) <= 0 and a != b:
            out.append(d0 if abs(a - lat) <= abs(b - lat) else d1)
    return out


def gauss_nodes(n: int, amp: float) -> tuple[list[float], list[float]]:
    """Gauss-Chebyshev declinations (deg, descending) and weights for an arcsine-distributed declination."""
    if n < 1:
        raise ValueError("n >= 1")
    dec = [amp * math.cos((2 * k - 1) * math.pi / (2 * n)) for k in range(1, n + 1)]
    return dec, [1.0 / n] * n


def lobatto_nodes(n: int, amp: float) -> tuple[list[float], list[float]]:
    """Chebyshev-Lobatto declinations (deg, descending, both solstices included) and weights."""
    if n < 2:
        raise ValueError("n >= 2")
    dec = [amp * math.cos(k * math.pi / (n - 1)) for k in range(n)]
    w = [1.0 / (n - 1)] * n
    w[0] = w[-1] = 1.0 / (2 * (n - 1))
    return [0.0 if abs(x) < 1e-12 else x for x in dec], w


def dates_for(decl: float, table: list[tuple[date, float, float]]) -> tuple[date, date]:
    """The two days of the year whose declination is closest to `decl`: one while declination is rising
    (December to June) and one while it is falling (June to December). At a solstice both are the same day."""
    decs = np.array([r[1] for r in table])
    rising = np.gradient(decs) > 0
    best = []
    for mask in (rising, ~rising):
        idx = np.flatnonzero(mask)
        best.append(table[int(idx[np.argmin(np.abs(decs[idx] - decl))])][0])
    return best[0], best[1]


def interp_weights(node_decls, day_decls) -> tuple[np.ndarray, np.ndarray]:
    """Linear-in-declination shares of each day between the bracketing nodes.

    Returns (weights per node, summing to 1; share matrix of shape (n_nodes, n_days), columns summing to 1).
    Days outside the node range go wholly to the nearest end node. Nodes with equal declination split equally."""
    nd = np.asarray(node_decls, float)
    dd = np.asarray(day_decls, float)
    order = np.argsort(nd)
    xs = nd[order]
    A = np.zeros((len(nd), len(dd)))
    uniq, inv = np.unique(xs, return_inverse=True)
    if len(uniq) == 1:
        A[:] = 1.0 / len(nd)
        return A.mean(axis=1), A
    Au = np.zeros((len(uniq), len(dd)))
    j = np.clip(np.searchsorted(uniq, dd) - 1, 0, len(uniq) - 2)
    f = np.clip((dd - uniq[j]) / (uniq[j + 1] - uniq[j]), 0.0, 1.0)
    cols = np.arange(len(dd))
    Au[j, cols] += 1.0 - f
    Au[j + 1, cols] += f
    counts = np.bincount(inv, minlength=len(uniq))
    for k, o in enumerate(order):
        A[o] = Au[inv[k]] / counts[inv[k]]
    return A.mean(axis=1), A


def poly_weights(node_decls, day_decls) -> tuple[np.ndarray, np.ndarray]:
    """Interpolatory (Lagrange) weights of the nodes on the actual days.

    Returns (weights per node, summing to 1; share matrix (n_nodes, n_days) whose columns sum to 1). Shares can
    be negative (a Lagrange basis overshoots); the weights are positive for Chebyshev-type nodes. Exact for any
    per-day quantity that is a polynomial of degree < n_nodes in declination."""
    nd = np.asarray(node_decls, float)
    dd = np.asarray(day_decls, float)
    if len(np.unique(nd)) != len(nd):
        raise ValueError("node declinations must be distinct")
    scale = max(np.abs(nd).max(), np.abs(dd).max(), 1e-9)
    x, y = nd / scale, dd / scale
    A = np.ones((len(nd), len(dd)))
    for i in range(len(nd)):
        for j in range(len(nd)):
            if i != j:
                A[i] *= (y - x[j]) / (x[i] - x[j])
    return A.mean(axis=1), A
