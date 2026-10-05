"""Time every available engine on a synthetic city. Run on the machine you care about:

    python benchmarks/bench_engines.py --size 4000

Prints seconds per km2 per timestep and agreement with the numpy reference.
"""
import argparse
import time

import numpy as np

from umbrage import cast_shadows


def city(n, seed=0):
    rng = np.random.default_rng(seed)
    z = np.zeros((n, n), np.float32)
    for _ in range(n * n // 10000):
        r, c = rng.integers(0, n - 50, 2)
        z[r:r + 40, c:c + 30] = rng.uniform(10, 150)
    ct = np.full_like(z, np.nan)
    for _ in range(n * n // 1300):
        r, c = rng.integers(0, n - 8, 2)
        blk, free = ct[r:r + 7, c:c + 7], z[r:r + 7, c:c + 7] == 0
        blk[free] = rng.uniform(6, 18)
    return z, ct, np.where(np.isnan(ct), np.nan, 0.25 * ct)


def engines():
    yield "numpy", lambda *a, **k: cast_shadows(*a, **k)
    try:
        import torch
        yield "torch-cpu", lambda *a, **k: cast_shadows(*a, device="cpu", **k)
        if torch.backends.mps.is_available():
            yield "torch-mps", lambda *a, **k: cast_shadows(*a, device="mps", **k)
        if torch.cuda.is_available():
            yield "torch-cuda", lambda *a, **k: cast_shadows(*a, device="cuda", **k)
    except ImportError:
        pass
    try:
        from umbrage.backends.solweig_rs import cast_shadows as rs
        yield "solweig-cpu", lambda *a, **k: rs(*a, gpu=False, **k)
        yield "solweig-gpu", lambda *a, **k: rs(*a, gpu=True, **k)
    except ImportError:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2000, help="grid side in pixels (1 m)")
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--az", type=float, default=100.0)
    a = ap.parse_args()
    z, ct, cb = city(a.size)
    km2 = a.size * a.size / 1e6
    ref = None
    print(f"{a.size}x{a.size} m ({km2:.1f} km2), sun az {a.az} el {a.elev}")
    for name, fn in engines():
        fn(z[:300, :300], 1.0, a.az, a.elev, ct[:300, :300], cb[:300, :300])   # warm-up
        t = time.perf_counter()
        r = fn(z, 1.0, a.az, a.elev, ct, cb)
        dt = time.perf_counter() - t
        if ref is None:
            ref = r
        agree_b = (r.building == ref.building).mean()
        both = r.building | r.vegetation
        agree_c = (both == (ref.building | ref.vegetation)).mean()
        print(f"{name:12s} {dt:7.2f} s  {dt / km2:6.2f} s/km2  building agree {agree_b:.5f}  combined agree {agree_c:.5f}")
