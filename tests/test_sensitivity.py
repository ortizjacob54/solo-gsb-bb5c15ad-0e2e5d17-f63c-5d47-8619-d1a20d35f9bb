"""Unit tests for the calibration-sensitivity perturbation engine.

Run: python tests/test_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.geometry import (  # noqa: E402
    SENSITIVITY_MIN_COVERAGE,
    SENSITIVITY_SAMPLES,
    _SplitMix64,
    assess_region_sensitivity,
    run_sensitivity,
    sensitivity_seed,
    solve_homography,
)

failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        failures.append(name)


# Deterministic synthetic camera (same setup as test_geometry.py).
K = np.array([[800.0, 0.0, 512.0], [0.0, 800.0, 384.0], [0.0, 0.0, 1.0]])
r1 = np.array([0.999, 0.0, -0.0499])
r2 = np.array([0.02, 1.0, 0.03])
P = K @ np.column_stack([r1, r2, [0.0, 0.0, 12.0]])


def project(x, y):
    v = P @ np.array([x, y, 1.0])
    return [float(v[0] / v[2]), float(v[1] / v[2])]


W, H = 100.0, 60.0
corners = [project(x, y) for x, y in [[0, 0], [W, 0], [W, H], [0, H]]]
outer = [project(x, y) for x, y in [(20, 10), (70, 10), (70, 50), (20, 50)]]
hole = [project(x, y) for x, y in [(30, 20), (50, 20), (50, 40), (30, 40)]]

print("== PRNG determinism ==")
rng_a, rng_b, rng_c = _SplitMix64(42), _SplitMix64(42), _SplitMix64(43)
a = [rng_a.next_u64() for _ in range(8)]
b = [rng_b.next_u64() for _ in range(8)]
c = [rng_c.next_u64() for _ in range(8)]
check("same seed -> same stream", a == b)
check("different seed -> different stream", a != c)
check("values are 64-bit", all(0 <= v < 2 ** 64 for v in a))
rng_u = _SplitMix64(7)
u = [rng_u.next_unit() for _ in range(20000)]
check("unit draws in [0,1)", all(0.0 <= x < 1.0 for x in u))
check("unit mean ~0.5", abs(sum(u) / len(u) - 0.5) < 0.02)
check("unit draws cover the range", min(u) < 0.02 and max(u) > 0.98)

print("== seed stability ==")
s1 = sensitivity_seed(corners, W, H, 2.0)
s2 = sensitivity_seed(corners, W, H, 2.0)
check("seed is deterministic", s1 == s2 and len(s1) == 64)
check("radius changes seed", sensitivity_seed(corners, W, H, 3.0) != s1)
check("corners change seed",
      sensitivity_seed([[x + 0.5, y] for x, y in corners], W, H, 2.0) != s1)
check("dims change seed", sensitivity_seed(corners, W + 1, H, 2.0) != s1)

print("== well-conditioned assessment (radius 2 px) ==")
res = run_sensitivity(corners, W, H, 2.0, [(outer, [])])
check("128 trials", res["samples"] == SENSITIVITY_SAMPLES == 128)
check("no failures", res["out_of_bounds_samples"] == 0 and res["degenerate_samples"] == 0)
check("homography tally", res["homography_valid_samples"] == 128)
rs = res["regions"][0]
check("coverage 100%", rs.valid_samples == 128 and rs.invalid_geometry_samples == 0)
check("area ordering", rs.area_p5 <= rs.area_p50 <= rs.area_p95,
      f"{rs.area_p5},{rs.area_p50},{rs.area_p95}")
check("perimeter ordering", rs.perimeter_p5 <= rs.perimeter_p50 <= rs.perimeter_p95)
check("P50 close to nominal area 2000", abs(rs.area_p50 - 2000.0) / 2000.0 < 0.02,
      f"p50={rs.area_p50}")
check("P50 close to nominal perimeter 180", abs(rs.perimeter_p50 - 180.0) / 180.0 < 0.02,
      f"p50={rs.perimeter_p50}")
check("perturbation actually spreads area", rs.area_p95 - rs.area_p5 > 1e-6)

print("== radius 0 collapses to nominal ==")
r0 = run_sensitivity(corners, W, H, 0.0, [(outer, [hole])])
rs0 = r0["regions"][0]
check("all valid at r=0", rs0.valid_samples == 128)
check("area band collapsed", abs(rs0.area_p5 - rs0.area_p95) < 1e-7)
check("net area 1600 at r=0", abs(rs0.area_p50 - 1600.0) < 1e-6, f"{rs0.area_p50}")

print("== reproducibility across calls ==")
res_b = run_sensitivity(corners, W, H, 2.0, [(outer, [])])
rs_b = res_b["regions"][0]
check("same seed", res_b["seed"] == res["seed"])
check("same percentiles",
      rs_b.area_p5 == rs.area_p5 and rs_b.perimeter_p95 == rs.perimeter_p95)

# A different set of stored regions must not alter the homography trial stream
# (the seed depends only on corners/dims/radius), so area bands for the same
# region are identical regardless of what else is in the session.
res_hole = run_sensitivity(corners, W, H, 2.0, [(outer, [hole]), (outer, [])])
check("seed independent of regions", res_hole["seed"] == res["seed"])
check("region bands independent of siblings",
      res_hole["regions"][1].area_p5 == rs.area_p5)
check("holed region measured", res_hole["regions"][0].valid_samples == 128
      and res_hole["regions"][0].area_p50 < rs.area_p50)

print("== stress: large radius drives failure buckets ==")
big = run_sensitivity(corners, W, H, 9000.0, [(outer, [])])
check("buckets sum to 128",
      big["out_of_bounds_samples"] + big["degenerate_samples"]
      + big["homography_valid_samples"] == 128)
check("large perturbation degenerates some homographies",
      big["degenerate_samples"] > 0, f"deg={big['degenerate_samples']}")
brs = big["regions"][0]
check("region failures never exceed homography successes",
      brs.invalid_geometry_samples <= big["homography_valid_samples"])
check("valid + invalid-region <= total",
      brs.valid_samples + brs.invalid_geometry_samples <= 128)

# Out-of-bounds bucket: with photo bounds, a corner sitting 2 px inside the
# edge crosses the border as soon as the radius exceeds that margin.
tight_corners = [[2.0, 2.0], [198.0, 2.0], [198.0, 148.0], [2.0, 148.0]]
oob = run_sensitivity(tight_corners, W, H, 10.0, [([[40, 30], [160, 30], [160, 120], [40, 120]], [])],
                      img_w=200, img_h=150)
check("out-of-bounds trials counted", oob["out_of_bounds_samples"] > 0,
      f"oob={oob['out_of_bounds_samples']}")
check("oob tally still 128",
      oob["out_of_bounds_samples"] + oob["degenerate_samples"]
      + oob["homography_valid_samples"] == 128)

print("= assess helper tolerates all-failed homographies ==")
all_none = assess_region_sensitivity([None] * 128, outer, [])
check("zero valid -> None percentiles",
      all_none.valid_samples == 0 and all_none.area_p5 is None
      and all_none.perimeter_p95 is None and all_none.invalid_geometry_samples == 0)

# A geometrically illegal region (hole crossing the outer edge at nominal H)
# must count every successful-H trial as an invalid-geometry failure.
Hm = solve_homography(corners, W, H).H
bad_hole = [project(x, y) for x, y in [(80, 40), (95, 40), (95, 55), (80, 55)]]
Hs = [Hm] * 128
bad = assess_region_sensitivity(Hs, outer, [bad_hole])
check("illegal region counted, not dropped",
      bad.valid_samples == 0 and bad.invalid_geometry_samples == 128
      and bad.area_p5 is None)
check("coverage threshold is 80%", SENSITIVITY_MIN_COVERAGE == 0.8)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL SENSITIVITY TESTS PASSED")
