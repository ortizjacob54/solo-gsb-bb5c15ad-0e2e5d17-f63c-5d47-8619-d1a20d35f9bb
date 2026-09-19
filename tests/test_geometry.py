"""Numerical sanity tests for geometry.py using a known synthetic homography.

Run: python tests/test_geometry.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.geometry import (  # noqa: E402
    convex_ordered_quad,
    measure_polygon,
    measure_region,
    polygon_self_intersects,
    solve_homography,
    validate_ring_topology,
)

# World -> image: a known, well-conditioned projective transform.
# A plane seen by a pinhole camera: image ~ K (r1*X + r2*Y + t), divided by
# depth.  Chosen so the whole rectangle stays well in front of the camera.
K = np.array([[800.0, 0.0, 512.0], [0.0, 800.0, 384.0], [0.0, 0.0, 1.0]])
r1 = np.array([0.999, 0.0, -0.0499])
r2 = np.array([0.02, 1.0, 0.03])
t = np.array([0.0, 0.0, 12.0])
# P restricted to world points (x, y, 0, 1): columns r1, r2, t.
P = K @ np.column_stack([r1, r2, t])  # 3x3 mapping (x,y,1) -> image


def project(x, y):
    v = P @ np.array([x, y, 1.0])
    assert v[2] > 1.0, "test point must be in front of camera"
    return [float(v[0] / v[2]), float(v[1] / v[2])]


W, H = 100.0, 60.0
corners_world = [[0, 0], [W, 0], [W, H], [0, H]]
corners_img = [project(x, y) for x, y in corners_world]

failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        failures.append(name)


print("== homography recovery ==")
res = solve_homography(corners_img, W, H)
check("solve ok", res.ok, str(res.errors))
if res.ok:
    Hm = np.array(res.H)
    max_err = 0.0
    for (x, y), (u, v) in zip(corners_world, corners_img):
        w = Hm[2, 0] * u + Hm[2, 1] * v + Hm[2, 2]
        xx = (Hm[0, 0] * u + Hm[0, 1] * v + Hm[0, 2]) / w
        yy = (Hm[1, 0] * u + Hm[1, 1] * v + Hm[1, 2]) / w
        max_err = max(max_err, math.hypot(xx - x, yy - y))
    check("corner remap residual ~0", max_err < 1e-4, f"err={max_err:.2e}")

    # Interior polygon: rectangle [20,70] x [10,50] -> area 50*40=2000, perim 180
    poly_img = [project(20, 10), project(70, 10), project(70, 50), project(20, 50)]
    m = measure_polygon(res.H, poly_img)
    check("interior polygon valid", m.ok, str(m.errors))
    check("area == 2000", m.ok and abs(m.area - 2000.0) < 1e-4, f"area={m.area:.6f}")
    check("perimeter == 180", m.ok and abs(m.perimeter - 180.0) < 1e-4, f"p={m.perimeter:.6f}")

    # Triangle: (0,0),(100,0),(0,60) -> area 3000, perim 100+60+sqrt(13600)
    tri = [project(0, 0), project(100, 0), project(0, 60)]
    mt = measure_polygon(res.H, tri)
    expect_p = 100 + 60 + math.hypot(100, 60)
    check("triangle area == 3000", mt.ok and abs(mt.area - 3000.0) < 1e-4, f"area={mt.area:.6f}")
    check("triangle perimeter", mt.ok and abs(mt.perimeter - expect_p) < 1e-4, f"p={mt.perimeter:.6f}")

print("== invalid calibrations ==")
# duplicate points
r = solve_homography([[0, 0], [100, 0], [100, 0], [0, 60]], W, H)
check("duplicate rejected", not r.ok and "重合" in r.errors[0])
# collinear
r = solve_homography([[0, 0], [50, 0], [100, 0], [150, 0]], W, H)
check("collinear rejected", not r.ok)
# self-crossing order (bowtie)
r = solve_homography([corners_img[0], corners_img[2], corners_img[1], corners_img[3]], W, H)
check("bowtie order rejected", not r.ok)
# bad dimensions
r = solve_homography(corners_img, 0, H)
check("zero width rejected", not r.ok)

print("== polygon self-intersection ==")
bowtie = [[0, 0], [10, 10], [10, 0], [0, 10]]
si = polygon_self_intersects(bowtie)
check("bowtie detected", si[0], str(si[1]))
okquad = [[0, 0], [10, 0], [10, 10], [0, 10]]
check("square clean", not polygon_self_intersects(okquad)[0])
dupped = [[0, 0], [10, 0], [10, 10], [10, 10]]
check("duplicate vertex detected", polygon_self_intersects(dupped)[0])

if res.ok:
    # self-intersecting polygon in image plane maps to self-intersection
    # build bowtie from projected world points
    bt_img = [project(0, 0), project(100, 60), project(100, 0), project(0, 60)]
    m = measure_polygon(res.H, bt_img)
    check("self-intersecting polygon blocked", not m.ok and "自交" in m.errors[0])

print("== ring topology (holes, world coords) ==")
outer = [[0, 0], [100, 0], [100, 60], [0, 60]]
hole1 = [[20, 10], [40, 10], [40, 30], [20, 30]]
hole2 = [[60, 20], [80, 20], [80, 40], [60, 40]]
check("two clean holes ok", validate_ring_topology([outer, hole1, hole2]) == [])

outside = [[105, 65], [120, 65], [120, 80], [105, 80]]
check("outside hole rejected",
      any("没有完全位于外环内" in e for e in validate_ring_topology([outer, outside])))

crossing = [[90, 10], [110, 10], [110, 30], [90, 30]]
check("crossing hole rejected",
      any("相交或相切" in e for e in validate_ring_topology([outer, crossing])))

# vertex of the hole lies exactly on the outer edge -> tangency rejected
tangent = [[20, 0], [40, 0], [40, 20], [20, 20]]
check("tangent hole rejected",
      any("相切" in e for e in validate_ring_topology([outer, tangent])))

hb1 = [[20, 20], [50, 20], [50, 45], [20, 45]]
hb2 = [[45, 25], [70, 25], [70, 50], [45, 50]]
check("inter-hole crossing rejected",
      any("孔洞 1 与孔洞 2 相交或相切" in e for e in validate_ring_topology([outer, hb1, hb2])))

nested1 = [[20, 10], [80, 10], [80, 50], [20, 50]]
nested2 = [[40, 20], [60, 20], [60, 40], [40, 40]]
check("nested holes rejected",
      any("包含" in e for e in validate_ring_topology([outer, nested1, nested2])))

self_cross_hole = [[20, 10], [40, 30], [40, 10], [20, 30]]
check("self-crossing hole rejected",
      any("孔洞 1自交" in e for e in validate_ring_topology([outer, self_cross_hole])))

print("== region measurement with holes (perspective) ==")
if res.ok:
    # outer [10,90] x [10,50] -> gross 3200, perimeter 240
    outer_img = [project(x, y) for x, y in [(10, 10), (90, 10), (90, 50), (10, 50)]]
    # hole [30,50] x [20,40] -> area 400, perimeter 80
    hole_img = [project(x, y) for x, y in [(30, 20), (50, 20), (50, 40), (30, 40)]]
    m = measure_region(res.H, outer_img, [hole_img])
    check("region valid", m.ok, str(m.errors))
    check("net area 2800", m.ok and abs(m.area - 2800) < 1e-4, f"area={m.area}")
    check("gross area 3200", m.ok and abs(m.gross_area - 3200) < 1e-4, f"gross={m.gross_area}")
    check("hole area 400", m.ok and abs(m.hole_areas[0] - 400) < 1e-4)
    check("total perimeter 320", m.ok and abs(m.perimeter - 320) < 1e-4, f"p={m.perimeter}")

    # two holes
    hole2_img = [project(x, y) for x, y in [(60, 20), (80, 20), (80, 40), (60, 40)]]
    m2 = measure_region(res.H, outer_img, [hole_img, hole2_img])
    check("two-hole net area 2400", m2.ok and abs(m2.area - 2400) < 1e-4, f"area={m2.area}")
    check("two-hole perimeter 400", m2.ok and abs(m2.perimeter - 400) < 1e-4)

    # hole straddling the outer edge -> invalid
    bad_hole = [project(x, y) for x, y in [(80, 40), (95, 40), (95, 55), (80, 55)]]
    mb = measure_region(res.H, outer_img, [bad_hole])
    check("straddling hole blocked", not mb.ok and any("外环" in e for e in mb.errors), str(mb.errors))

    # backward-compatible plain wrapper
    mp = measure_polygon(res.H, outer_img)
    check("plain wrapper works", mp.ok and abs(mp.area - 3200) < 1e-4)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL GEOMETRY TESTS PASSED")
