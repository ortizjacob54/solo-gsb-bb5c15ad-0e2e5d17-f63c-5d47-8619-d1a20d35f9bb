"""Planar homography / measurement geometry.

The calibration rectangle is mapped from *image pixel coordinates* (u, v)
into the real-world plane (x, y) of a known rectangle of width W and height H.
The four corners must be supplied in order:

    0 = top-left  -> (0, 0)
    1 = top-right -> (W, 0)
    2 = bottom-right -> (W, H)
    3 = bottom-left  -> (0, H)

The 3x3 homography H is estimated with the normalized 8-point DLT (we have
exactly 4 point pairs => 8 equations) and validated for invertibility before
use.  The same numerical tolerances are mirrored by the frontend JavaScript
(frontend/app.js) so live results and the authoritative backend results agree.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Tunables (keep in sync with frontend/app.js)
# ---------------------------------------------------------------------------

DUP_TOL = 1e-7          # relative duplicate-point tolerance
COLINEAR_TOL = 1e-9     # |cross| / edge-length ratio for collinearity
MIN_PIVOT = 1e-9        # least-squares pivot threshold (singular DLT)
RESIDUAL_TOL = 1e-7     # max normalized DLT residual
H_DET_TOL = 1e-8        # |det H| relative to |H|^3
DENOM_TOL = 1e-9        # |w| / |H row 3| for mapping through infinity
RING_TOL = 1e-9         # ring separation / tangency tolerance (relative to size)

_EPS = 1e-12


def _point(p) -> np.ndarray:
    return np.asarray(p, dtype=float).reshape(2)


# ---------------------------------------------------------------------------
# Elementary checks
# ---------------------------------------------------------------------------

def _duplicate(points: list[list[float]]) -> tuple[int, int, float] | None:
    """Return (i, j, min_len) for the first coincident point pair, else None."""
    pts = [_point(p) for p in points]
    diag = max(
        (float(np.hypot(*(pts[i] - pts[j]))) for i in range(len(pts)) for j in range(i + 1, len(pts))),
        default=0.0,
    )
    ref = max(diag, 1.0)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if float(np.hypot(*(pts[i] - pts[j]))) < DUP_TOL * ref:
                return i, j, diag
    return None


def convex_ordered_quad(points: list[list[float]]) -> tuple[bool, str]:
    """True if the 4 points form a strictly convex quadrilateral in order.

    A concave or self-crossing ordering makes the projective map degenerate
    (one triangle folds over), so it is rejected with a readable reason.
    """
    dup = _duplicate(points)
    if dup is not None:
        return False, (
            f"角点 {dup[0] + 1} 与角点 {dup[1] + 1} 重合：四点必须彼此分离"
        )
    pts = [_point(p) for p in points]
    crosses = []
    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        c = pts[(i + 2) % 4]
        cr = float((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        crosses.append(cr)
    scale = max(
        float(np.hypot(*(pts[(i + 1) % 4] - pts[i]))) for i in range(4)
    ) ** 2
    if any(abs(cr) < COLINEAR_TOL * max(scale, 1.0) for cr in crosses):
        return False, "有三个或更多角点近似共线，四边形已退化"
    if not all(cr > 0 for cr in crosses) and not all(cr < 0 for cr in crosses):
        return False, "四个角点没有构成凸四边形（顺序错乱或自交），请按 左上→右上→右下→左下 依次点击"
    return True, ""


# ---------------------------------------------------------------------------
# Homography (DLT)
# ---------------------------------------------------------------------------

@dataclass
class HomographyResult:
    ok: bool
    H: list[list[float]] | None = None
    errors: list[str] | None = None


def solve_homography(
    image_points: list[list[float]],
    width: float,
    height: float,
) -> HomographyResult:
    """Solve image -> world homography for the calibration rectangle."""
    errors: list[str] = []
    if len(image_points) != 4:
        return HomographyResult(False, None, ["需要恰好 4 个角点"])
    if not (np.isfinite(width) and np.isfinite(height)) or width <= 0 or height <= 0:
        errors.append("已知矩形的宽和高必须为正数")
    convex, reason = convex_ordered_quad(image_points)
    if not convex:
        errors.append(reason)
    if errors:
        return HomographyResult(False, None, errors)

    world_points = [
        [0.0, 0.0],
        [float(width), 0.0],
        [float(width), float(height)],
        [0.0, float(height)],
    ]

    # Normalized DLT: isotropic scaling of both point sets for stability.
    src = np.asarray(image_points, dtype=float)
    dst = np.asarray(world_points, dtype=float)
    Ts, sn = _normalizer(src)
    Td, dn = _normalizer(dst)
    ns = (Ts @ np.vstack([src.T, np.ones((1, 4))]))[:2].T
    nd = (Td @ np.vstack([dst.T, np.ones((1, 4))]))[:2].T

    A = np.zeros((8, 9))
    for i in range(4):
        x, y = ns[i]
        X, Y = nd[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, X * x, X * y, X]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, Y * x, Y * y, Y]

    try:
        _, svals, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return HomographyResult(False, None, ["透视矩阵求解失败：方程组数值不稳定（SVD 未收敛）"])
    s_min = svals[-1]
    if s_min < MIN_PIVOT * max(svals[0], sn * dn, 1.0):
        return HomographyResult(False, None, [
            "透视变换不可逆：标定四点共线或构成退化四边形，无法从平面照片恢复唯一映射"
        ])

    h = Vt[-1].reshape(3, 3)
    Hn = np.linalg.solve(Td, h @ Ts)
    H = Hn / Hn[2, 2]

    # DLT residual: the 4 correspondences must be satisfied.
    res = 0.0
    for (u, v), (X, Y) in zip(image_points, world_points):
        w = H[2, 0] * u + H[2, 1] * v + H[2, 2]
        rx = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / w - X
        ry = (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / w - Y
        res = max(res, float(np.hypot(rx, ry)))
    if res > RESIDUAL_TOL * max(width, height):
        return HomographyResult(False, None, [
            f"透视校准残差过大（{res:.3g} 单位），角点与已知矩形对应关系不一致，请检查点击顺序"
        ])

    # Invertibility guard.
    det = float(np.linalg.det(H))
    scale = float(np.linalg.norm(H)) ** 3
    if abs(det) < H_DET_TOL * max(scale, 1e-30):
        return HomographyResult(False, None, [
            "透视矩阵行列式接近 0，变换不可逆：四点几乎共线或矩形在照片中退化"
        ])
    row3_scale = float(np.hypot(H[2, 0], H[2, 1]))
    if row3_scale < _EPS:
        row3_scale = abs(float(H[2, 2]))
    for i, (u, v) in enumerate(image_points):
        w = abs(H[2, 0] * u + H[2, 1] * v + H[2, 2])
        if w < DENOM_TOL * max(row3_scale, 1.0):
            return HomographyResult(False, None, [
                f"角点 {i + 1} 映射时分母接近 0（落在无穷远线上），此视角下透视不可求逆"
            ])

    return HomographyResult(True, H.tolist())


def _normalizer(pts: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = pts.mean(axis=0)
    d = np.hypot(*(pts - centroid).T).mean()
    if d < _EPS:
        return np.eye(3), 1.0
    s = np.sqrt(2.0) / d
    T = np.array([[s, 0, -s * centroid[0]], [0, s, -s * centroid[1]], [0, 0, 1]])
    return T, float(s)


# ---------------------------------------------------------------------------
# Point / polygon transformation
# ---------------------------------------------------------------------------

def apply_homography(H: list[list[float]] | np.ndarray, point: list[float]) -> tuple[float, float, float]:
    """Return (x, y, w) where w is the homogeneous denominator."""
    M = np.asarray(H, dtype=float)
    u, v = float(point[0]), float(point[1])
    x = M[0, 0] * u + M[0, 1] * v + M[0, 2]
    y = M[1, 0] * u + M[1, 1] * v + M[1, 2]
    w = M[2, 0] * u + M[2, 1] * v + M[2, 2]
    return x, y, w


def _segments_cross(p: np.ndarray, q: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """Proper intersection of open segments p-q and a-b (excluding shared endpoints)."""
    def orient(r, s, t):
        return (s[0] - r[0]) * (t[1] - r[1]) - (s[1] - r[1]) * (t[0] - r[0])

    o1 = orient(p, q, a)
    o2 = orient(p, q, b)
    o3 = orient(a, b, p)
    o4 = orient(a, b, q)
    tol = 1e-10 * max(
        float(np.hypot(*(q - p))),
        float(np.hypot(*(b - a))),
        1.0,
    )
    if all(abs(o) < tol for o in (o1, o2, o3, o4)):
        # Collinear overlap -> self-intersecting.
        def on_seg(x, r, s):
            return (
                min(r[0], s[0]) - tol <= x[0] <= max(r[0], s[0]) + tol
                and min(r[1], s[1]) - tol <= x[1] <= max(r[1], s[1]) + tol
            )

        return any(on_seg(z, p, q) and on_seg(z, a, b) for z in (p, q, a, b)) and not (
            np.allclose(p, a, atol=tol) or np.allclose(p, b, atol=tol)
            or np.allclose(q, a, atol=tol) or np.allclose(q, b, atol=tol)
        )
    return ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol)) and (
        (o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol)
    )


def polygon_self_intersects(points: list[list[float]]) -> tuple[bool, str]:
    """Check a closed polygon for duplicate vertices or edge crossings."""
    n = len(points)
    if n < 3:
        return False, ""
    dup = _duplicate(points)
    if dup is not None:
        return True, f"顶点 {dup[0] + 1} 与顶点 {dup[1] + 1} 重合"
    pts = [_point(p) for p in points]
    for i in range(n):
        p, q = pts[i], pts[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue  # edges sharing a vertex are allowed
            a, b = pts[j], pts[(j + 1) % n]
            if _segments_cross(p, q, a, b):
                return True, f"边 {i + 1} 与边 {j + 1} 相交"
    return False, ""


@dataclass
class MeasurementResult:
    ok: bool
    area: float | None = None
    perimeter: float | None = None
    world_points: list[list[float]] | None = None
    errors: list[str] | None = None
    # Region-with-holes extras (None for a plain polygon / on failure).
    hole_areas: list[float] | None = None
    hole_perimeters: list[float] | None = None
    hole_world_points: list[list[list[float]]] | None = None
    gross_area: float | None = None


def measure_polygon(H: list[list[float]], image_points: list[list[float]]) -> MeasurementResult:
    """Transform a closed polygon and compute its area/perimeter in real units."""
    return measure_region(H, image_points, [])


# ---------------------------------------------------------------------------
# Multi-ring regions (one outer ring + zero or more holes)
# ---------------------------------------------------------------------------

def _ring_abs_area(points: list[list[float]]) -> float:
    s = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _ring_perimeter(points: list[list[float]]) -> float:
    p = 0.0
    n = len(points)
    for i in range(n):
        p += float(np.hypot(points[(i + 1) % n][0] - points[i][0],
                            points[(i + 1) % n][1] - points[i][1]))
    return p


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    abx, aby = float(b[0] - a[0]), float(b[1] - a[1])
    den = abx * abx + aby * aby
    if den < _EPS:
        return float(np.hypot(*(p - a)))
    t = (float(p[0] - a[0]) * abx + float(p[1] - a[1]) * aby) / den
    t = min(1.0, max(0.0, t))
    q = (a[0] + abx * t, a[1] + aby * t)
    return float(np.hypot(q[0] - p[0], q[1] - p[1]))


def _segment_relation(p: np.ndarray, q: np.ndarray,
                      a: np.ndarray, b: np.ndarray, tol: float) -> tuple[str, float]:
    """Classify two closed segments.

    Returns ``(kind, d)`` with kind in {"cross", "near", "far"}:

    * cross — proper intersection, endpoint contact, or collinear overlap;
    * near  — disjoint but closest approach at/below ``tol`` (near-tangency);
    * far   — well separated.
    """
    def orient(r, s, t):
        return (s[0] - r[0]) * (t[1] - r[1]) - (s[1] - r[1]) * (t[0] - r[0])

    o1, o2 = orient(p, q, a), orient(p, q, b)
    o3, o4 = orient(a, b, p), orient(a, b, q)
    if all(abs(o) <= tol for o in (o1, o2, o3, o4)):
        def on_seg(x, r, s):
            return (min(r[0], s[0]) - tol <= x[0] <= max(r[0], s[0]) + tol
                    and min(r[1], s[1]) - tol <= x[1] <= max(r[1], s[1]) + tol)

        if any(on_seg(z, p, q) and on_seg(z, a, b) for z in (p, q, a, b)):
            return "cross", 0.0  # collinear overlap

    if ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol)) and \
       ((o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol)):
        return "cross", 0.0

    # Endpoint-on-segment contacts (tangency through a vertex): perpendicular
    # distance within tol AND the projection lying on the segment itself.
    def on_segment(z, r, s):
        edge_len = float(np.hypot(s[0] - r[0], s[1] - r[1]))
        if edge_len < _EPS:
            return float(np.hypot(z[0] - r[0], z[1] - r[1])) <= tol
        cross = abs((s[0] - r[0]) * (z[1] - r[1]) - (s[1] - r[1]) * (z[0] - r[0]))
        if cross > tol * edge_len:
            return False
        return (min(r[0], s[0]) - tol <= z[0] <= max(r[0], s[0]) + tol
                and min(r[1], s[1]) - tol <= z[1] <= max(r[1], s[1]) + tol)

    if on_segment(a, p, q) or on_segment(b, p, q) \
            or on_segment(p, a, b) or on_segment(q, a, b):
        return "cross", 0.0

    d = min(
        _point_segment_distance(a, p, q),
        _point_segment_distance(b, p, q),
        _point_segment_distance(p, a, b),
        _point_segment_distance(q, a, b),
    )
    return ("near" if d <= tol else "far"), float(d)


def _point_in_ring(p: np.ndarray, ring: list[np.ndarray]) -> bool:
    """Strict ray-cast containment; points exactly on the boundary count as out."""
    inside = False
    n = len(ring)
    for i in range(n):
        a = ring[i]
        b = ring[(i + 1) % n]
        if (a[1] > p[1]) != (b[1] > p[1]):
            x_cross = a[0] + (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1])
            if x_cross > p[0]:
                inside = not inside
    return inside


def _ring_scale(ring: list[np.ndarray]) -> float:
    lo = (min(p[0] for p in ring), min(p[1] for p in ring))
    hi = (max(p[0] for p in ring), max(p[1] for p in ring))
    return max(float(np.hypot(hi[0] - lo[0], hi[1] - lo[1])), 1.0)


def validate_ring_topology(rings: list[list[list[float]]]) -> list[str]:
    """Validate outer ring + holes in *corrected (real-plane) coordinates*.

    Every ring must be a simple (non-self-intersecting) polygon.  Each hole
    must lie strictly inside the outer ring; holes must be pairwise strictly
    separated: crossing, touching (tangent), near-contact (within a relative
    tolerance) and nesting are all rejected.
    """
    if not rings:
        return []
    errors: list[str] = []

    def label(ri: int) -> str:
        return "外环" if ri == 0 else f"孔洞 {ri}"

    for ri, ring in enumerate(rings):
        if len(ring) < 3:
            errors.append(label(ri) + "至少需要 3 个顶点")
            continue
        hit, reason = polygon_self_intersects(ring)
        if hit:
            errors.append(f"{label(ri)}自交（{reason}），请调整顶点")

    if any(len(r) < 3 for r in rings):
        return errors

    np_rings = [[_point(p) for p in ring] for ring in rings]
    outer = np_rings[0]

    # outer ring vs each hole ------------------------------------------------
    for hi in range(1, len(np_rings)):
        hole = np_rings[hi]
        tol = RING_TOL * max(_ring_scale(outer), _ring_scale(hole))
        blocked = False
        near_d: float | None = None
        for i in range(len(outer)):
            p, q = outer[i], outer[(i + 1) % len(outer)]
            for j in range(len(hole)):
                a, b = hole[j], hole[(j + 1) % len(hole)]
                kind, d = _segment_relation(p, q, a, b, tol)
                if kind == "cross":
                    errors.append(
                        f"孔洞 {hi} 与外环相交或相切（外环边 {i + 1} 与孔洞边 {j + 1}），"
                        "孔洞必须完全位于外环内且与外环边界严格分离"
                    )
                    blocked = True
                    break
                if kind == "near" and near_d is None:
                    near_d = d
            if blocked:
                break
        if blocked:
            continue
        if near_d is not None:
            errors.append(
                f"孔洞 {hi} 与外环距离过近（{near_d:.3g}，判定为相切/越界），"
                "请将孔洞移入外环内部并与外边界保持距离"
            )
            continue
        for k, z in enumerate(hole):
            if not _point_in_ring(z, outer):
                errors.append(f"孔洞 {hi} 没有完全位于外环内（孔洞顶点 {k + 1} 位于外环之外）")
                break

    # hole vs hole -----------------------------------------------------------
    for h1 in range(1, len(np_rings)):
        for h2 in range(h1 + 1, len(np_rings)):
            r1, r2 = np_rings[h1], np_rings[h2]
            tol = RING_TOL * max(_ring_scale(r1), _ring_scale(r2))
            blocked = False
            near_d = None
            for i in range(len(r1)):
                p, q = r1[i], r1[(i + 1) % len(r1)]
                for j in range(len(r2)):
                    a, b = r2[j], r2[(j + 1) % len(r2)]
                    kind, d = _segment_relation(p, q, a, b, tol)
                    if kind == "cross":
                        errors.append(
                            f"孔洞 {h1} 与孔洞 {h2} 相交或相切（边 {i + 1} 与边 {j + 1}），"
                            "孔洞之间必须严格分离"
                        )
                        blocked = True
                        break
                    if kind == "near" and near_d is None:
                        near_d = d
                if blocked:
                    break
            if blocked:
                continue
            if near_d is not None:
                errors.append(
                    f"孔洞 {h1} 与孔洞 {h2} 距离过近（{near_d:.3g}，判定为相切），请将两个孔洞分开"
                )
                continue
            if _point_in_ring(r2[0], r1):
                errors.append(f"孔洞 {h2} 位于孔洞 {h1} 内部（孔洞不允许相互包含）")
            elif _point_in_ring(r1[0], r2):
                errors.append(f"孔洞 {h1} 位于孔洞 {h2} 内部（孔洞不允许相互包含）")

    return errors


def _transform_ring(H: list[list[float]], image_points: list[list[float]],
                    ring_label: str) -> tuple[list[list[float]] | None, str | None]:
    row3 = np.asarray(H, dtype=float)[2]
    row3_scale = float(np.hypot(row3[0], row3[1]))
    if row3_scale < _EPS:
        row3_scale = abs(float(row3[2]))
    world: list[list[float]] = []
    for i, p in enumerate(image_points):
        x, y, w = apply_homography(H, p)
        if abs(w) < DENOM_TOL * max(row3_scale, 1.0):
            return None, f"{ring_label}顶点 {i + 1} 的透视分母接近 0（位于无穷远线上），无法计算校正后坐标"
        world.append([x / w, y / w])
    return world, None


def measure_region(
    H: list[list[float]],
    outer_image: list[list[float]],
    holes_image: list[list[list[float]]] | None = None,
) -> MeasurementResult:
    """Measure a region with holes.

    Area is the outer-ring area minus every hole area; perimeter includes
    the outer boundary plus every hole boundary.  All ring relationships are
    validated in corrected real-plane coordinates.
    """
    holes_image = holes_image or []
    if len(outer_image) < 3:
        return MeasurementResult(False, None, None, None, ["多边形至少需要 3 个顶点"])

    rings_world: list[list[list[float]]] = []
    outer_world, err = _transform_ring(H, outer_image, "多边形")
    if err:
        return MeasurementResult(False, None, None, None, [err])
    rings_world.append(outer_world)

    for hi, hp in enumerate(holes_image, start=1):
        if len(hp) < 3:
            return MeasurementResult(False, None, None, None, [f"孔洞 {hi} 至少需要 3 个顶点"])
        hw, err = _transform_ring(H, hp, f"孔洞 {hi} ")
        if err:
            return MeasurementResult(False, None, None, None, [err])
        rings_world.append(hw)

    topo_errors = validate_ring_topology(rings_world)
    if topo_errors:
        return MeasurementResult(False, None, None, None, topo_errors)

    outer = rings_world[0]
    gross = _ring_abs_area(outer)
    perimeter = _ring_perimeter(outer)
    hole_areas: list[float] = []
    hole_perimeters: list[float] = []
    for ring in rings_world[1:]:
        ha = _ring_abs_area(ring)
        hp_ = _ring_perimeter(ring)
        hole_areas.append(ha)
        hole_perimeters.append(hp_)
        perimeter += hp_
    area = gross - sum(hole_areas)

    return MeasurementResult(
        True,
        area=area,
        perimeter=perimeter,
        world_points=outer,
        errors=[],
        hole_areas=hole_areas,
        hole_perimeters=hole_perimeters,
        hole_world_points=rings_world[1:],
        gross_area=gross,
    )


# ---------------------------------------------------------------------------
# Calibration sensitivity (Monte-Carlo perturbation of the four corners)
# ---------------------------------------------------------------------------
#
# The four calibration corners are perturbed jointly by a user-chosen pixel
# error radius.  For each of SENSITIVITY_SAMPLES fixed, reproducible trials the
# homography (and every region) is recomputed from scratch.  Degenerate
# homographies, corners landing outside the photo and geometrically illegal
# regions are *counted as failures* (never silently dropped): the response
# carries the valid/total coverage so the UI can refuse to show intervals when
# too many trials failed.
SENSITIVITY_SAMPLES = 128
SENSITIVITY_MIN_COVERAGE = 0.80


class _SplitMix64:
    """Deterministic 64-bit PRNG (pure Python, identical on every platform)."""

    MASK = (1 << 64) - 1

    def __init__(self, seed: int):
        self.state = seed & self.MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self.MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK
        return z ^ (z >> 31)

    def next_unit(self) -> float:
        # 53-bit mantissa -> [0, 1)
        return (self.next_u64() >> 11) * (2.0 ** -53)


def sensitivity_seed(corners: list[list[float]], width: float, height: float,
                     radius: float) -> str:
    """Stable hex seed for one (geometry + radius) configuration."""
    payload = json.dumps(
        {
            "c": [[round(float(u), 6), round(float(v), 6)] for u, v in corners],
            "w": round(float(width), 9),
            "h": round(float(height), 9),
            "r": round(float(radius), 6),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _disk_offsets(rng: _SplitMix64, radius: float) -> tuple[float, float]:
    """Uniform sample inside the closed disk of pixel radius ``radius``."""
    ang = 2.0 * math.pi * rng.next_unit()
    rad = radius * math.sqrt(rng.next_unit())
    return rad * math.cos(ang), rad * math.sin(ang)


@dataclass
class RegionSensitivity:
    valid_samples: int
    total_samples: int
    area_p5: float | None
    area_p50: float | None
    area_p95: float | None
    perimeter_p5: float | None
    perimeter_p50: float | None
    perimeter_p95: float | None
    invalid_geometry_samples: int


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    """P5 / P50 / P95 with numpy's default linear interpolation."""
    arr = np.asarray(values, dtype=float)
    p5, p50, p95 = np.percentile(arr, [5.0, 50.0, 95.0])
    return float(p5), float(p50), float(p95)


def assess_region_sensitivity(
    H_matrices: list[list[list[float]] | None],
    outer_image: list[list[float]],
    holes_image: list[list[list[float]]],
) -> RegionSensitivity:
    """Re-measure one region over the successful-homography trial set.

    ``H_matrices`` has one entry per trial: ``None`` when that trial's
    calibration failed (out-of-bounds corner or degenerate homography); such
    trials count against coverage.  A region that becomes illegal (self
    intersection, hole escapes the outer ring, point near the line at
    infinity, non-finite numbers) is a failed trial too — it is never dropped.
    """
    total = len(H_matrices)
    areas: list[float] = []
    perimeters: list[float] = []
    invalid_geom = 0
    for H in H_matrices:
        if H is None:
            continue
        r = measure_region(H, outer_image, holes_image)
        if not r.ok or not (math.isfinite(r.area) and math.isfinite(r.perimeter)):
            invalid_geom += 1
            continue
        areas.append(r.area)
        perimeters.append(r.perimeter)

    valid = len(areas)
    if areas:
        a5, a50, a95 = _percentiles(areas)
        l5, l50, l95 = _percentiles(perimeters)
    else:
        a5 = a50 = a95 = l5 = l50 = l95 = None
    return RegionSensitivity(
        valid_samples=valid,
        total_samples=total,
        area_p5=a5, area_p50=a50, area_p95=a95,
        perimeter_p5=l5, perimeter_p50=l50, perimeter_p95=l95,
        invalid_geometry_samples=invalid_geom,
    )


def run_sensitivity(
    corners: list[list[float]],
    width: float,
    height: float,
    radius: float,
    regions: list[tuple[list[list[float]], list[list[list[float]]]]],
    img_w: float | None = None,
    img_h: float | None = None,
) -> dict:
    """Run the fixed 128-trial calibration-sensitivity assessment.

    Parameters
    ----------
    corners:
        The four current calibration corners (image pixels), order
        TL -> TR -> BR -> BL.
    width, height:
        Known real dimensions of the calibration rectangle.
    radius:
        Uniform pixel error radius applied to every corner (>= 0).
    regions:
        Stored regions as ``(outer_image_points, [hole_image_points, ...])``.
    img_w, img_h:
        Photo size.  When given, trials pushing a corner outside the photo
        (strictly past its borders) are classified as out-of-bounds failures
        and counted, never retried.

    The perturbation stream derives from :func:`sensitivity_seed`, so the same
    session geometry + radius always reproduce the exact same 128 trials.
    """
    n = SENSITIVITY_SAMPLES
    seed = sensitivity_seed(corners, width, height, radius)
    rng = _SplitMix64(int(seed[:16], 16))

    H_matrices: list[list[list[float]] | None] = []
    out_of_bounds = 0
    degenerate = 0
    for _ in range(n):
        trial_corners = []
        for u, v in corners:
            du, dv = _disk_offsets(rng, radius)
            trial_corners.append([u + du, v + dv])
        if img_w is not None and img_h is not None and any(
            not (0.0 <= u <= img_w and 0.0 <= v <= img_h)
            for u, v in trial_corners
        ):
            out_of_bounds += 1
            H_matrices.append(None)
            continue
        res = solve_homography(trial_corners, width, height)
        if not res.ok:
            degenerate += 1
            H_matrices.append(None)
            continue
        H_matrices.append(res.H)

    region_results = [
        assess_region_sensitivity(H_matrices, outer, holes)
        for outer, holes in regions
    ]
    return {
        "samples": n,
        "radius_px": float(radius),
        "seed": seed,
        "out_of_bounds_samples": out_of_bounds,
        "degenerate_samples": degenerate,
        "homography_valid_samples": n - out_of_bounds - degenerate,
        "min_coverage": SENSITIVITY_MIN_COVERAGE,
        "regions": region_results,
    }
