"""Calibration sensitivity analysis (标定敏感性评估).

How much do the measurements move if every calibration corner is off by up to
``radius_px`` pixels?  A fixed set of ``SAMPLE_COUNT`` perturbations is derived
deterministically from the calibration geometry (same geometry -> same
perturbations -> same report); each corner is nudged inside the error disk,
the homography is re-solved and every region is re-measured.

Failure accounting — failing samples are counted, never silently dropped:

* 越界 (out of bounds) — a perturbed corner leaves the image bounds;
* 退化 (degenerate)   — the perturbed quad no longer yields a valid homography
  (collinear / non-convex / near-singular / residual too large);
* 几何非法 (invalid)  — a region fails validation under the perturbed
  homography (self-intersection, hole escaping the outer ring, vertex on the
  line at infinity, ...).

Percentile intervals (P5/P50/P95) are reported only when the valid-sample
ratio reaches ``MIN_VALID_RATIO``; below it the intervals are withheld and the
caller is told to re-calibrate.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from .geometry import measure_region, solve_homography

SAMPLE_COUNT = 128          # fixed number of perturbation sets
MIN_VALID_RATIO = 0.8       # intervals are hidden below this valid-sample ratio

RECALIBRATE_HINT = (
    "有效样本比例低于 80%：当前标定对该像素误差过于敏感，区间不予显示。"
    "请重新标定（让参照矩形在画面中占更大比例、减小拍摄倾角，或调小误差半径）。"
)


def _seed(corners: list[list[float]], rect_width: float, rect_height: float) -> int:
    payload = json.dumps(
        {"corners": corners, "rect_width": rect_width, "rect_height": rect_height},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def perturbation_offsets(
    corners: list[list[float]],
    rect_width: float,
    rect_height: float,
    n: int = SAMPLE_COUNT,
) -> np.ndarray:
    """(n, 4, 2) unit-disk offsets, deterministic for a given calibration.

    Multiplying by the error radius yields the actual pixel offsets, so one
    geometry probed with different radii reuses a single perturbation family.
    """
    rng = np.random.default_rng(_seed(corners, rect_width, rect_height))
    angle = rng.uniform(0.0, 2.0 * np.pi, size=(n, 4))
    radius = np.sqrt(rng.uniform(0.0, 1.0, size=(n, 4)))  # uniform over the disk
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=-1)


def _percentiles(values: list[float]) -> dict[str, float]:
    p5, p50, p95 = np.percentile(np.asarray(values, dtype=float), [5.0, 50.0, 95.0])
    return {"p5": float(p5), "p50": float(p50), "p95": float(p95)}


def run_sensitivity(
    corners: list[list[float]],
    rect_width: float,
    rect_height: float,
    radius_px: float,
    polygons: list[dict[str, Any]],
    img_width: int,
    img_height: int,
) -> dict[str, Any]:
    """Perturb the calibration corners and re-measure every region.

    ``polygons`` entries: ``{"id", "name", "outer": [[u, v], ...],
    "holes": [[[u, v], ...], ...]}`` (image pixel coordinates).
    """
    base = np.asarray(corners, dtype=float).reshape(4, 2)
    offsets = perturbation_offsets(corners, rect_width, rect_height)

    out_of_bounds = 0
    degenerate = 0
    hom_ok = 0
    valid_counts = [0] * len(polygons)
    failed_geometry = [0] * len(polygons)
    areas: list[list[float]] = [[] for _ in polygons]
    perimeters: list[list[float]] = [[] for _ in polygons]

    for k in range(SAMPLE_COUNT):
        perturbed = base + offsets[k] * float(radius_px)
        # 越界：扰动后的角点必须仍在图像内（与标定接口同一容差）。
        us, vs = perturbed[:, 0], perturbed[:, 1]
        if not (bool(np.all(us >= -1.0)) and bool(np.all(us <= img_width + 1.0))
                and bool(np.all(vs >= -1.0)) and bool(np.all(vs <= img_height + 1.0))):
            out_of_bounds += 1
            continue
        res = solve_homography(perturbed.tolist(), rect_width, rect_height)
        if not res.ok or res.H is None:
            degenerate += 1  # 退化：共线 / 非凸 / 近奇异 / 残差过大
            continue
        hom_ok += 1
        for i, poly in enumerate(polygons):
            m = measure_region(res.H, poly["outer"], poly["holes"])
            if not m.ok:
                failed_geometry[i] += 1  # 几何非法：计入该区域失败比例
                continue
            valid_counts[i] += 1
            areas[i].append(float(m.area))
            perimeters[i].append(float(m.perimeter))

    valid_ratio = hom_ok / SAMPLE_COUNT
    reliable = valid_ratio >= MIN_VALID_RATIO
    report: dict[str, Any] = {
        "radius_px": float(radius_px),
        "sample_count": SAMPLE_COUNT,
        "valid_samples": hom_ok,
        "valid_ratio": valid_ratio,
        "reliable": reliable,
        "message": None if reliable else RECALIBRATE_HINT,
        "failure_counts": {"out_of_bounds": out_of_bounds, "degenerate": degenerate},
        "polygons": [],
    }
    for i, poly in enumerate(polygons):
        ratio = valid_counts[i] / SAMPLE_COUNT
        ok = ratio >= MIN_VALID_RATIO
        report["polygons"].append({
            "id": poly["id"],
            "name": poly["name"],
            "sample_count": SAMPLE_COUNT,
            "valid_samples": valid_counts[i],
            "valid_ratio": ratio,
            "reliable": ok,
            "failed_geometry": failed_geometry[i],
            "area": _percentiles(areas[i]) if ok else None,
            "perimeter": _percentiles(perimeters[i]) if ok else None,
        })
    return report
