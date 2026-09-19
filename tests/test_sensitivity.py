"""Sensitivity analysis tests: deterministic perturbations, failure accounting,
and the 80% valid-sample gate. Run: python tests/test_sensitivity.py
"""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.sensitivity import SAMPLE_COUNT, perturbation_offsets  # noqa: E402

client = TestClient(app)

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)


# Same synthetic camera as test_api.py.
K = np.array([[800.0, 0, 512], [0, 800, 384], [0, 0, 1.0]])
r1 = np.array([0.99, 0.0, -0.1494])
r2 = np.array([0.06, 1.0, 0.09])
P = K @ np.column_stack([r1, r2, [0.0, 0.0, 1.2]])


def project(x, y):
    v = P @ np.array([x, y, 1.0])
    return v[0] / v[2], v[1] / v[2]


def pts(world):
    return [{"x": float(u), "y": float(v)} for u, v in (project(x, y) for x, y in world)]


print("== perturbation generator ==")
W, H = 0.1, 0.06
corners = [project(0, 0), project(W, 0), project(W, H), project(0, H)]
o1 = perturbation_offsets(corners, W, H)
o2 = perturbation_offsets(corners, W, H)
check("shape (128,4,2)", o1.shape == (SAMPLE_COUNT, 4, 2))
check("deterministic for same geometry", np.array_equal(o1, o2))
check("offsets within unit disk", float(np.hypot(o1[..., 0], o1[..., 1]).max()) <= 1.0 + 1e-12)
o3 = perturbation_offsets([[c[0] + 1.0, c[1]] for c in corners], W, H)
check("depends on geometry", not np.array_equal(o1, o3))

print("== session setup ==")
buf = io.BytesIO()
Image.new("RGB", (1024, 768), (40, 50, 60)).save(buf, "JPEG")
buf.seek(0)
sid = client.post("/api/sessions").json()["session_id"]
r = client.post(f"/api/sessions/{sid}/image", files={"file": ("t.jpg", buf, "image/jpeg")})
check("image upload", r.status_code == 200)

cal_payload = {"corners": pts([(0, 0), (W, 0), (W, H), (0, H)]),
               "rect_width": W, "rect_height": H, "unit": "mm"}
r = client.put(f"/api/sessions/{sid}/calibration", json=cal_payload)
check("calibration valid", r.status_code == 200 and r.json()["calibration"]["valid"])

# plain rectangle: world [0.02,0.07]x[0.01,0.05] -> area .002, perimeter .18
r = client.post(f"/api/sessions/{sid}/polygons",
                json={"name": "净区", "points": pts([(0.02, 0.01), (0.07, 0.01), (0.07, 0.05), (0.02, 0.05)])})
check("plain region added", r.status_code == 200 and r.json()["polygon"]["valid"])
# holed region: gross .0032, hole .0004 -> net .0028, perimeter .32
r = client.post(f"/api/sessions/{sid}/polygons", json={
    "name": "带孔",
    "points": pts([(0.01, 0.01), (0.09, 0.01), (0.09, 0.05), (0.01, 0.05)]),
    "holes": [{"points": pts([(0.03, 0.02), (0.05, 0.02), (0.05, 0.04), (0.03, 0.04)])}],
})
check("holed region added", r.status_code == 200 and r.json()["polygon"]["valid"])

print("== sensitivity: small radius ==")
r = client.post(f"/api/sessions/{sid}/sensitivity", json={"radius_px": 0.5})
check("sensitivity 200", r.status_code == 200, str(r.status_code))
d = r.json()
check("128 samples", d["sample_count"] == 128)
check("radius echoed", d["radius_px"] == 0.5)
check("all samples valid", d["valid_samples"] == 128, f"valid={d['valid_samples']}")
check("valid ratio 1.0", d["valid_ratio"] == 1.0)
check("reliable", d["reliable"] is True)
check("no recalibrate message", d["message"] is None)
check("no failures", d["failure_counts"] == {"out_of_bounds": 0, "degenerate": 0})
check("two polygon entries", len(d["polygons"]) == 2)
e1 = d["polygons"][0]
check("poly reliable", e1["reliable"] is True and e1["valid_ratio"] == 1.0)
check("poly sample math", e1["valid_samples"] == 128 and e1["failed_geometry"] == 0)
check("area percentiles ordered",
      e1["area"]["p5"] <= e1["area"]["p50"] <= e1["area"]["p95"], str(e1["area"]))
check("perim percentiles ordered",
      e1["perimeter"]["p5"] <= e1["perimeter"]["p50"] <= e1["perimeter"]["p95"])
check("area p50 near baseline .002", abs(e1["area"]["p50"] - 0.002) < 2e-4,
      f"p50={e1['area']['p50']}")
check("interval brackets baseline", e1["area"]["p5"] <= 0.002 <= e1["area"]["p95"])
check("perim p50 near baseline .18", abs(e1["perimeter"]["p50"] - 0.18) < 2e-3)
e2 = d["polygons"][1]
check("holed p50 near .0028", abs(e2["area"]["p50"] - 0.0028) < 3e-4,
      f"p50={e2['area']['p50']}")

print("== reproducibility over HTTP ==")
r = client.post(f"/api/sessions/{sid}/sensitivity", json={"radius_px": 0.5})
check("identical repeat report", r.json() == d)

print("== interval widens with radius ==")
d3 = client.post(f"/api/sessions/{sid}/sensitivity", json={"radius_px": 3.0}).json()
w_small = e1["area"]["p95"] - e1["area"]["p5"]
w_big = d3["polygons"][0]["area"]["p95"] - d3["polygons"][0]["area"]["p5"]
check("larger radius -> wider interval", w_big > w_small, f"{w_small:.3g} -> {w_big:.3g}")

print("== failure accounting (huge radius) ==")
r = client.post(f"/api/sessions/{sid}/sensitivity", json={"radius_px": 600.0})
d = r.json()
check("failures counted, not dropped", d["valid_samples"] < 128, f"valid={d['valid_samples']}")
check("ratio consistent with counts",
      abs(d["valid_ratio"] - d["valid_samples"] / 128) < 1e-12)
check("out-of-bounds/degenerate recorded",
      d["failure_counts"]["out_of_bounds"] + d["failure_counts"]["degenerate"] > 0,
      str(d["failure_counts"]))
check("unreliable below 80%", d["reliable"] is False and d["valid_ratio"] < 0.8,
      f"ratio={d['valid_ratio']:.3f}")
check("recalibrate hint", bool(d["message"]) and "重新标定" in d["message"])
for e in d["polygons"]:
    check("intervals withheld", e["area"] is None and e["perimeter"] is None)
    check("poly unreliable", e["reliable"] is False and e["valid_ratio"] < 0.8)
    check("poly failures within homography failures", e["valid_samples"] <= d["valid_samples"])
    check("poly sample math", e["valid_samples"] + (128 - e["valid_samples"]) == 128
          and e["failed_geometry"] >= 0)

print("== error cases ==")
r = client.post(f"/api/sessions/{sid}/sensitivity", json={"radius_px": 0})
check("zero radius -> 422", r.status_code == 422)
r = client.post(f"/api/sessions/{sid}/sensitivity", json={"radius_px": -2})
check("negative radius -> 422", r.status_code == 422)
r = client.post(f"/api/sessions/{sid}/sensitivity", json={"radius_px": 5000})
check("radius above cap -> 422", r.status_code == 422)
r = client.post("/api/sessions/deadbeef/sensitivity", json={"radius_px": 1})
check("missing session -> 404", r.status_code == 404)

sid2 = client.post("/api/sessions").json()["session_id"]
r = client.post(f"/api/sessions/{sid2}/sensitivity", json={"radius_px": 1})
check("no image -> 400", r.status_code == 400)
buf.seek(0)
client.post(f"/api/sessions/{sid2}/image", files={"file": ("t.jpg", buf, "image/jpeg")})
r = client.post(f"/api/sessions/{sid2}/sensitivity", json={"radius_px": 1})
check("no calibration -> 400", r.status_code == 400)
client.put(f"/api/sessions/{sid2}/calibration", json=cal_payload)
r = client.post(f"/api/sessions/{sid2}/sensitivity", json={"radius_px": 1})
check("no polygons -> 200 with empty list",
      r.status_code == 200 and r.json()["polygons"] == [] and r.json()["sample_count"] == 128)

print()
if fails:
    print(f"{len(fails)} FAILURES: {fails}")
    sys.exit(1)
print("ALL SENSITIVITY TESTS PASSED")
