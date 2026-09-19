"""End-to-end API test using FastAPI TestClient and a generated photo.

The photo renders a known rectangle in perspective (using the same camera
matrix as test_geometry.py); corners are queried from the projection and
posted to the API. Run: python tests/test_api.py
"""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------- camera ---
# A closer camera with noticeable perspective; the whole target fits a
# 1024x768 photo.
K = np.array([[800.0, 0, 512], [0, 800, 384], [0, 0, 1.0]])
r1 = np.array([0.99, 0.0, -0.1494])
r2 = np.array([0.06, 1.0, 0.09])
P = K @ np.column_stack([r1, r2, [0.0, 0.0, 1.2]])


def project(x, y):
    v = P @ np.array([x, y, 1.0])
    return v[0] / v[2], v[1] / v[2]


# Make a simple image (the pixels are irrelevant; geometry comes from the API
# points, but endpoints require a decodable image).
buf = io.BytesIO()
Image.new("RGB", (1024, 768), (40, 50, 60)).save(buf, "JPEG")
buf.seek(0)

print("== session + upload ==")
sid = client.post("/api/sessions").json()["session_id"]
check("session created", bool(sid))

r = client.post(f"/api/sessions/{sid}/image", files={"file": ("t.jpg", buf, "image/jpeg")})
check("image upload", r.status_code == 200, str(r.status_code))
check("image dims", r.json()["image"]["width"] == 1024)

r = client.post(f"/api/sessions/{sid}/image", files={"file": ("x.bin", b"not-an-image", "application/octet-stream")})
check("garbage upload rejected", r.status_code == 400)

r = client.get(f"/api/sessions/{'deadbeef'}")
check("missing session 404", r.status_code == 404)

print("== calibration ==")
W, H = 0.1, 0.06
corners = [project(0, 0), project(W, 0), project(W, H), project(0, H)]
payload = {"corners": [{"x": float(u), "y": float(v)} for u, v in corners],
           "rect_width": W, "rect_height": H, "unit": "mm"}
r = client.put(f"/api/sessions/{sid}/calibration", json=payload)
check("calibration 200", r.status_code == 200, str(r.status_code))
cal = r.json()["calibration"]
check("calibration valid", cal["valid"], cal.get("errors"))
check("H is 3x3", cal["valid"] and len(cal["H"]) == 3 and len(cal["H"][0]) == 3)

# degenerate calibration
bad = {"corners": [{"x": 0, "y": 0}, {"x": 50, "y": 0}, {"x": 100, "y": 0}, {"x": 150, "y": 0}],
       "rect_width": W, "rect_height": H, "unit": "mm"}
r = client.put(f"/api/sessions/{sid}/calibration", json=bad)
check("collinear rejected", r.status_code == 200 and not r.json()["calibration"]["valid"])

# restore good calibration for the rest
client.put(f"/api/sessions/{sid}/calibration", json=payload)

print("== polygons ==")
# interior rectangle [0.02,0.07] x [0.01,0.05] -> area .05*.04=.002, perim .18
poly_img = [project(0.02, 0.01), project(0.07, 0.01), project(0.07, 0.05), project(0.02, 0.05)]
r = client.post(f"/api/sessions/{sid}/polygons",
                json={"points": [{"x": float(u), "y": float(v)} for u, v in poly_img]})
check("polygon add", r.status_code == 200)
p1 = r.json()["polygon"]
check("poly area .002", p1["valid"] and abs(p1["area"] - 0.002) < 1e-9, f"area={p1['area']}")
check("poly perim .18", p1["valid"] and abs(p1["perimeter"] - 0.18) < 1e-9)
check("empty holes echo", p1["holes"] == [])
check("gross area mirrors area", abs(p1["gross_area"] - 0.002) < 1e-9)

# self-intersecting polygon: POST is rejected with 422 and stores nothing
bt = [project(0, 0), project(W, H), project(W, 0), project(0, H)]
r = client.post(f"/api/sessions/{sid}/polygons",
                json={"points": [{"x": float(u), "y": float(v)} for u, v in bt]})
check("self-intersect POST -> 422", r.status_code == 422, str(r.status_code))
check("422 mentions self-intersection", "自交" in r.json()["detail"], r.json().get("detail"))
check("422 stores no polygon", r.json()["polygon"] is None)

# a VALID triangle is accepted: (0,0),(0.05,0),(0,0.03) area .00075
tri = [project(0, 0), project(0.05, 0), project(0, 0.03)]
r = client.post(f"/api/sessions/{sid}/polygons",
                json={"points": [{"x": float(u), "y": float(v)} for u, v in tri],
                      "name": "三角"})
check("triangle add 200", r.status_code == 200, str(r.status_code))
p2 = r.json()["polygon"]
check("triangle area .00075", p2["valid"] and abs(p2["area"] - 0.00075) < 1e-9, f"area={p2['area']}")
check("triangle rename", p2["name"] == "三角")

# invalid geometry PUT must be rejected with 422 and keep the last valid triangle
r = client.put(f"/api/sessions/{sid}/polygons/{p2['id']}",
               json={"points": [{"x": float(u), "y": float(v)} for u, v in bt]})
check("invalid PUT -> 422", r.status_code == 422, str(r.status_code))
body = r.json()
check("422 keeps last valid", body["polygon"]["valid"] and abs(body["polygon"]["area"] - 0.00075) < 1e-9)
r = client.get(f"/api/sessions/{sid}")
kept = [p for p in r.json()["polygons"] if p["id"] == p2["id"]][0]
check("server geometry unchanged", kept["valid"] and abs(kept["area"] - 0.00075) < 1e-9)

r = client.delete(f"/api/sessions/{sid}/polygons/{p2['id']}")
check("poly delete", r.status_code == 200)

# out-of-range vertex rejected
r = client.post(f"/api/sessions/{sid}/polygons",
                json={"points": [{"x": -5, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}]})
check("out-of-range rejected", r.status_code == 400)

print("== holes ==")
# big region: world [0.01,0.09] x [0.01,0.05] -> gross .08*.04=.0032
reg_img = [project(0.01, 0.01), project(0.09, 0.01), project(0.09, 0.05), project(0.01, 0.05)]
r = client.post(f"/api/sessions/{sid}/polygons", json={
    "name": "带孔区域",
    "points": [{"x": float(u), "y": float(v)} for u, v in reg_img],
    "holes": [{"points": [{"x": float(u), "y": float(v)} for u, v in
               [project(0.03, 0.02), project(0.05, 0.02), project(0.05, 0.04), project(0.03, 0.04)]]}],
})
check("holed region add", r.status_code == 200, str(r.status_code))
ph = r.json()["polygon"]
# hole area .02*.02=.0004 -> net .0028 ; perim outer .24 + hole .08 = .32
check("holed region valid", ph["valid"], ph.get("errors"))
check("net area .0028", ph["valid"] and abs(ph["area"] - 0.0028) < 1e-9, f"area={ph['area']}")
check("gross .0032", ph["valid"] and abs(ph["gross_area"] - 0.0032) < 1e-9)
check("total perim .32", ph["valid"] and abs(ph["perimeter"] - 0.32) < 1e-9, f"p={ph['perimeter']}")
check("one hole echoed", len(ph["holes"]) == 1 and abs(ph["holes"][0]["area"] - 0.0004) < 1e-9)
phid = ph["id"]

# hole crossing the outer boundary -> 422, last valid holed region retained
bad_hole = [{"points": [{"x": float(u), "y": float(v)} for u, v in
             [project(0.08, 0.04), project(0.095, 0.04), project(0.095, 0.055), project(0.08, 0.055)]]}]
r = client.put(f"/api/sessions/{sid}/polygons/{phid}", json={
    "points": [{"x": float(u), "y": float(v)} for u, v in reg_img],
    "holes": bad_hole,
})
check("crossing hole PUT 422", r.status_code == 422, str(r.status_code))
check("422 mentions conflict", "相交或相切" in r.json()["detail"])
check("last valid holes preserved", len(r.json()["polygon"]["holes"]) == 1)

# two overlapping holes -> 422
r = client.put(f"/api/sessions/{sid}/polygons/{phid}", json={
    "points": [{"x": float(u), "y": float(v)} for u, v in reg_img],
    "holes": [
        {"points": [{"x": float(u), "y": float(v)} for u, v in
         [project(0.02, 0.02), project(0.05, 0.02), project(0.05, 0.045), project(0.02, 0.045)]]},
        {"points": [{"x": float(u), "y": float(v)} for u, v in
         [project(0.045, 0.025), project(0.07, 0.025), project(0.07, 0.045), project(0.045, 0.045)]]},
    ],
})
check("overlapping holes 422", r.status_code == 422 and "孔洞" in r.json()["detail"])

# valid second (separate) hole -> net .0032 - .0004 - .0004 = .0024
r = client.put(f"/api/sessions/{sid}/polygons/{phid}", json={
    "points": [{"x": float(u), "y": float(v)} for u, v in reg_img],
    "holes": [
        {"points": [{"x": float(u), "y": float(v)} for u, v in
         [project(0.03, 0.02), project(0.05, 0.02), project(0.05, 0.04), project(0.03, 0.04)]]},
        {"points": [{"x": float(u), "y": float(v)} for u, v in
         [project(0.06, 0.02), project(0.08, 0.02), project(0.08, 0.04), project(0.06, 0.04)]]},
    ],
})
check("two-hole update 200", r.status_code == 200)
ph2 = r.json()["polygon"]
check("two-hole net .0024", ph2["valid"] and abs(ph2["area"] - 0.0024) < 1e-9, f"area={ph2['area']}")
check("two-hole perim .40", ph2["valid"] and abs(ph2["perimeter"] - 0.40) < 1e-9)

print("== calibration sensitivity ==")
sens_url = f"/api/sessions/{sid}/sensitivity"

# guards: bad radius shape
r = client.post(sens_url, json={"radius_px": -1})
check("negative radius 422", r.status_code == 422, str(r.status_code))

r = client.post(sens_url, json={"radius_px": 2})
check("sensitivity 200", r.status_code == 200, str(r.status_code))
sd = r.json()
check("128 fixed trials", sd["samples"] == 128, str(sd.get("samples")))
check("radius echoed", abs(sd["radius_px"] - 2) < 1e-12)
check("no oob/deg at 2px", sd["out_of_bounds_samples"] == 0 and sd["degenerate_samples"] == 0,
      f"oob={sd['out_of_bounds_samples']} deg={sd['degenerate_samples']}")
check("homography valid 128", sd["homography_valid_samples"] == 128)
check("region count matches", len(sd["regions"]) == 2, str(len(sd["regions"])))
reg1 = sd["regions"][0]
check("coverage field", abs(reg1["coverage"] - 1.0) < 1e-12)
check("intervals shown at 100%", reg1["intervals_shown"] is True)
check("p5<=p50<=p95 area",
      reg1["area_p5"] <= reg1["area_p50"] <= reg1["area_p95"],
      f"{reg1['area_p5']},{reg1['area_p50']},{reg1['area_p95']}")
check("p5<=p50<=p95 perim",
      reg1["perimeter_p5"] <= reg1["perimeter_p50"] <= reg1["perimeter_p95"])
# nominal area .002 should sit inside its [P5,P95] band
check("nominal inside band", reg1["area_p5"] <= 0.002 <= reg1["area_p95"],
      f"[{reg1['area_p5']},{reg1['area_p95']}]")
check("no note when shown", reg1["note"] == "")

# reproducibility: same geometry + radius -> identical seed and percentiles
r2b = client.post(sens_url, json={"radius_px": 2})
sd2 = r2b.json()
check("seed reproducible", sd2["seed"] == sd["seed"])
check("percentiles reproducible",
      sd2["regions"][0]["area_p5"] == reg1["area_p5"]
      and sd2["regions"][0]["perimeter_p95"] == reg1["perimeter_p95"])

# radius changes the perturbation stream -> different seed
r3 = client.post(sens_url, json={"radius_px": 3})
check("different radius -> different seed", r3.json()["seed"] != sd["seed"])

# radius 0: every trial is the unperturbed calibration -> degenerate band
r0 = client.post(sens_url, json={"radius_px": 0})
sd0 = r0.json()
e0 = sd0["regions"][0]
check("radius 0 collapses band",
      e0["valid_samples"] == 128 and abs(e0["area_p5"] - e0["area_p95"]) < 1e-12
      and abs(e0["area_p50"] - 0.002) < 1e-9, str(e0["area_p50"]))

# huge radius drives most trials out of bounds / degenerate: failures are
# counted (not dropped), coverage falls under 80%, intervals are hidden.
rb = client.post(sens_url, json={"radius_px": 600})
check("big radius 200", rb.status_code == 200)
sdb = rb.json()
check("failures counted", sdb["samples"] == 128 and
      sdb["out_of_bounds_samples"] + sdb["degenerate_samples"]
      + sdb["homography_valid_samples"] == 128,
      f"{sdb['out_of_bounds_samples']}+{sdb['degenerate_samples']}+{sdb['homography_valid_samples']}")
check("big radius has failures",
      sdb["out_of_bounds_samples"] + sdb["degenerate_samples"] > 0)
badreg = sdb["regions"][0]
check("low coverage hides intervals",
      badreg["intervals_shown"] is False and badreg["area_p5"] is None
      and badreg["perimeter_p95"] is None)
check("re-calibration hint present", "重新标定" in badreg["note"], badreg["note"])
check("global flag false", sdb["all_intervals_shown"] is False)
check("coverage consistent",
      abs(badreg["coverage"] - badreg["valid_samples"] / 128) < 1e-12
      and badreg["coverage"] < 0.8)

# radius persists with the session (old sessions simply have null)
g = client.get(f"/api/sessions/{sid}").json()
check("radius persisted", abs(g["sensitivity_radius"] - 600) < 1e-9)

# changing geometry invalidates nothing silently: the assessment is derived,
# so the next run uses the new geometry and a new seed.  Restore a good 2px
# radius for the export assertions below.
client.post(sens_url, json={"radius_px": 2})

print("== export ==")
r = client.get(f"/api/sessions/{sid}/export")
check("export 200", r.status_code == 200)
data = r.json()
check("schema tag", data["schema"] == "photo-ruler/1.0")
check("calibration in export", data["calibration"] and data["calibration"]["valid"])
check("polygon count", len(data["polygons"]) == 2)
check("all valid flag", data["all_measurements_valid"] is True)
check("export area", abs(data["polygons"][0]["area"] - 0.002) < 1e-9)
check("homography present", len(data["calibration"]["homography_image_to_world"]) == 3)
ph_exp = data["polygons"][1]
check("export hole count", len(ph_exp["holes"]) == 2)
check("export hole fields",
      ph_exp["holes"][0]["world_points"] is not None and ph_exp["holes"][0]["index"] == 1)
check("export gross area", abs(ph_exp["gross_area"] - 0.0032) < 1e-9)
sens_exp = data.get("sensitivity")
check("export has sensitivity section", sens_exp is not None)
check("export sensitivity available", sens_exp and sens_exp.get("available") is True)
check("export sensitivity 128", sens_exp and sens_exp["samples"] == 128)
check("export sensitivity regions", sens_exp and len(sens_exp["regions"]) == 2)
if sens_exp and sens_exp["regions"]:
    check("export interval values",
          sens_exp["regions"][0]["area_p5"] is not None
          and sens_exp["regions"][0]["perimeter_p5"] is not None)

print("== persistence / old session compat ==")
r = client.get(f"/api/sessions/{sid}")
check("session reload has data", r.json()["image"] is not None and len(r.json()["polygons"]) == 2)
reloaded_holes = r.json()["polygons"][1]["holes"]
check("holes survive reload", len(reloaded_holes) == 2 and len(reloaded_holes[0]["points"]) == 4)

# Simulate an old-session JSON (polygon record without "holes") and reload it.
import json as _json
from app import main as mainmod
old_sid = "legacy0001"
legacy = {
    "id": old_sid,
    "created_at": 0,
    "image": {"filename": "x.jpg", "width": 1024, "height": 768},
    "calibration": None,
    "polygons": [{"id": "pold", "name": "旧区域",
                  "points": [{"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}, {"x": 2.0, "y": 2.0}]}],
}
(mainmod.SESS_DIR / f"{old_sid}.json").write_text(_json.dumps(legacy), encoding="utf-8")
mainmod._load_sessions()
r = client.get(f"/api/sessions/{old_sid}")
check("legacy session loads", r.status_code == 200)
check("legacy polygon gets empty holes", r.json()["polygons"][0]["holes"] == [])
check("legacy polygon measurable=false w/o calib", r.json()["polygons"][0]["valid"] is False)
check("legacy session no sensitivity radius", r.json().get("sensitivity_radius") is None)
(mainmod.SESS_DIR / f"{old_sid}.json").unlink(missing_ok=True)
mainmod._SESSIONS.pop(old_sid, None)

r = client.get("/api/health")
check("health", r.json()["status"] == "ok")

print()
if fails:
    print(f"{len(fails)} FAILURES: {fails}")
    sys.exit(1)
print("ALL API TESTS PASSED")
