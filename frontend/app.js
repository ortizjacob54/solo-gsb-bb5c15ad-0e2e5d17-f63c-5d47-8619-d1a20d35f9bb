/* Planar photo ruler calibration — frontend.
 *
 * Coordinate systems:
 *   image  : raw photo pixels (u, v), the single source of truth for geometry
 *   world  : real-plane coords (x, y) after applying the 3x3 homography H
 *   screen : CSS pixels of the canvas element (devicePixelRatio handled here)
 *
 * Every stored point (corners, polygon vertices) is in image coordinates, so
 * zoom / pan / resize never change measurement geometry.
 */
"use strict";

/* ============================================================ geometry === */
const GEO = {
  DUP_TOL: 1e-7,
  COLINEAR_TOL: 1e-9,
  MIN_PIVOT: 1e-9,
  RESIDUAL_TOL: 1e-7,
  H_DET_TOL: 1e-8,
  DENOM_TOL: 1e-9,
  RING_TOL: 1e-9,
};

const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

function findDuplicate(pts) {
  let diag = 0;
  for (let i = 0; i < pts.length; i++)
    for (let j = i + 1; j < pts.length; j++)
      diag = Math.max(diag, dist(pts[i], pts[j]));
  const ref = Math.max(diag, 1);
  for (let i = 0; i < pts.length; i++)
    for (let j = i + 1; j < pts.length; j++)
      if (dist(pts[i], pts[j]) < GEO.DUP_TOL * ref) return [i, j];
  return null;
}

function convexOrderedQuad(pts) {
  const dup = findDuplicate(pts);
  if (dup) return { ok: false, error: `角点 ${dup[0] + 1} 与角点 ${dup[1] + 1} 重合：四点必须彼此分离` };
  const crosses = [];
  let edgeScale = 0;
  for (let i = 0; i < 4; i++) {
    const a = pts[i], b = pts[(i + 1) % 4], c = pts[(i + 2) % 4];
    crosses.push((b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x));
    edgeScale = Math.max(edgeScale, dist(a, b));
  }
  edgeScale *= edgeScale;
  if (crosses.some((cr) => Math.abs(cr) < GEO.COLINEAR_TOL * Math.max(edgeScale, 1)))
    return { ok: false, error: "有三个或更多角点近似共线，四边形已退化" };
  if (!(crosses.every((c) => c > 0) || crosses.every((c) => c < 0)))
    return { ok: false, error: "四个角点没有构成凸四边形（顺序错乱或自交），请按 左上→右上→右下→左下 依次点击" };
  return { ok: true };
}

/** 8-point DLT homography (mirror of backend solve_homography). */
function solveHomography(imgPts, W, H) {
  const errors = [];
  if (!(isFinite(W) && isFinite(H) && W > 0 && H > 0))
    errors.push("已知矩形的宽和高必须为正数");
  const conv = convexOrderedQuad(imgPts);
  if (!conv.ok) errors.push(conv.error);
  if (errors.length) return { ok: false, errors };

  const worldPts = [
    { x: 0, y: 0 }, { x: W, y: 0 }, { x: W, y: H }, { x: 0, y: H },
  ];

  // Normalize both point sets (isotropic scaling).
  const norm = (pts) => {
    const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
    const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
    let d = 0;
    for (const p of pts) d += Math.hypot(p.x - cx, p.y - cy);
    d /= pts.length;
    if (d < 1e-12) return { T: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], s: 1 };
    const s = Math.SQRT2 / d;
    return { T: [[s, 0, -s * cx], [0, s, -s * cy], [0, 0, 1]], s };
  };
  const Ts = norm(imgPts), Td = norm(worldPts);
  const apply = (T, p) => ({ x: T[0][0] * p.x + T[0][1] * p.y + T[0][2],
                              y: T[1][0] * p.x + T[1][1] * p.y + T[1][2] });
  const ns = imgPts.map((p) => apply(Ts.T, p));
  const nd = worldPts.map((p) => apply(Td.T, p));

  const Arows = [];
  for (let i = 0; i < 4; i++) {
    const { x, y } = ns[i], { x: X, y: Y } = nd[i];
    Arows.push([-x, -y, -1, 0, 0, 0, X * x, X * y, X]);
    Arows.push([0, 0, 0, -x, -y, -1, Y * x, Y * y, Y]);
  }
  // Smallest singular vector of A (8x9): rank-revealing Gauss-Jordan on AᵀA
  // with complete pivoting.  The last pivoted column is the free (null) one.
  const AtA = Array.from({ length: 9 }, () => new Array(9).fill(0));
  for (const row of Arows)
    for (let i = 0; i < 9; i++)
      for (let j = 0; j < 9; j++) AtA[i][j] += row[i] * row[j];

  const M = AtA.map((r) => r.slice());
  const perm = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  let pivotMag = 0;
  for (let k = 0; k < 8; k++) {
    let pr = k, pc = k, best = 0;
    for (let i = k; i < 9; i++)
      for (let j = k; j < 9; j++)
        if (Math.abs(M[i][j]) > best) { best = Math.abs(M[i][j]); pr = i; pc = j; }
    if (k === 0) pivotMag = best;
    // A genuine rank drop here (two near-null directions) means the four
    // point pairs do not determine a unique homography.
    if (best < GEO.MIN_PIVOT * (pivotMag || 1))
      return { ok: false, errors: ["透视变换不可逆：标定四点共线或构成退化四边形，无法从平面照片恢复唯一映射"] };
    if (pr !== k) [M[k], M[pr]] = [M[pr], M[k]];
    if (pc !== k) {
      M.forEach((row) => { const t = row[k]; row[k] = row[pc]; row[pc] = t; });
      [perm[k], perm[pc]] = [perm[pc], perm[k]];
    }
    // Eliminate column k in every other row, then normalize pivot row.
    for (let i = 0; i < 9; i++) {
      if (i === k || M[i][k] === 0) continue;
      const f = M[i][k] / M[k][k];
      M[i][k] = 0;
      for (let j = k + 1; j < 9; j++) M[i][j] -= f * M[k][j];
    }
    const pv = M[k][k];
    for (let j = k + 1; j < 9; j++) M[k][j] /= pv;
    M[k][k] = 1;
  }

  const hn = new Array(9);
  for (let k = 0; k < 8; k++) hn[perm[k]] = -M[k][8];
  hn[perm[8]] = 1;

  // Null-vector quality: |A h| / (|A| |h|) must be tiny.  This residual — not
  // the null eigen value (which is supposed to be ~0) — decides singularity.
  let maxA = 0;
  for (const row of Arows) maxA = Math.max(maxA, ...row.map(Math.abs));
  let maxRes = 0, normH = 0;
  for (const row of Arows) {
    let s = 0;
    for (let i = 0; i < 9; i++) s += row[i] * hn[i];
    maxRes = Math.max(maxRes, Math.abs(s));
  }
  for (const v of hn) normH += v * v;
  if (maxRes > GEO.MIN_PIVOT * maxA * Math.sqrt(normH))
    return { ok: false, errors: ["透视变换不可逆：标定四点共线或构成退化四边形，无法从平面照片恢复唯一映射"] };
  const h3x3 = [
    [hn[0], hn[1], hn[2]],
    [hn[3], hn[4], hn[5]],
    [hn[6], hn[7], hn[8]],
  ];
  // Denormalize: H = Td⁻¹ h Ts ; multiply by hand (3x3).
  const inv3 = (T) => {
    const [[a, b, tx], [, d, ty]] = T;
    return [[1 / a, 0, -tx / a], [0, 1 / d, -ty / d], [0, 0, 1]];
  };
  const mul3 = (A, B) => {
    const R = Array.from({ length: 3 }, () => [0, 0, 0]);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        for (let k = 0; k < 3; k++) R[i][j] += A[i][k] * B[k][j];
    return R;
  };
  let Hmat = mul3(inv3(Td.T), mul3(h3x3, Ts.T));
  const w00 = Hmat[2][2] || 1;
  Hmat = Hmat.map((row) => row.map((v) => v / w00));

  // DLT residual against the actual correspondences.
  let res = 0;
  for (let i = 0; i < 4; i++) {
    const p = imgPts[i], q = worldPts[i];
    const w = Hmat[2][0] * p.x + Hmat[2][1] * p.y + Hmat[2][2];
    const x = (Hmat[0][0] * p.x + Hmat[0][1] * p.y + Hmat[0][2]) / w;
    const y = (Hmat[1][0] * p.x + Hmat[1][1] * p.y + Hmat[1][2]) / w;
    res = Math.max(res, Math.hypot(x - q.x, y - q.y));
  }
  if (res > GEO.RESIDUAL_TOL * Math.max(W, H))
    return { ok: false, errors: [`透视校准残差过大（${res.toPrecision(3)} 单位），角点与已知矩形对应关系不一致，请检查点击顺序`] };

  // Determinant guard.
  const det =
    Hmat[0][0] * (Hmat[1][1] * Hmat[2][2] - Hmat[1][2] * Hmat[2][1]) -
    Hmat[0][1] * (Hmat[1][0] * Hmat[2][2] - Hmat[1][2] * Hmat[2][0]) +
    Hmat[0][2] * (Hmat[1][0] * Hmat[2][1] - Hmat[1][1] * Hmat[2][0]);
  let hNorm = 0;
  Hmat.forEach((r) => r.forEach((v) => (hNorm += v * v)));
  if (Math.abs(det) < GEO.H_DET_TOL * Math.max(Math.pow(hNorm, 1.5), 1e-30))
    return { ok: false, errors: ["透视矩阵行列式接近 0，变换不可逆：四点几乎共线或矩形在照片中退化"] };

  const row3Scale = Math.max(Math.hypot(Hmat[2][0], Hmat[2][1]), Math.abs(Hmat[2][2]));
  for (let i = 0; i < 4; i++) {
    const p = imgPts[i];
    const w = Math.abs(Hmat[2][0] * p.x + Hmat[2][1] * p.y + Hmat[2][2]);
    if (w < GEO.DENOM_TOL * row3Scale)
      return { ok: false, errors: [`角点 ${i + 1} 映射时分母接近 0（落在无穷远线上），此视角下透视不可求逆`] };
  }
  return { ok: true, H: Hmat };
}

function applyH(H, p) {
  const w = H[2][0] * p.x + H[2][1] * p.y + H[2][2];
  return {
    x: (H[0][0] * p.x + H[0][1] * p.y + H[0][2]) / w,
    y: (H[1][0] * p.x + H[1][1] * p.y + H[1][2]) / w,
    w,
  };
}

function segmentsCross(p, q, a, b) {
  const orient = (r, s, t) => (s.x - r.x) * (t.y - r.y) - (s.y - r.y) * (t.x - r.x);
  const o1 = orient(p, q, a), o2 = orient(p, q, b);
  const o3 = orient(a, b, p), o4 = orient(a, b, q);
  const tol = 1e-10 * Math.max(dist(p, q), dist(a, b), 1);
  if ([o1, o2, o3, o4].every((o) => Math.abs(o) < tol)) return false; // collinear: rare; endpoints handled by dup check
  return ((o1 > tol && o2 < -tol) || (o1 < -tol && o2 > tol)) &&
         ((o3 > tol && o4 < -tol) || (o3 < -tol && o4 > tol));
}

function polygonSelfIntersects(pts) {
  if (pts.length < 3) return { hit: false };
  const dup = findDuplicate(pts);
  if (dup) return { hit: true, error: `顶点 ${dup[0] + 1} 与顶点 ${dup[1] + 1} 重合` };
  const n = pts.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (j === i || (j + 1) % n === i || (i + 1) % n === j) continue;
      if (segmentsCross(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]))
        return { hit: true, error: `边 ${i + 1} 与边 ${j + 1} 相交` };
    }
  }
  return { hit: false };
}

function measurePolygon(H, imgPts) {
  return measureRegion(H, imgPts, []);
}

/* ------------------------------------------------------- ring regions --- */
function ringAbsArea(pts) {
  let s = 0;
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    s += a.x * b.y - b.x * a.y;
  }
  return Math.abs(s) / 2;
}

function ringPerimeter(pts) {
  let p = 0;
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    p += Math.hypot(b.x - a.x, b.y - a.y);
  }
  return p;
}

function pointSegmentDistance(p, a, b) {
  const abx = b.x - a.x, aby = b.y - a.y;
  const den = abx * abx + aby * aby;
  if (den < 1e-12) return dist(p, a);
  let t = ((p.x - a.x) * abx + (p.y - a.y) * aby) / den;
  t = Math.min(1, Math.max(0, t));
  return dist(p, { x: a.x + abx * t, y: a.y + aby * t });
}

/** Classify two closed segments: "cross" (intersect/touch/overlap),
 *  "near" (disjoint but within tol), "far". Mirrors backend _segment_relation. */
function segmentRelation(p, q, a, b, tol) {
  const orient = (r, s, t) => (s.x - r.x) * (t.y - r.y) - (s.y - r.y) * (t.x - r.x);
  const o1 = orient(p, q, a), o2 = orient(p, q, b);
  const o3 = orient(a, b, p), o4 = orient(a, b, q);
  if ([o1, o2, o3, o4].every((o) => Math.abs(o) <= tol)) {
    const onSeg = (x, r, s) =>
      x.x >= Math.min(r.x, s.x) - tol && x.x <= Math.max(r.x, s.x) + tol &&
      x.y >= Math.min(r.y, s.y) - tol && x.y <= Math.max(r.y, s.y) + tol;
    if ([p, q, a, b].some((z) => onSeg(z, p, q) && onSeg(z, a, b)))
      return ["cross", 0];
  }
  if (((o1 > tol && o2 < -tol) || (o1 < -tol && o2 > tol)) &&
      ((o3 > tol && o4 < -tol) || (o3 < -tol && o4 > tol)))
    return ["cross", 0];

  // Endpoint-on-segment contact (tangency through a vertex): perpendicular
  // distance within tol with the projection actually on the segment.
  const onSegment = (z, r, s) => {
    const edgeLen = dist(r, s);
    if (edgeLen < 1e-12) return dist(z, r) <= tol;
    const cr = Math.abs((s.x - r.x) * (z.y - r.y) - (s.y - r.y) * (z.x - r.x));
    if (cr > tol * edgeLen) return false;
    return z.x >= Math.min(r.x, s.x) - tol && z.x <= Math.max(r.x, s.x) + tol &&
           z.y >= Math.min(r.y, s.y) - tol && z.y <= Math.max(r.y, s.y) + tol;
  };
  if (onSegment(a, p, q) || onSegment(b, p, q) ||
      onSegment(p, a, b) || onSegment(q, a, b))
    return ["cross", 0];
  const d = Math.min(
    pointSegmentDistance(a, p, q), pointSegmentDistance(b, p, q),
    pointSegmentDistance(p, a, b), pointSegmentDistance(q, a, b)
  );
  return [d <= tol ? "near" : "far", d];
}

function pointInRingStrict(p, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const a = ring[i], b = ring[j];
    if ((a.y > p.y) !== (b.y > p.y) &&
        p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x)
      inside = !inside;
  }
  return inside;
}

function ringScale(ring) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of ring) {
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
  }
  return Math.max(Math.hypot(maxX - minX, maxY - minY), 1);
}

function ringLabel(ri) { return ri === 0 ? "外环" : `孔洞 ${ri}`; }

/** Validate outer ring + holes in corrected real-plane coordinates. */
function validateRingTopology(rings) {
  if (!rings.length) return [];
  const errors = [];
  for (let ri = 0; ri < rings.length; ri++) {
    if (rings[ri].length < 3) { errors.push(`${ringLabel(ri)}至少需要 3 个顶点`); continue; }
    const si = polygonSelfIntersects(rings[ri]);
    if (si.hit) errors.push(`${ringLabel(ri)}自交（${si.error}），请调整顶点`);
  }
  if (rings.some((r) => r.length < 3)) return errors;

  const outer = rings[0];

  // outer ring vs each hole
  for (let hi = 1; hi < rings.length; hi++) {
    const hole = rings[hi];
    const tol = GEO.RING_TOL * Math.max(ringScale(outer), ringScale(hole));
    let blocked = false, nearD = null;
    for (let i = 0; i < outer.length && !blocked; i++) {
      const p = outer[i], q = outer[(i + 1) % outer.length];
      for (let j = 0; j < hole.length; j++) {
        const a = hole[j], b = hole[(j + 1) % hole.length];
        const [kind, d] = segmentRelation(p, q, a, b, tol);
        if (kind === "cross") {
          errors.push(`孔洞 ${hi} 与外环相交或相切（外环边 ${i + 1} 与孔洞边 ${j + 1}），孔洞必须完全位于外环内且与外环边界严格分离`);
          blocked = true;
          break;
        }
        if (kind === "near" && nearD === null) nearD = d;
      }
    }
    if (blocked) continue;
    if (nearD !== null) {
      errors.push(`孔洞 ${hi} 与外环距离过近（${nearD.toPrecision(3)}，判定为相切/越界），请将孔洞移入外环内部并与外边界保持距离`);
      continue;
    }
    const badK = hole.findIndex((z) => !pointInRingStrict(z, outer));
    if (badK >= 0) errors.push(`孔洞 ${hi} 没有完全位于外环内（孔洞顶点 ${badK + 1} 位于外环之外）`);
  }

  // hole vs hole
  for (let h1 = 1; h1 < rings.length; h1++) {
    for (let h2 = h1 + 1; h2 < rings.length; h2++) {
      const r1 = rings[h1], r2 = rings[h2];
      const tol = GEO.RING_TOL * Math.max(ringScale(r1), ringScale(r2));
      let blocked = false, nearD = null;
      for (let i = 0; i < r1.length && !blocked; i++) {
        const p = r1[i], q = r1[(i + 1) % r1.length];
        for (let j = 0; j < r2.length; j++) {
          const a = r2[j], b = r2[(j + 1) % r2.length];
          const [kind, d] = segmentRelation(p, q, a, b, tol);
          if (kind === "cross") {
            errors.push(`孔洞 ${h1} 与孔洞 ${h2} 相交或相切（边 ${i + 1} 与边 ${j + 1}），孔洞之间必须严格分离`);
            blocked = true;
            break;
          }
          if (kind === "near" && nearD === null) nearD = d;
        }
      }
      if (blocked) continue;
      if (nearD !== null) {
        errors.push(`孔洞 ${h1} 与孔洞 ${h2} 距离过近（${nearD.toPrecision(3)}，判定为相切），请将两个孔洞分开`);
        continue;
      }
      if (pointInRingStrict(r2[0], r1)) errors.push(`孔洞 ${h2} 位于孔洞 ${h1} 内部（孔洞不允许相互包含）`);
      else if (pointInRingStrict(r1[0], r2)) errors.push(`孔洞 ${h1} 位于孔洞 ${h2} 内部（孔洞不允许相互包含）`);
    }
  }
  return errors;
}

function transformRing(H, imgPts, label) {
  const row3Scale = Math.max(Math.hypot(H[2][0], H[2][1]), Math.abs(H[2][2]));
  const world = [];
  for (let i = 0; i < imgPts.length; i++) {
    const { x, y, w } = applyH(H, imgPts[i]);
    if (Math.abs(w) < GEO.DENOM_TOL * row3Scale)
      return { error: `${label}顶点 ${i + 1} 的透视分母接近 0（位于无穷远线上），无法计算校正后坐标` };
    world.push({ x, y });
  }
  return { world };
}

/** Measure a region (outer ring + holes) in corrected real-plane coordinates. */
function measureRegion(H, outerImg, holesImg) {
  holesImg = holesImg || [];
  if (outerImg.length < 3) return { valid: false, errors: ["多边形至少需要 3 个顶点"] };
  const rings = [];
  const t0 = transformRing(H, outerImg, "多边形");
  if (t0.error) return { valid: false, errors: [t0.error] };
  rings.push(t0.world);
  for (let hi = 0; hi < holesImg.length; hi++) {
    if (holesImg[hi].length < 3) return { valid: false, errors: [`孔洞 ${hi + 1} 至少需要 3 个顶点`] };
    const t = transformRing(H, holesImg[hi], `孔洞 ${hi + 1} `);
    if (t.error) return { valid: false, errors: [t.error] };
    rings.push(t.world);
  }
  const topo = validateRingTopology(rings);
  if (topo.length) return { valid: false, errors: topo };

  const outer = rings[0];
  const gross = ringAbsArea(outer);
  let perimeter = ringPerimeter(outer);
  const holeAreas = [], holePerimeters = [];
  for (let k = 1; k < rings.length; k++) {
    holeAreas.push(ringAbsArea(rings[k]));
    holePerimeters.push(ringPerimeter(rings[k]));
    perimeter += holePerimeters[k - 1];
  }
  const area = gross - holeAreas.reduce((s, a) => s + a, 0);
  return {
    valid: true, area, grossArea: gross, perimeter, worldPoints: outer,
    holeAreas, holePerimeters, holeWorldPoints: rings.slice(1),
  };
}

/* ============================================================ state ====== */
const els = {};
document.querySelectorAll("[id]").forEach((e) => (els[e.id] = e));

const UNITS = ["mm", "cm", "m", "in", "px"];
const UNIT_TO_MM = { mm: 1, cm: 10, m: 1000, in: 25.4, px: null };

const state = {
  sessionId: null,
  image: null,            // {filename, width, height}
  imgEl: null,
  corners: [],            // [{x,y}] image coords
  rectW: null,
  rectH: null,
  unit: "mm",
  H: null,
  calibValid: false,
  calibErrors: [],
  polygons: [],           // {id,name,points,holes:[{points}],area,...,holeAreas,lastValid}
  selected: null,         // polygon id
  draft: [],              // in-progress outer polygon points (image coords)
  holeDraft: [],          // in-progress hole points (image coords)
  holeTarget: null,       // polygon id receiving the hole being drawn
  mode: "pan",            // calib | draw | hole | edit | pan
  displayUnit: "mm",
  sensRadius: 1,          // 角点统一像素误差半径（敏感性评估输入）
  sensitivity: null,      // {fp, radius, data} 最近一次敏感性评估结果
  sensBusy: false,
  view: { scale: 1, x: 0, y: 0 }, // screen = image*scale + (x,y)
  hoverImg: null,
  dragging: null,         // drag descriptor
  spaceDown: false,
};

const COLORS = ["#4da3ff", "#3ddc97", "#ffc24d", "#ff8f6b", "#c792ea",
  "#7cddff", "#a6e22e", "#ff6b6b", "#f4a261", "#8de0c2"];
const polyColor = (i) => COLORS[i % COLORS.length];

const $canvas = els.canvas;
const ctx = $canvas.getContext("2d");

/* ============================================================ API ======== */
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: opts.body ? { "Content-Type": "application/json" } : {},
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    let data = null;
    try { data = await res.json(); detail = data.detail || detail; } catch { /* ignore */ }
    const err = new Error(detail);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return res.json();
}

function toast(msg, isError = false) {
  const t = els.toast || (() => {
    const d = document.createElement("div");
    d.className = "toast";
    d.id = "toast";
    document.body.appendChild(d);
    els.toast = d;
    return d;
  })();
  t.textContent = msg;
  t.classList.toggle("error", isError);
  t.classList.add("show");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove("show"), 2600);
}

/* ====================================================== view transforms == */
function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const r = $canvas.parentElement.getBoundingClientRect();
  $canvas.width = Math.max(1, Math.round(r.width * dpr));
  $canvas.height = Math.max(1, Math.round(r.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  render();
}
const screenSize = () => {
  const dpr = window.devicePixelRatio || 1;
  return { w: $canvas.width / dpr, h: $canvas.height / dpr };
};
const imgToScreen = (p) => ({ x: p.x * state.view.scale + state.view.x, y: p.y * state.view.scale + state.view.y });
const screenToImg = (p) => ({ x: (p.x - state.view.x) / state.view.scale, y: (p.y - state.view.y) / state.view.scale });

function clampImagePoint(p) {
  if (!state.image) return p;
  return {
    x: Math.min(Math.max(p.x, 0), state.image.width),
    y: Math.min(Math.max(p.y, 0), state.image.height),
  };
}

function fitView() {
  if (!state.image) return;
  const { w, h } = screenSize();
  const pad = 36;
  const s = Math.min((w - pad * 2) / state.image.width, (h - pad * 2) / state.image.height);
  state.view.scale = Math.min(s, 8);
  state.view.x = (w - state.image.width * state.view.scale) / 2;
  state.view.y = (h - state.image.height * state.view.scale) / 2;
  render();
}

function zoomAt(screenPt, factor) {
  if (!state.image) return;
  const before = screenToImg(screenPt);
  state.view.scale = Math.min(40, Math.max(0.02, state.view.scale * factor));
  state.view.x = screenPt.x - before.x * state.view.scale;
  state.view.y = screenPt.y - before.y * state.view.scale;
  updateZoomLabel();
  render();
}

/* ============================================================ rendering == */
const HANDLE_PX = 6;
const cornerColor = (i) => ["#ffc24d", "#4da3ff", "#3ddc97", "#ff8f6b"][i % 4];
const CORNER_NAMES = ["左上", "右上", "右下", "左下"];

function drawHandle(sp, color, filled = true, radius = HANDLE_PX) {
  ctx.beginPath();
  ctx.arc(sp.x, sp.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = filled ? color : "#0f141b";
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = color;
  ctx.stroke();
}

function drawDiamond(sp, color, filled = true, radius = HANDLE_PX) {
  ctx.beginPath();
  ctx.moveTo(sp.x, sp.y - radius);
  ctx.lineTo(sp.x + radius, sp.y);
  ctx.lineTo(sp.x, sp.y + radius);
  ctx.lineTo(sp.x - radius, sp.y);
  ctx.closePath();
  ctx.fillStyle = filled ? color : "#0f141b";
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = color;
  ctx.stroke();
}

function drawDashedLine(a, b, color) {
  ctx.save();
  ctx.setLineDash([6, 5]);
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.restore();
}

function render() {
  const { w, h } = screenSize();
  ctx.clearRect(0, 0, w, h);
  updateZoomLabel();
  if (!state.imgEl || !state.image) return;

  // image
  const tl = imgToScreen({ x: 0, y: 0 });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(
    state.imgEl,
    tl.x, tl.y,
    state.image.width * state.view.scale,
    state.image.height * state.view.scale
  );
  // image border
  ctx.strokeStyle = "rgba(255,255,255,.18)";
  ctx.lineWidth = 1;
  ctx.strokeRect(tl.x, tl.y, state.image.width * state.view.scale, state.image.height * state.view.scale);

  // calibration quad
  if (state.corners.length > 0) {
    const scr = state.corners.map(imgToScreen);
    if (state.corners.length >= 2) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(scr[0].x, scr[0].y);
      for (let i = 1; i < scr.length; i++) ctx.lineTo(scr[i].x, scr[i].y);
      if (state.corners.length === 4) ctx.closePath();
      ctx.strokeStyle = state.calibValid ? "rgba(61,220,151,.95)" : "rgba(255,194,77,.95)";
      ctx.lineWidth = 2;
      ctx.stroke();
      if (state.corners.length === 4) {
        ctx.fillStyle = state.calibValid ? "rgba(61,220,151,.10)" : "rgba(255,194,77,.08)";
        ctx.fill();
      }
      ctx.restore();
    }
    state.corners.forEach((p, i) => {
      const sp = scr[i];
      drawHandle(sp, cornerColor(i), true, 8);
      ctx.font = "700 10px ui-monospace, monospace";
      ctx.fillStyle = "#0f141b";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(i + 1), sp.x, sp.y + 0.5);
    });
    // rubber line to next corner
    if (state.mode === "calib" && state.corners.length < 4 && state.hoverImg) {
      const inside = pointInImage(state.hoverImg);
      if (inside) drawDashedLine(scr[scr.length - 1], imgToScreen(state.hoverImg), "rgba(255,194,77,.7)");
    }
  }

  // polygons (outer ring + holes)
  state.polygons.forEach((poly, i) => {
    const color = polyColor(i);
    const selected = poly.id === state.selected;
    const scrOuter = poly.points.map(imgToScreen);

    // Composite fill path using even-odd so holes are punched out.
    ctx.beginPath();
    ctx.moveTo(scrOuter[0].x, scrOuter[0].y);
    for (let k = 1; k < scrOuter.length; k++) ctx.lineTo(scrOuter[k].x, scrOuter[k].y);
    ctx.closePath();
    poly.holes.forEach((hole) => {
      const sh = hole.points.map(imgToScreen);
      ctx.moveTo(sh[0].x, sh[0].y);
      for (let k = 1; k < sh.length; k++) ctx.lineTo(sh[k].x, sh[k].y);
      ctx.closePath();
    });
    ctx.fillStyle = poly.valid ? color + "26" : "rgba(255,107,107,.13)";
    ctx.fill("evenodd");

    // strokes
    const outerColor = poly.valid ? color : "#ff6b6b";
    const holeTarget = state.mode === "hole" && state.holeTarget === poly.id;
    ctx.beginPath();
    ctx.moveTo(scrOuter[0].x, scrOuter[0].y);
    for (let k = 1; k < scrOuter.length; k++) ctx.lineTo(scrOuter[k].x, scrOuter[k].y);
    ctx.closePath();
    ctx.lineWidth = selected || holeTarget ? 2.6 : 1.8;
    ctx.strokeStyle = outerColor;
    ctx.stroke();

    poly.holes.forEach((hole, hk) => {
      const sh = hole.points.map(imgToScreen);
      const hValid = poly.valid;
      ctx.beginPath();
      ctx.moveTo(sh[0].x, sh[0].y);
      for (let k = 1; k < sh.length; k++) ctx.lineTo(sh[k].x, sh[k].y);
      ctx.closePath();
      ctx.save();
      ctx.setLineDash([7, 4]);
      ctx.lineWidth = selected || holeTarget ? 2.2 : 1.5;
      ctx.strokeStyle = hValid ? color : "#ff6b6b";
      ctx.stroke();
      ctx.restore();
      if (hValid) {
        ctx.fillStyle = "rgba(15,20,27,.82)";
        const hx = sh.reduce((s, p) => s + p.x, 0) / sh.length;
        const hy = sh.reduce((s, p) => s + p.y, 0) / sh.length;
        ctx.beginPath(); ctx.arc(hx, hy, 8, 0, Math.PI * 2); ctx.fill();
        ctx.font = "700 9px ui-monospace, monospace";
        ctx.fillStyle = color;
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(String(hk + 1), hx, hy + 0.5);
      }
    });

    if (state.mode === "edit" || selected) {
      poly.points.forEach((p, k) => {
        const active = k === state.dragging?.vertex && state.dragging?.polyId === poly.id && state.dragging?.ring === 0;
        drawHandle(imgToScreen(p), poly.valid ? color : "#ff6b6b", !active);
      });
      poly.holes.forEach((hole, hk) => {
        hole.points.forEach((p, k) => {
          const active = k === state.dragging?.vertex && state.dragging?.polyId === poly.id && state.dragging?.ring === hk + 1;
          drawDiamond(imgToScreen(p), poly.valid ? color : "#ff6b6b", !active);
        });
      });
    }
    // label at outer-ring centroid
    const cx = scrOuter.reduce((s, p) => s + p.x, 0) / scrOuter.length;
    const cy = scrOuter.reduce((s, p) => s + p.y, 0) / scrOuter.length;
    ctx.font = "600 11px -apple-system, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "rgba(0,0,0,.65)";
    const label = poly.name;
    const tw = ctx.measureText(label).width;
    ctx.fillRect(cx - tw / 2 - 6, cy - 9, tw + 12, 18);
    ctx.fillStyle = color;
    ctx.fillText(label, cx, cy + .5);
  });

  // hole draft (inside the targeted polygon)
  if (state.mode === "hole" && state.holeDraft.length > 0) {
    const scr = state.holeDraft.map(imgToScreen);
    ctx.beginPath();
    ctx.moveTo(scr[0].x, scr[0].y);
    for (let k = 1; k < scr.length; k++) ctx.lineTo(scr[k].x, scr[k].y);
    if (state.hoverImg && pointInImage(state.hoverImg))
      ctx.lineTo(imgToScreen(state.hoverImg).x, imgToScreen(state.hoverImg).y);
    ctx.strokeStyle = "rgba(255,205,120,.95)";
    ctx.lineWidth = 1.6;
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.restore();
    if (state.holeDraft.length >= 2 && state.hoverImg && pointInImage(state.hoverImg))
      drawDashedLine(imgToScreen(state.hoverImg), scr[0], "rgba(255,205,120,.5)");
    state.holeDraft.forEach((p, k) => {
      const sp = imgToScreen(p);
      if (k === 0 && state.hoverImg && dist(state.hoverImg, p) * state.view.scale < 12 && state.holeDraft.length >= 3)
        drawDiamond(sp, "#3ddc97", true, 9);
      else drawDiamond(sp, "#ffcd78");
    });
  }

  // draft polygon
  if (state.draft.length > 0) {
    const scr = state.draft.map(imgToScreen);
    ctx.beginPath();
    ctx.moveTo(scr[0].x, scr[0].y);
    for (let k = 1; k < scr.length; k++) ctx.lineTo(scr[k].x, scr[k].y);
    if (state.hoverImg && pointInImage(state.hoverImg))
      ctx.lineTo(imgToScreen(state.hoverImg).x, imgToScreen(state.hoverImg).y);
    ctx.strokeStyle = "rgba(255,255,255,.85)";
    ctx.lineWidth = 1.6;
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.stroke();
    ctx.restore();
    // close hint line back to first point
    if (state.draft.length >= 2 && state.hoverImg && pointInImage(state.hoverImg))
      drawDashedLine(imgToScreen(state.hoverImg), scr[0], "rgba(255,255,255,.45)");
    state.draft.forEach((p, k) => {
      const sp = imgToScreen(p);
      // highlight first vertex when close enough to close
      if (k === 0 && state.hoverImg && dist(state.hoverImg, p) * state.view.scale < 12 && state.draft.length >= 3)
        drawHandle(sp, "#3ddc97", true, 8);
      else drawHandle(sp, "#ffffff");
    });
  }
}

const pointInImage = (p) =>
  state.image && p.x >= 0 && p.y >= 0 && p.x <= state.image.width && p.y <= state.image.height;

/* ====================================================== sidebar rendering */
function fmtNum(v) {
  if (v == null || !isFinite(v)) return "—";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e7 || a < 1e-4) return v.toExponential(3);
  return String(Math.round(v * 1e4) / 1e4);
}

function convertValue(v, fromUnit, toUnit, isArea) {
  if (v == null) return null;
  if (fromUnit === toUnit) return v;
  const f = UNIT_TO_MM[fromUnit], t = UNIT_TO_MM[toUnit];
  if (f == null || t == null) return null; // px is only comparable to itself
  const linear = f / t;
  return v * (isArea ? linear * linear : linear);
}

function renderCorners() {
  const ol = els.cornerList;
  ol.innerHTML = "";
  for (let i = 0; i < 4; i++) {
    const li = document.createElement("li");
    const p = state.corners[i];
    if (p) li.classList.add("placed");
    if (state.mode === "calib" && i === state.corners.length) li.classList.add("active");
    li.innerHTML = `<span class="corner-dot" style="background:${cornerColor(i)}"></span>` +
      `<span>${CORNER_NAMES[i]}</span>` +
      `<span style="margin-left:auto">${p ? `${Math.round(p.x)},${Math.round(p.y)}` : "—"}</span>`;
    ol.appendChild(li);
  }
  const st = els.calibStatus;
  if (!state.image) {
    st.innerHTML = `<span class="dot dot-idle"></span><span>先上传照片，再依次点击矩形四角</span>`;
  } else if (state.corners.length < 4) {
    st.innerHTML = `<span class="dot dot-idle"></span><span>已标记 ${state.corners.length}/4 —— 下一点：${CORNER_NAMES[state.corners.length] || ""}</span>`;
  } else if (state.calibValid) {
    st.innerHTML = `<span class="dot dot-ok"></span><span>标定有效：${fmtNum(state.rectW)} × ${fmtNum(state.rectH)} ${state.unit} 的矩形已对齐，可绘制测量区域</span>`;
  } else {
    st.innerHTML = `<span class="dot dot-err"></span><span>${state.calibErrors[0] || "标定无效"}</span>`;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderPolyList() {
  dropStaleSensitivity(); // 角点/半径/区域已变化 -> 旧评估自动作废
  const list = els.polyList;
  list.innerHTML = "";
  if (state.polygons.length === 0) {
    list.innerHTML = `<p class="empty-text">完成标定并绘制闭合多边形后，这里实时显示校正后的面积与周长；可在外环内继续添加镂空孔洞。</p>`;
    return;
  }
  state.polygons.forEach((poly, i) => {
    const color = polyColor(i);
    const card = document.createElement("div");
    const invalidNow = !poly.valid;
    const lv = poly.lastValid;
    const showLast = invalidNow && lv && lv.valid;
    card.className = "poly-card" + (poly.id === state.selected ? " selected" : "") + (invalidNow ? " invalid" : "");
    const areaConv = convertValue(poly.area, state.unit, state.displayUnit, true);
    const perConv = convertValue(poly.perimeter, state.unit, state.displayUnit, false);
    const lvArea = showLast ? convertValue(lv.area, state.unit, state.displayUnit, true) : null;
    const lvPer = showLast ? convertValue(lv.perimeter, state.unit, state.displayUnit, false) : null;
    const grossConv = poly.valid && poly.grossArea != null
      ? convertValue(poly.grossArea, state.unit, state.displayUnit, true) : null;
    const areaUnit = state.displayUnit === "px" ? "px²" : `${state.displayUnit}²`;

    const holeRows = (poly.holes || []).map((hole, hk) => {
      const ha = poly.valid && poly.holeAreas ? poly.holeAreas[hk] : null;
      const hp = poly.valid && poly.holePerimeters ? poly.holePerimeters[hk] : null;
      const haConv = ha != null ? convertValue(ha, state.unit, state.displayUnit, true) : null;
      const hpConv = hp != null ? convertValue(hp, state.unit, state.displayUnit, false) : null;
      return `<div class="hole-row" data-holy="${poly.id}" data-hidx="${hk}">
        <span class="hole-badge" style="color:${color}">▰ 孔洞 ${hk + 1}</span>
        <span class="hole-stats">${haConv != null ? fmtNum(haConv) + " " + areaUnit : "—"} ·
          周长 ${hpConv != null ? fmtNum(hpConv) + " " + state.displayUnit : "—"} ·
          ${hole.points.length} 顶点</span>
        <button class="hole-del" data-holy="${poly.id}" data-hidx="${hk}" title="删除该孔洞">✕</button>
      </div>`;
    }).join("");

    card.innerHTML = `
      <div class="poly-card-head">
        <span class="poly-swatch" style="background:${color}"></span>
        <input type="text" value="${escapeHtml(poly.name)}" data-id="${poly.id}" class="poly-name" />
        <button class="poly-del" data-id="${poly.id}" title="删除该区域">🗑</button>
      </div>
      <div class="poly-metrics">
        <div class="metric"><span class="m-label">净面积（外环 − 孔洞）</span>
          <span class="m-value m-area ${poly.valid ? "" : "bad"}">${poly.valid ? fmtNum(areaConv) : "—"}</span>
          <span class="m-label m-unit">${areaUnit}</span></div>
        <div class="metric"><span class="m-label">总周长（外环 + 孔洞）</span>
          <span class="m-value m-per ${poly.valid ? "" : "bad"}">${poly.valid ? fmtNum(perConv) : "—"}</span>
          <span class="m-label m-unit m-unit-len">${state.displayUnit}</span></div>
      </div>
      ${grossConv != null && poly.holes.length
        ? `<div class="poly-meta m-gross">外环面积 ${fmtNum(grossConv)} ${areaUnit} · ${poly.holes.length} 个孔洞 · ${poly.points.length} 外环顶点</div>`
        : `<div class="poly-meta m-gross">${poly.points.length} 个外环顶点${poly.holes.length ? " · " + poly.holes.length + " 个孔洞" : ""}</div>`}
      ${sensBlockHtml(poly, areaUnit)}
      ${holeRows ? `<div class="hole-list">${holeRows}</div>` : ""}
      <div class="poly-actions">
        <button class="btn btn-ghost add-hole-btn" data-id="${poly.id}" title="在该区域外环内绘制新的镂空孔洞">▰ 添加孔洞</button>
      </div>
      ${showLast ? `
        <div class="last-valid">
          <span>服务端最后一次有效：净面积 <b>${fmtNum(lvArea)}</b> ${areaUnit} · 周长 <b>${fmtNum(lvPer)}</b> ${state.displayUnit}</span>
          <button class="btn btn-ghost revert-btn" data-id="${poly.id}">↩ 恢复到上次有效</button>
        </div>` : ""}
      ${invalidNow ? `<div class="poly-error">${(poly.errors || []).map(escapeHtml).join("<br>")}</div>` : ""}
    `;
    card.addEventListener("click", (e) => {
      const delHole = e.target.closest(".hole-del");
      if (delHole) { deleteHole(poly.id, Number(delHole.dataset.hidx)); return; }
      if (e.target.closest(".add-hole-btn")) {
        state.selected = poly.id;
        state.holeTarget = poly.id;
        startHoleMode();
        return;
      }
      if (e.target.closest(".revert-btn")) { revertPolygon(poly.id); return; }
      if (e.target.closest(".poly-del")) {
        deletePolygon(poly.id);
      } else if (e.target.closest(".poly-name")) {
        // allow editing; no selection change / mode switch
      } else {
        state.selected = poly.id;
        state.holeTarget = poly.id;
        setMode("edit");
      }
    });
    card.querySelector(".poly-name").addEventListener("change", (e) => {
      poly.name = e.target.value.trim() || poly.name;
      if (poly.valid) syncPolygon(poly);
    });
    list.appendChild(card);
  });
}

/** Update metric values inside existing cards without rebuilding the DOM
 *  (used while dragging, so the rename <input> keeps focus). */
function updatePolyCardMetrics() {
  // Dragging a vertex changes region geometry: stale sensitivity blocks go.
  const hadSens = state.sensitivity !== null;
  dropStaleSensitivity();
  const sensGone = hadSens && !state.sensitivity;
  const cards = els.polyList.querySelectorAll(".poly-card");
  cards.forEach((card, i) => {
    const poly = state.polygons[i];
    if (!poly) return;
    if (sensGone) card.querySelectorAll(".sens-info").forEach((el) => el.remove());
    const areaConv = convertValue(poly.area, state.unit, state.displayUnit, true);
    const perConv = convertValue(poly.perimeter, state.unit, state.displayUnit, false);
    const areaUnit = state.displayUnit === "px" ? "px²" : `${state.displayUnit}²`;
    const vArea = card.querySelector(".m-area");
    const vPer = card.querySelector(".m-per");
    if (vArea) {
      vArea.textContent = poly.valid ? fmtNum(areaConv) : "—";
      vArea.classList.toggle("bad", !poly.valid);
    }
    if (vPer) {
      vPer.textContent = poly.valid ? fmtNum(perConv) : "—";
      vPer.classList.toggle("bad", !poly.valid);
    }
    const units = card.querySelectorAll(".m-unit");
    if (units[0]) units[0].textContent = areaUnit;
    if (units[1]) units[1].textContent = state.displayUnit;

    // per-hole live stats
    card.querySelectorAll(".hole-row").forEach((row, hk) => {
      const stats = row.querySelector(".hole-stats");
      if (!stats) return;
      const ha = poly.valid && poly.holeAreas ? poly.holeAreas[hk] : null;
      const hp = poly.valid && poly.holePerimeters ? poly.holePerimeters[hk] : null;
      const haConv = ha != null ? convertValue(ha, state.unit, state.displayUnit, true) : null;
      const hpConv = hp != null ? convertValue(hp, state.unit, state.displayUnit, false) : null;
      stats.textContent = `${haConv != null ? fmtNum(haConv) + " " + areaUnit : "—"} · 周长 ${hpConv != null ? fmtNum(hpConv) + " " + state.displayUnit : "—"} · ${poly.holes[hk]?.points.length || 0} 顶点`;
    });
  });
}

function renderModeButtons() {
  els.modeCalib.classList.toggle("active", state.mode === "calib");
  els.modeDraw.classList.toggle("active", state.mode === "draw");
  els.modeHole.classList.toggle("active", state.mode === "hole");
  els.modeEdit.classList.toggle("active", state.mode === "edit");
  els.modePan.classList.toggle("active", state.mode === "pan");
  const panning = state.spaceDown || state.dragging?.type === "pan";
  $canvas.className = `mode-${state.mode}` + (panning ? " panning" : "");
  const labels = { calib: "角点标定", draw: "绘制外环", hole: "绘制镂空孔洞", edit: "编辑顶点", pan: "平移视图" };
  els.modeLabel.textContent = `模式：${labels[state.mode]}`;
  const target = state.polygons.find((p) => p.id === state.holeTarget);
  const hints = {
    calib: "单击放置/拖动角点 · 右键拖动平移 · 滚轮缩放",
    draw: "单击添加顶点 · 双击或 Enter 闭合 · Esc 取消 · Backspace 撤销",
    hole: target
      ? `正在为「${target.name}」添加孔洞：在其外环内部逐点单击，双击/Enter 闭合；孔洞不得与外环或其他孔洞相交/相切`
      : "先在结果列表或画布上选中一个区域，再为其绘制孔洞",
    edit: "拖动外环/孔洞顶点或整个区域 · 右键顶点删除 · Del 删除区域",
    pan: "拖动平移 · 滚轮缩放",
  };
  els.hintLabel.textContent = state.image ? hints[state.mode] : "";
}

function refreshUI() {
  renderCorners();
  renderPolyList();
  renderModeButtons();
  updateUnitOptions();
  refreshSensButton();
  render();
}

function updateUnitOptions() {
  const sel = els.displayUnit;
  if (sel.options.length !== UNITS.length || sel.value !== state.displayUnit) {
    const prev = state.displayUnit;
    sel.innerHTML = UNITS.map((u) => `<option value="${u}">${u === "px" ? "标定单位" : u}</option>`).join("");
    sel.value = UNITS.includes(prev) ? prev : state.unit;
    state.displayUnit = sel.value;
  }
}

function updateZoomLabel() {
  els.zoomLevel.textContent = state.image ? Math.round(state.view.scale * 100) + "%" : "—";
}

/* ====================================================== calibration logic */
let calibSeq = 0;
async function recomputeCalibration({ push = true } = {}) {
  // Local live computation (instant while dragging).
  const dimsValid = isFinite(state.rectW) && isFinite(state.rectH) &&
    state.rectW > 0 && state.rectH > 0;
  if (state.corners.length === 4 && dimsValid) {
    const r = solveHomography(state.corners, state.rectW, state.rectH);
    state.calibValid = r.ok;
    state.H = r.ok ? r.H : null;
    state.calibErrors = r.errors || [];
  } else {
    state.calibValid = false;
    state.H = null;
    state.calibErrors = [];
  }
  recomputePolygonMetrics();

  const seq = ++calibSeq;
  if (push && state.corners.length === 4 && dimsValid) {
    try {
      const data = await api(`/api/sessions/${state.sessionId}/calibration`, {
        method: "PUT",
        body: JSON.stringify({
          corners: state.corners,
          rect_width: state.rectW,
          rect_height: state.rectH,
          unit: state.unit,
        }),
      });
      if (seq !== calibSeq) return; // a newer edit is already in flight
      // Adopt authoritative server answer.
      state.calibValid = data.calibration.valid;
      state.H = data.calibration.H;
      state.calibErrors = data.calibration.errors || [];
      mergeServerPolygons(data.polygons);
    } catch (err) {
      if (seq === calibSeq) toast("标定同步失败（本地结果仍显示）：" + err.message, true);
    }
  }
  els.modeDraw.disabled = !state.calibValid;
  els.modeEdit.disabled = state.polygons.length === 0;
  els.modeHole.disabled = !state.calibValid || state.polygons.length === 0;
  els.modeDrawAnother.disabled = !state.calibValid;
  els.clearPolys.disabled = state.polygons.length === 0;
  refreshUI();
}

function ringSignature(points) {
  return points.map((p) => `${Math.round(p.x * 1000)},${Math.round(p.y * 1000)}`).join("|");
}

function polyGeometrySignature(poly) {
  return JSON.stringify({
    o: poly.points.map((p) => [Math.round(p.x * 1000), Math.round(p.y * 1000)]),
    h: (poly.holes || []).map((h) => h.points.map((p) => [Math.round(p.x * 1000), Math.round(p.y * 1000)])),
  });
}

/* ------------------------------------------------------- sensitivity --- */
/** Fingerprint of everything a sensitivity report depends on: calibration
 *  corners + rect dims + unit, the error radius, and every region's geometry. */
function sensFingerprint() {
  return JSON.stringify({
    c: state.corners.map((p) => [Math.round(p.x * 1000), Math.round(p.y * 1000)]),
    w: state.rectW, h: state.rectH, u: state.unit, r: state.sensRadius,
    p: state.polygons.map((pl) => [pl.id, polyGeometrySignature(pl)]),
  });
}

/** 角点、误差半径或区域变化后自动作废旧评估结果。 */
function dropStaleSensitivity() {
  if (state.sensitivity && state.sensitivity.fp !== sensFingerprint())
    state.sensitivity = null;
}

function refreshSensButton() {
  els.sensRun.disabled = state.sensBusy || !state.calibValid || state.polygons.length === 0;
}

async function runSensitivity() {
  const r = state.sensRadius;
  if (!state.calibValid) { showNotice("请先完成有效的四点透视标定", true); return; }
  if (!state.polygons.length) { showNotice("请先绘制至少一个测量区域", true); return; }
  if (!(isFinite(r) && r > 0 && r <= 1000)) {
    showNotice("误差半径需为 0–1000 px 之间的正数", true);
    return;
  }
  state.sensBusy = true;
  refreshSensButton();
  els.sensRun.textContent = "评估中…";
  try {
    const data = await api(`/api/sessions/${state.sessionId}/sensitivity`, {
      method: "POST",
      body: JSON.stringify({ radius_px: r }),
    });
    // Fingerprint AFTER the await: if the geometry changed in flight the
    // result is immediately stale and gets dropped on the next render.
    state.sensitivity = { fp: sensFingerprint(), radius: r, data };
    renderPolyList();
    if (!data.reliable)
      showNotice("标定敏感性：" + (data.message || "有效样本比例过低，请重新标定"), true);
    else
      toast(`敏感性评估完成：${data.sample_count} 组扰动，覆盖率 ${(data.valid_ratio * 100).toFixed(1)}%`);
  } catch (err) {
    toast("敏感性评估失败：" + err.message, true);
  } finally {
    state.sensBusy = false;
    els.sensRun.textContent = "评估敏感性";
    refreshSensButton();
  }
}

/** Per-card sensitivity block: P5–P95 intervals + valid-sample coverage. */
function sensBlockHtml(poly, areaUnit) {
  const pack = state.sensitivity;
  if (!pack) return "";
  const entry = pack.data.polygons.find((sp) => sp.id === poly.id);
  if (!entry) return "";
  const pct = (entry.valid_ratio * 100).toFixed(1);
  const head = `标定敏感性 ±${fmtNum(pack.radius)}px · 覆盖率 ${pct}%（${entry.valid_samples}/${entry.sample_count}）`;
  if (!entry.reliable) {
    return `<div class="sens-info bad">
      <div class="sens-head">${head} · 低于 80%</div>
      <div class="sens-row"><span>退化 / 越界 / 几何非法样本过多，区间不予显示 —— 请重新标定（让参照矩形占更大画面比例、减小拍摄倾角，或调小误差半径）。</span></div>
    </div>`;
  }
  const a5 = convertValue(entry.area.p5, state.unit, state.displayUnit, true);
  const a50 = convertValue(entry.area.p50, state.unit, state.displayUnit, true);
  const a95 = convertValue(entry.area.p95, state.unit, state.displayUnit, true);
  const p5 = convertValue(entry.perimeter.p5, state.unit, state.displayUnit, false);
  const p50 = convertValue(entry.perimeter.p50, state.unit, state.displayUnit, false);
  const p95 = convertValue(entry.perimeter.p95, state.unit, state.displayUnit, false);
  return `<div class="sens-info">
    <div class="sens-head">${head}</div>
    <div class="sens-row"><span>面积 P5–P95</span><b>${fmtNum(a5)} – ${fmtNum(a95)} ${areaUnit}</b></div>
    <div class="sens-row"><span>周长 P5–P95</span><b>${fmtNum(p5)} – ${fmtNum(p95)} ${state.displayUnit}</b></div>
    <div class="sens-row sens-med"><span>P50 中位</span><span>${fmtNum(a50)} ${areaUnit} · ${fmtNum(p50)} ${state.displayUnit}</span></div>
  </div>`;
}

function makeLastValid(poly) {
  return {
    points: poly.points.map((p) => ({ ...p })),
    holes: (poly.holes || []).map((h) => ({ points: h.points.map((p) => ({ ...p })) })),
    area: poly.area,
    grossArea: poly.grossArea,
    perimeter: poly.perimeter,
    valid: poly.valid,
  };
}

function adoptServerPolygon(poly, sp) {
  const sameGeometry =
    ringSignature(poly.points) === ringSignature(sp.points) &&
    (poly.holes || []).length === (sp.holes || []).length &&
    (poly.holes || []).every((h, k) =>
      ringSignature(h.points) === ringSignature((sp.holes[k] || {}).points || []));
  poly.area = sp.area;
  poly.perimeter = sp.perimeter;
  poly.grossArea = sp.gross_area ?? null;
  poly.valid = sp.valid;
  poly.errors = sp.errors || [];
  poly.worldPoints = sp.world_points;
  poly.holeAreas = (sp.holes || []).map((h) => h.area);
  poly.holePerimeters = (sp.holes || []).map((h) => h.perimeter);
  poly.holeWorldPoints = (sp.holes || []).map((h) => h.world_points);
  if (sp.valid && sameGeometry) poly.lastValid = makeLastValid(poly);
  return sameGeometry;
}

function mergeServerPolygons(serverPolys) {
  for (const sp of serverPolys) {
    const local = state.polygons.find((p) => p.id === sp.id);
    if (!local) continue;
    adoptServerPolygon(local, sp);
  }
}

function applyLiveMetrics(poly, m) {
  poly.valid = m.valid;
  poly.area = m.valid ? m.area : null;
  poly.grossArea = m.valid ? m.grossArea ?? null : null;
  poly.perimeter = m.valid ? m.perimeter : null;
  poly.errors = m.errors || [];
  poly.worldPoints = m.valid ? m.worldPoints : null;
  poly.holeAreas = m.valid ? m.holeAreas || [] : [];
  poly.holePerimeters = m.valid ? m.holePerimeters || [] : [];
  poly.holeWorldPoints = m.valid ? m.holeWorldPoints || [] : [];
}

function recomputePolygonMetrics() {
  for (const poly of state.polygons) {
    if (state.calibValid && state.H) {
      const m = measureRegion(state.H, poly.points, (poly.holes || []).map((h) => h.points));
      applyLiveMetrics(poly, m);
    } else {
      poly.valid = false;
      poly.area = poly.grossArea = poly.perimeter = null;
      poly.errors = ["尚未完成透视校准"];
      poly.worldPoints = null;
      poly.holeAreas = []; poly.holePerimeters = []; poly.holeWorldPoints = [];
    }
  }
}

/* ============================================================ polygons === */
function hydratePolygon(sp) {
  return {
    id: sp.id,
    name: sp.name,
    points: sp.points.map((p) => ({ x: p.x, y: p.y })),
    holes: (sp.holes || []).map((h) => ({ points: h.points.map((q) => ({ x: q.x, y: q.y })) })),
    area: sp.area,
    grossArea: sp.gross_area ?? null,
    perimeter: sp.perimeter,
    valid: sp.valid,
    errors: sp.errors,
    worldPoints: sp.world_points,
    holeAreas: (sp.holes || []).map((h) => h.area),
    holePerimeters: (sp.holes || []).map((h) => h.perimeter),
    holeWorldPoints: (sp.holes || []).map((h) => h.world_points),
    lastValid: null,
  };
}

async function addPolygon(points, holes = [], name) {
  const payload = { points, holes: holes.map((h) => ({ points: h })), name };
  let created;
  try {
    const r = await api(`/api/sessions/${state.sessionId}/polygons`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    created = r.polygon;
  } catch (err) {
    toast("区域保存失败：" + err.message, true);
    return null;
  }
  const poly = hydratePolygon(created);
  state.polygons.push(poly);
  state.selected = created.id;
  recomputePolygonMetrics();
  if (created.valid) poly.lastValid = makeLastValid(poly);
  els.modeEdit.disabled = false;
  els.modeHole.disabled = false;
  els.modeDrawAnother.disabled = false;
  els.clearPolys.disabled = false;
  setMode("edit");
  refreshUI();
  return poly;
}

const syncTimers = new Map();
function syncPolygon(poly) {
  clearTimeout(syncTimers.get(poly.id));
  const timer = setTimeout(async () => {
    syncTimers.delete(poly.id);
    // Name-only edits while geometry is locally invalid are deferred until the
    // geometry becomes valid again (the PUT must carry the last valid rings).
    if (!poly.valid) return;
    try {
      const r = await api(`/api/sessions/${state.sessionId}/polygons/${poly.id}`, {
        method: "PUT",
        body: JSON.stringify({
          points: poly.points,
          holes: (poly.holes || []).map((h) => ({ points: h.points })),
          name: poly.name,
        }),
      });
      adoptServerPolygon(poly, r.polygon);
      refreshUI();
    } catch (err) {
      if (err.status === 422 && err.data?.polygon) {
        // Server rejected the edit and kept its last valid geometry. Adopt it
        // locally so the two sides never disagree about the "good" result.
        const sp = err.data.polygon;
        poly.points = sp.points.map((p) => ({ x: p.x, y: p.y }));
        poly.holes = (sp.holes || []).map((h) => ({ points: h.points.map((q) => ({ x: q.x, y: q.y })) }));
        adoptServerPolygon(poly, sp);
        poly.lastValid = makeLastValid(poly);
        showNotice("服务端拒绝了该非法几何，已恢复为最后一次有效结果：" + (sp.errors || []).join("；"), true);
        refreshUI();
      } else {
        toast("区域更新失败：" + err.message + "（本地结果仍显示，未覆盖服务端最后有效结果）", true);
      }
    }
  }, 250);
}

/** Revert the local (possibly invalid) geometry to the last accepted one. */
function revertPolygon(id) {
  const poly = state.polygons.find((p) => p.id === id);
  if (!poly || !poly.lastValid) return;
  poly.points = poly.lastValid.points.map((p) => ({ ...p }));
  poly.holes = poly.lastValid.holes.map((h) => ({ points: h.points.map((p) => ({ ...p })) }));
  recomputePolygonMetrics();
  poly.lastValid = poly.valid ? makeLastValid(poly) : poly.lastValid;
  if (poly.valid) syncPolygon(poly);
  showNotice(null);
  refreshUI();
}

async function deleteHole(polyId, holeIndex) {
  const poly = state.polygons.find((p) => p.id === polyId);
  if (!poly || !poly.holes[holeIndex]) return;
  poly.holes.splice(holeIndex, 1);
  recomputePolygonMetrics();
  if (poly.valid) syncPolygon(poly);
  refreshUI();
}

async function deletePolygon(id) {
  try {
    await api(`/api/sessions/${state.sessionId}/polygons/${id}`, { method: "DELETE" });
  } catch (err) {
    toast("删除失败：" + err.message, true);
    return;
  }
  state.polygons = state.polygons.filter((p) => p.id !== id);
  if (state.selected === id || state.holeTarget === id) {
    state.selected = state.polygons[0]?.id || null;
    state.holeTarget = state.selected;
    if (!state.selected && (state.mode === "edit" || state.mode === "hole"))
      setMode(state.calibValid ? "draw" : "calib");
  }
  els.modeEdit.disabled = state.polygons.length === 0;
  els.modeHole.disabled = state.polygons.length === 0;
  els.modeDrawAnother.disabled = !state.calibValid;
  els.clearPolys.disabled = state.polygons.length === 0;
  refreshUI();
}

/* ====================================================== hit testing ====== */
function hitCorner(imgPt) {
  const TOL = 9 / state.view.scale;
  let best = -1, bestD = TOL;
  state.corners.forEach((p, i) => {
    const d = dist(p, imgPt);
    if (d < bestD) { bestD = d; best = i; }
  });
  return best;
}

function hitPolygonVertex(imgPt) {
  const TOL = 9 / state.view.scale;
  for (let i = state.polygons.length - 1; i >= 0; i--) {
    const poly = state.polygons[i];
    for (let k = 0; k < poly.points.length; k++)
      if (dist(poly.points[k], imgPt) < TOL) return { polyIndex: i, polyId: poly.id, ring: 0, vertex: k };
    for (let hk = 0; hk < (poly.holes || []).length; hk++)
      for (let k = 0; k < poly.holes[hk].points.length; k++)
        if (dist(poly.holes[hk].points[k], imgPt) < TOL)
          return { polyIndex: i, polyId: poly.id, ring: hk + 1, vertex: k };
  }
  return null;
}

function pointInPolygon(pt, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const a = pts[i], b = pts[j];
    if ((a.y > pt.y) !== (b.y > pt.y) &&
        pt.x < ((b.x - a.x) * (pt.y - a.y)) / (b.y - a.y) + a.x)
      inside = !inside;
  }
  return inside;
}

/** Inside the solid material of a region (outer ring minus holes). */
function pointInRegion(imgPt, poly) {
  if (!pointInPolygon(imgPt, poly.points)) return false;
  for (const hole of poly.holes || []) {
    if (pointInPolygon(imgPt, hole.points)) return false;
  }
  return true;
}

function hitPolygonBody(imgPt) {
  for (let i = state.polygons.length - 1; i >= 0; i--) {
    const poly = state.polygons[i];
    if (pointInRegion(imgPt, poly)) return i;
    // near an outer edge or hole edge
    const TOL = 6 / state.view.scale;
    for (let k = 0; k < poly.points.length; k++) {
      if (distToSegment(imgPt, poly.points[k], poly.points[(k + 1) % poly.points.length]) < TOL) return i;
    }
    for (const hole of poly.holes || []) {
      for (let k = 0; k < hole.points.length; k++) {
        if (distToSegment(imgPt, hole.points[k], hole.points[(k + 1) % hole.points.length]) < TOL) return i;
      }
    }
  }
  return -1;
}

function distToSegment(p, a, b) {
  const abx = b.x - a.x, aby = b.y - a.y;
  const t = Math.min(1, Math.max(0, ((p.x - a.x) * abx + (p.y - a.y) * aby) / (abx * abx + aby * aby || 1)));
  return dist(p, { x: a.x + abx * t, y: a.y + aby * t });
}

/* ============================================================ pointer === */
function eventPos(e) {
  const r = $canvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

$canvas.addEventListener("pointerdown", (e) => {
  if (!state.image) return;
  $canvas.setPointerCapture(e.pointerId);
  const sp = eventPos(e);
  const ip = screenToImg(sp);
  const panMode = state.spaceDown || state.mode === "pan" || e.button === 2;

  if (panMode) {
    state.dragging = { type: "pan", start: sp, view: { ...state.view } };
    renderModeButtons();
    return;
  }

  if (e.button !== 0) return;

  if (state.mode === "calib") {
    const ci = hitCorner(ip);
    if (ci >= 0) {
      state.dragging = { type: "corner", index: ci };
    } else if (state.corners.length < 4) {
      state.corners.push(clampImagePoint(ip));
      void recomputeCalibration({ push: false });
      if (state.corners.length === 4) void recomputeCalibration();
    }
    return;
  }

  if (state.mode === "draw") {
    if (!state.calibValid) return;
    // A double-click closes the polygon; its second pointerdown must not add
    // a duplicate vertex at the same spot.
    if (state.draft.length && dist(ip, state.draft[state.draft.length - 1]) < 8 / state.view.scale)
      return;
    // clicking near first vertex closes
    if (state.draft.length >= 3 && dist(ip, state.draft[0]) < 12 / state.view.scale) {
      finishDraft();
      return;
    }
    state.draft.push(clampImagePoint(ip));
    render();
    return;
  }

  if (state.mode === "hole") {
    if (!state.calibValid) return;
    let target = state.polygons.find((p) => p.id === state.holeTarget);
    // First click of a new hole also picks the region: it must land inside
    // some region's outer ring (nesting in an existing hole is rejected at
    // closing time with a precise conflict message).
    if (state.holeDraft.length === 0) {
      let bi = -1;
      for (let k = state.polygons.length - 1; k >= 0; k--) {
        if (pointInPolygon(ip, state.polygons[k].points)) { bi = k; break; }
      }
      if (bi < 0) { showNotice("请在要镂空的区域外环内部开始绘制孔洞", true); return; }
      target = state.polygons[bi];
      state.holeTarget = target.id;
      state.selected = target.id;
      renderModeButtons();
    }
    if (state.holeDraft.length && dist(ip, state.holeDraft[state.holeDraft.length - 1]) < 8 / state.view.scale)
      return;
    if (state.holeDraft.length >= 3 && dist(ip, state.holeDraft[0]) < 12 / state.view.scale) {
      finishHoleDraft();
      return;
    }
    state.holeDraft.push(clampImagePoint(ip));
    render();
    return;
  }

  if (state.mode === "edit") {
    const hv = hitPolygonVertex(ip);
    if (hv) {
      state.selected = hv.polyId;
      state.dragging = { type: "vertex", ...hv };
    } else {
      const bi = hitPolygonBody(ip);
      if (bi >= 0) {
        state.selected = state.polygons[bi].id;
        state.dragging = {
          type: "body",
          polyId: state.selected,
          startImg: ip,
          origOuter: state.polygons[bi].points.map((p) => ({ ...p })),
          origHoles: (state.polygons[bi].holes || []).map((h) => h.points.map((p) => ({ ...p }))),
        };
      } else {
        state.selected = null;
      }
    }
    refreshUI();
  }
});

$canvas.addEventListener("pointermove", (e) => {
  const sp = eventPos(e);
  const ip = screenToImg(sp);
  state.hoverImg = ip;
  els.cursorLabel.textContent = state.image
    ? `图像坐标 ${isFinite(ip.x) ? Math.round(ip.x) : "—"}, ${isFinite(ip.y) ? Math.round(ip.y) : "—"} px`
    : "图像坐标 —";

  const d = state.dragging;
  if (!d) { render(); return; }

  if (d.type === "pan") {
    state.view.x = d.view.x + (sp.x - d.start.x);
    state.view.y = d.view.y + (sp.y - d.start.y);
    render();
    return;
  }
  if (d.type === "corner") {
    state.corners[d.index] = clampImagePoint(ip);
    void recomputeCalibration({ push: false });
    return;
  }
  const poly = state.polygons.find((p) => p.id === d.polyId);
  if (!poly) return;
  if (d.type === "vertex") {
    if (d.ring === 0) poly.points[d.vertex] = clampImagePoint(ip);
    else poly.holes[d.ring - 1].points[d.vertex] = clampImagePoint(ip);
  } else if (d.type === "body") {
    const dx = ip.x - d.startImg.x, dy = ip.y - d.startImg.y;
    poly.points = d.origOuter.map((p) => clampImagePoint({ x: p.x + dx, y: p.y + dy }));
    poly.holes = d.origHoles.map((hpts) => hpts.map((p) => clampImagePoint({ x: p.x + dx, y: p.y + dy })));
  }
  recomputePolygonMetrics();
  render();
  updatePolyCardMetrics();
});

$canvas.addEventListener("pointerup", (e) => {
  const d = state.dragging;
  state.dragging = null;
  $canvas.releasePointerCapture?.(e.pointerId);
  if (!d) { render(); return; }
  if (d.type === "corner") {
    void recomputeCalibration();
  } else if (d.type === "vertex" || d.type === "body") {
    const poly = state.polygons.find((p) => p.id === d.polyId);
    if (poly) syncPolygon(poly);
  }
  renderModeButtons();
  render();
});

$canvas.addEventListener("pointerleave", () => {
  state.hoverImg = null;
  render();
});

$canvas.addEventListener("dblclick", () => {
  if (state.mode === "draw" && state.draft.length >= 3) finishDraft();
  if (state.mode === "hole" && state.holeDraft.length >= 3) finishHoleDraft();
});

$canvas.addEventListener("contextmenu", (e) => {
  e.preventDefault();
  if (!state.image) return;
  const ip = screenToImg(eventPos(e));
  if (state.mode === "edit") {
    const hv = hitPolygonVertex(ip);
    if (hv) {
      const poly = state.polygons[hv.polyIndex];
      const ringPts = hv.ring === 0 ? poly.points : poly.holes[hv.ring - 1].points;
      if (ringPts.length <= 3) {
        if (hv.ring === 0) {
          toast("闭合多边形至少保留 3 个顶点，已改为删除整个区域");
          void deletePolygon(poly.id);
        } else {
          toast("孔洞至少保留 3 个顶点，已改为删除该孔洞");
          void deleteHole(poly.id, hv.ring - 1);
        }
      } else {
        ringPts.splice(hv.vertex, 1);
        recomputePolygonMetrics();
        if (poly.valid) syncPolygon(poly);
        refreshUI();
      }
    }
  }
});

$canvas.addEventListener("wheel", (e) => {
  if (!state.image) return;
  e.preventDefault();
  const factor = Math.exp(-e.deltaY * 0.0012);
  zoomAt(eventPos(e), factor);
}, { passive: false });

function finishDraft() {
  const pts = state.draft.map((p) => ({ ...p }));
  state.draft = [];
  if (pts.length < 3) { toast("至少需要 3 个顶点"); return; }
  // Instant local validity feedback, server re-checks on persist.
  const m = state.calibValid ? measureRegion(state.H, pts, []) : { valid: false, errors: ["尚未标定"] };
  if (!m.valid) {
    showNotice(m.errors.join("；"), true);
    state.draft = pts; // keep editing
    return;
  }
  void addPolygon(pts);
}

function finishHoleDraft() {
  const pts = state.holeDraft.map((p) => ({ ...p }));
  const target = state.polygons.find((p) => p.id === state.holeTarget);
  state.holeDraft = [];
  if (!target) return;
  if (pts.length < 3) { toast("孔洞至少需要 3 个顶点"); return; }
  const holesImg = [...(target.holes || []).map((h) => h.points), pts];
  const m = measureRegion(state.H, target.points, holesImg);
  if (!m.valid) {
    showNotice(m.errors.join("；"), true);
    state.holeDraft = pts; // keep editing the draft
    return;
  }
  // Valid: commit locally and persist the whole region.
  target.holes.push({ points: pts });
  applyLiveMetrics(target, m);
  target.lastValid = makeLastValid(target);
  syncPolygon(target);
  state.selected = target.id;
  showNotice(`已为「${target.name}」添加孔洞 ${target.holes.length}：面积与周长已实时重算`);
  setMode("edit");
  refreshUI();
}

/* ============================================================ keyboard == */
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.code === "Space") { state.spaceDown = true; renderModeButtons(); e.preventDefault(); }
  if (e.key === "Escape") {
    state.draft = [];
    state.holeDraft = [];
    state.dragging = null;
    showNotice(null);
    render();
  }
  if (e.key === "Backspace") {
    if (state.mode === "draw" && state.draft.length) { state.draft.pop(); render(); e.preventDefault(); }
    if (state.mode === "hole" && state.holeDraft.length) { state.holeDraft.pop(); render(); e.preventDefault(); }
  }
  if (e.key === "Enter" && state.mode === "draw" && state.draft.length >= 3) finishDraft();
  if (e.key === "Enter" && state.mode === "hole" && state.holeDraft.length >= 3) finishHoleDraft();
  if (e.key === "Delete" && state.mode === "edit" && state.selected) deletePolygon(state.selected);
  if (e.key === "1") setMode("calib");
  if (e.key === "2" && !els.modeDraw.disabled) setMode("draw");
  if (e.key === "3" && !els.modeEdit.disabled) setMode("edit");
  if (e.key === "4") setMode("pan");
  if (e.key === "5" && !els.modeHole.disabled) startHoleMode();
  if (e.key === "f" || e.key === "F") fitView();
  if (e.key === "+" || e.key === "=") {
    const { w, h } = screenSize();
    zoomAt({ x: w / 2, y: h / 2 }, 1.2);
  }
  if (e.key === "-") {
    const { w, h } = screenSize();
    zoomAt({ x: w / 2, y: h / 2 }, 1 / 1.2);
  }
});
window.addEventListener("keyup", (e) => {
  if (e.code === "Space") { state.spaceDown = false; renderModeButtons(); }
});

/* ============================================================ modes ====== */
function setMode(m) {
  if (m === "draw" && !state.calibValid) {
    showNotice("请先完成有效的四点透视标定", true);
    return;
  }
  if (m === "edit" && state.polygons.length === 0) return;
  if (m === "hole" && state.polygons.length === 0) return;
  state.mode = m;
  if (m !== "draw") state.draft = [];
  if (m !== "hole") state.holeDraft = [];
  if (m === "hole" && !state.holeTarget) state.holeTarget = state.selected;
  showNotice(null);
  renderModeButtons();
  render();
}

function startHoleMode() {
  if (!state.calibValid) { showNotice("请先完成有效的四点透视标定", true); return; }
  if (state.polygons.length === 0) { showNotice("请先绘制一个闭合外环区域，再添加孔洞", true); return; }
  if (!state.holeTarget) state.holeTarget = state.selected || state.polygons[0].id;
  state.selected = state.holeTarget;
  state.holeDraft = [];
  const target = state.polygons.find((p) => p.id === state.holeTarget);
  if (target && !target.valid) {
    showNotice("目标区域当前几何非法，请先恢复为有效状态再添加孔洞", true);
    setMode("edit");
    return;
  }
  setMode("hole");
}

els.modeCalib.addEventListener("click", () => setMode("calib"));
els.modeDraw.addEventListener("click", () => setMode("draw"));
els.modeHole.addEventListener("click", startHoleMode);
els.modeEdit.addEventListener("click", () => setMode("edit"));
els.modePan.addEventListener("click", () => setMode("pan"));
els.modeDrawAnother.addEventListener("click", () => setMode("draw"));

els.clearCorners.addEventListener("click", () => {
  state.corners = [];
  state.H = null;
  state.calibValid = false;
  state.calibErrors = [];
  recomputePolygonMetrics();
  els.modeDraw.disabled = true;
  els.modeHole.disabled = state.polygons.length === 0;
  setMode("calib");
  refreshUI();
  toast("已清除角点，可重新标记");
});

els.clearPolys.addEventListener("click", async () => {
  if (!state.polygons.length) return;
  if (!confirm(`确定删除全部 ${state.polygons.length} 个测量区域（含孔洞）？`)) return;
  try {
    await api(`/api/sessions/${state.sessionId}/polygons`, { method: "DELETE" });
  } catch (err) { toast("清空失败：" + err.message, true); return; }
  state.polygons = [];
  state.selected = null;
  state.holeTarget = null;
  state.holeDraft = [];
  els.modeEdit.disabled = true;
  els.modeHole.disabled = true;
  els.modeDrawAnother.disabled = !state.calibValid;
  els.clearPolys.disabled = true;
  setMode(state.calibValid ? "draw" : "calib");
  refreshUI();
});

/* ----------------------------------------------------------- zoom UI --- */
els.zoomIn.addEventListener("click", () => zoomCenter(1.25));
els.zoomOut.addEventListener("click", () => zoomCenter(1 / 1.25));
els.zoomFit.addEventListener("click", fitView);
els.zoomOne.addEventListener("click", () => zoomCenterTo(1));
function zoomCenter(f) { const { w, h } = screenSize(); zoomAt({ x: w / 2, y: h / 2 }, f); }
function zoomCenterTo(s) { const { w, h } = screenSize(); zoomAt({ x: w / 2, y: h / 2 }, s / state.view.scale); }

/* ------------------------------------------------------------- notice --- */
let noticeTimer = null;
function showNotice(msg, warn = false) {
  const n = els.notice;
  if (!msg) { n.classList.add("hidden"); return; }
  els.noticeText.textContent = msg;
  n.classList.toggle("warn", warn);
  n.classList.remove("hidden");
  clearTimeout(noticeTimer);
}
els.noticeClose.addEventListener("click", () => showNotice(null));

/* --------------------------------------------------------- dimension UI - */
els.rectW.addEventListener("input", () => {
  state.rectW = parseFloat(els.rectW.value);
  void recomputeCalibration();
});
els.rectH.addEventListener("input", () => {
  state.rectH = parseFloat(els.rectH.value);
  void recomputeCalibration();
});
els.rectUnit.addEventListener("change", () => {
  state.unit = els.rectUnit.value;
  if (!UNITS.includes(state.displayUnit) || state.displayUnit === "px")
    state.displayUnit = state.unit;
  void recomputeCalibration();
});
els.displayUnit.addEventListener("change", () => {
  state.displayUnit = els.displayUnit.value;
  renderPolyList();
});

/* --------------------------------------------------------- sensitivity - */
els.sensRadius.addEventListener("input", () => {
  state.sensRadius = parseFloat(els.sensRadius.value);
  dropStaleSensitivity(); // 误差半径变化 -> 旧评估作废
  renderPolyList();
});
els.sensRun.addEventListener("click", () => { void runSensitivity(); });

/* ============================================================ upload ===== */
async function ensureSession() {
  let sid = null;
  try { sid = localStorage.getItem("pr_sid"); } catch { /* ignore */ }
  if (sid) {
    try {
      const s = await api(`/api/sessions/${sid}`);
      hydrateSession(s);
      return;
    } catch { /* expired session -> create new */ }
  }
  const r = await api("/api/sessions", { method: "POST" });
  state.sessionId = r.session_id;
  try { localStorage.setItem("pr_sid", r.session_id); } catch { /* ignore */ }
  setMode("calib");
  refreshUI();
}

function hydrateSession(s) {
  state.sessionId = s.session_id;
  state.sensitivity = null;
  try { localStorage.setItem("pr_sid", s.session_id); } catch { /* ignore */ }
  if (s.image) {
    state.image = s.image;
    if (s.calibration) {
      state.corners = s.calibration.corners.map((p) => ({ x: p.x, y: p.y }));
      state.rectW = s.calibration.rect_width;
      state.rectH = s.calibration.rect_height;
      state.unit = s.calibration.unit;
      state.H = s.calibration.H;
      state.calibValid = s.calibration.valid;
      state.calibErrors = s.calibration.errors || [];
      els.rectW.value = state.rectW;
      els.rectH.value = state.rectH;
      els.rectUnit.value = state.unit;
    }
    state.polygons = s.polygons.map((p) => {
      const poly = hydratePolygon(p);
      if (p.valid) poly.lastValid = makeLastValid(poly);
      return poly;
    });
    state.selected = state.polygons[0]?.id || null;
    state.holeTarget = state.selected;
    loadImage(s.image.filename, false);
  } else {
    resetUI();
  }
  els.modeDraw.disabled = !state.calibValid;
  els.modeEdit.disabled = state.polygons.length === 0;
  els.modeHole.disabled = !state.calibValid || state.polygons.length === 0;
  els.modeDrawAnother.disabled = !state.calibValid;
  els.clearPolys.disabled = state.polygons.length === 0;
  els.exportBtn.disabled = !s.image;
  state.displayUnit = state.unit;
  setMode(state.calibValid ? (state.polygons.length ? "edit" : "draw") : "calib");
  refreshUI();
}

function resetUI() {
  state.image = null;
  state.imgEl = null;
  state.corners = [];
  state.H = null;
  state.calibValid = false;
  state.calibErrors = [];
  state.polygons = [];
  state.draft = [];
  state.holeDraft = [];
  state.holeTarget = null;
  state.selected = null;
  state.sensitivity = null;
  els.exportBtn.disabled = true;
  els.emptyOverlay.classList.remove("hidden");
}

function loadImage(filename, keepView = false) {
  const img = new Image();
  img.onload = () => {
    state.imgEl = img;
    els.emptyOverlay.classList.add("hidden");
    els.exportBtn.disabled = false;
    if (!keepView) fitView();
    render();
  };
  img.src = `/api/images/${filename}`;
}

async function uploadFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showNotice("请选择图像文件（JPG / PNG / WebP）", true);
    return;
  }
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`/api/sessions/${state.sessionId}/image`, { method: "POST", body: fd });
  if (!res.ok) {
    let detail = "上传失败";
    try { detail = (await res.json()).detail || detail; } catch { /* ignore */ }
    showNotice(detail, true);
    return;
  }
  const data = await res.json();
  state.image = data.image;
  state.corners = [];
  state.H = null;
  state.calibValid = false;
  state.polygons = [];
  state.draft = [];
  state.holeDraft = [];
  state.holeTarget = null;
  state.selected = null;
  state.sensitivity = null;
  els.rectW.value = ""; els.rectH.value = "";
  state.rectW = null; state.rectH = null;
  els.modeDraw.disabled = true;
  els.modeEdit.disabled = true;
  els.modeHole.disabled = true;
  els.modeDrawAnother.disabled = true;
  els.clearPolys.disabled = true;
  loadImage(data.image.filename);
  setMode("calib");
  refreshUI();
  toast("照片已上传，请标记已知矩形的四个角点");
}

els.fileInput.addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if (f) void uploadFile(f);
  e.target.value = "";
});

["dragenter", "dragover"].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    els.dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    els.dropzone.classList.remove("drag");
  })
);
els.dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files?.[0];
  if (f) void uploadFile(f);
});
// Also allow dropping directly on the canvas to replace the photo.
$canvas.addEventListener("dragover", (e) => e.preventDefault());
$canvas.addEventListener("drop", (e) => {
  e.preventDefault();
  const f = e.dataTransfer.files?.[0];
  if (f) void uploadFile(f);
});

/* ============================================================ export ===== */
els.exportBtn.addEventListener("click", () => {
  if (!state.sessionId) return;
  const url = `/api/sessions/${state.sessionId}/export`;
  const a = document.createElement("a");
  a.href = url;
  a.download = `measurements-${state.sessionId.slice(0, 8)}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
});

/* ============================================================ boot ======== */
window.addEventListener("resize", resizeCanvas);

(async function boot() {
  resizeCanvas();
  renderModeButtons();
  try {
    await ensureSession();
  } catch (err) {
    showNotice("无法连接后端服务：" + err.message, true);
  }
})();
