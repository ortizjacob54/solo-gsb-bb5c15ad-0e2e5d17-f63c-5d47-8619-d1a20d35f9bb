/* Parity test: frontend/app.js geometry vs the same synthetic camera used in
 * tests/test_geometry.py. Run: node tests/parity_geometry.mjs
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
let src = readFileSync(join(here, "..", "frontend", "app.js"), "utf8");

// Strip browser-only sections; keep pure geometry functions.
const start = src.indexOf("/* ============================================================ geometry");
const end = src.indexOf("/* ============================================================ state");
src = src.slice(start, end)
  .replace("\"use strict\";", "");

const fns = new Function(`${src}; return {
  solveHomography, applyH, measurePolygon, measureRegion, validateRingTopology,
  polygonSelfIntersects, convexOrderedQuad, dist
};`);
const G = fns();

// --- same synthetic camera as the Python test ---
const K = [[800, 0, 512], [0, 800, 384], [0, 0, 1]];
const r1 = [0.999, 0, -0.0499];
const r2 = [0.02, 1, 0.03];
const t = [0, 0, 12];
const P = [
  [K[0][0] * r1[0] + K[0][2] * r1[2], K[0][0] * r2[0] + K[0][2] * r2[2], K[0][2] * t[2]],
  [K[1][1] * r1[1] + K[1][2] * r1[2], K[1][1] * r2[1] + K[1][2] * r2[2], K[1][2] * t[2]],
  [r1[2], r2[2], t[2]],
];
const project = (x, y) => {
  const w = P[2][0] * x + P[2][1] * y + P[2][2];
  if (w <= 1) throw new Error("behind camera");
  return {
    x: (P[0][0] * x + P[0][1] * y + P[0][2]) / w,
    y: (P[1][0] * x + P[1][1] * y + P[1][2]) / w,
  };
};

const W = 100, H = 60;
const corners = [[0, 0], [W, 0], [W, H], [0, H]].map(([x, y]) => project(x, y));
let fails = 0;
const check = (name, cond, detail = "") => {
  console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}  ${detail}`);
  if (!cond) fails++;
};

console.log("== JS homography ==");
const r = G.solveHomography(corners, W, H);
check("solve ok", r.ok, JSON.stringify(r.errors));
if (r.ok) {
  let err = 0;
  [[0, 0], [W, 0], [W, H], [0, H]].forEach(([x, y], i) => {
    const q = G.applyH(r.H, corners[i]);
    err = Math.max(err, Math.hypot(q.x - x, q.y - y));
  });
  check("residual ~0", err < 1e-4, `err=${err.toExponential(2)}`);

  const poly = [project(20, 10), project(70, 10), project(70, 50), project(20, 50)];
  const m = G.measurePolygon(r.H, poly);
  check("area == 2000", m.valid && Math.abs(m.area - 2000) < 1e-4, `area=${m.area}`);
  check("perimeter == 180", m.valid && Math.abs(m.perimeter - 180) < 1e-4, `p=${m.perimeter}`);
}

console.log("== JS invalid inputs ==");
check("duplicate", !G.solveHomography([{x:0,y:0},{x:100,y:0},{x:100,y:0},{x:0,y:60}], W, H).ok);
check("collinear", !G.solveHomography([{x:0,y:0},{x:50,y:0},{x:100,y:0},{x:150,y:0}], W, H).ok);
check("bowtie order", !G.solveHomography([corners[0], corners[2], corners[1], corners[3]], W, H).ok);
check("zero width", !G.solveHomography(corners, 0, H).ok);

const bt = [project(0, 0), project(100, 60), project(100, 0), project(0, 60)];
if (r.ok) {
  const m = G.measurePolygon(r.H, bt);
  check("self-intersect blocked", !m.valid && m.errors[0].includes("自交"), JSON.stringify(m.errors));
}
const si = G.polygonSelfIntersects([{x:0,y:0},{x:10,y:10},{x:10,y:0},{x:0,y:10}]);
check("bowtie detection", si.hit);

console.log("== JS ring topology (holes) — world coords ==");
const outer = [{x:0,y:0},{x:100,y:0},{x:100,y:60},{x:0,y:60}];
const h1 = [{x:20,y:10},{x:40,y:10},{x:40,y:30},{x:20,y:30}];
const h2 = [{x:60,y:20},{x:80,y:20},{x:80,y:40},{x:60,y:40}];
check("valid two holes", G.validateRingTopology([outer, h1, h2]).length === 0,
  JSON.stringify(G.validateRingTopology([outer, h1, h2])));

// hole fully outside outer ring (no edge crossing)
const outside = [{x:105,y:65},{x:120,y:65},{x:120,y:80},{x:105,y:80}];
const outErrs = G.validateRingTopology([outer, outside]);
check("hole outside rejected", /没有完全位于外环内/.test(outErrs.join("")), JSON.stringify(outErrs));

// hole crossing the outer boundary
const crossing = [{x:90,y:10},{x:110,y:10},{x:110,y:30},{x:90,y:30}];
check("hole crossing outer rejected", /相交或相切/.test(G.validateRingTopology([outer, crossing])[0] || ""),
  G.validateRingTopology([outer, crossing]));

// hole vertex exactly on outer edge -> tangency
const tangent = [{x:20,y:0},{x:40,y:0},{x:40,y:20},{x:20,y:20}];
check("tangent hole rejected", /相切/.test(G.validateRingTopology([outer, tangent])[0] || ""),
  G.validateRingTopology([outer, tangent]));

// holes intersect each other
const hb1 = [{x:20,y:20},{x:50,y:20},{x:50,y:45},{x:20,y:45}];
const hb2 = [{x:45,y:25},{x:70,y:25},{x:70,y:50},{x:45,y:50}];
check("holes crossing rejected", /孔洞 1 与孔洞 2 相交或相切/.test((G.validateRingTopology([outer, hb1, hb2])[0] || "")));

// nested holes
const nested1 = [{x:20,y:10},{x:80,y:10},{x:80,y:50},{x:20,y:50}];
const nested2 = [{x:40,y:20},{x:60,y:20},{x:60,y:40},{x:40,y:40}];
check("nested holes rejected", /包含/.test(G.validateRingTopology([outer, nested1, nested2]).join("")));

// hole self-intersection
const hs = [{x:20,y:10},{x:40,y:30},{x:40,y:10},{x:20,y:30}];
check("self-cross hole rejected", /孔洞 1自交/.test(G.validateRingTopology([outer, hs]).join("")));

console.log("== JS region measurement (perspective) ==");
if (r.ok) {
  // outer [10,90]x[10,50] area 80*40=3200 perim 240;
  // hole  [30,50]x[20,40] area 20*20=400  perim 80
  const oImg = [[10,10],[90,10],[90,50],[10,50]].map(([x,y]) => project(x,y));
  const holeImg = [[30,20],[50,20],[50,40],[30,40]].map(([x,y]) => project(x,y));
  const m = G.measureRegion(r.H, oImg, [holeImg]);
  check("region valid", m.valid, JSON.stringify(m.errors));
  check("net area 2800", m.valid && Math.abs(m.area - 2800) < 1e-4, `area=${m.area}`);
  check("gross area 3200", m.valid && Math.abs(m.grossArea - 3200) < 1e-4, `gross=${m.grossArea}`);
  check("hole area 400", m.valid && Math.abs(m.holeAreas[0] - 400) < 1e-4, JSON.stringify(m.holeAreas));
  check("perimeter 320", m.valid && Math.abs(m.perimeter - 320) < 1e-4, `p=${m.perimeter}`);

  // hole partly outside -> invalid, no area
  const badHole = [[80,40],[95,40],[95,55],[80,55]].map(([x,y]) => project(x,y));
  const mb = G.measureRegion(r.H, oImg, [badHole]);
  check("outside hole blocked", !mb.valid && /外环/.test(mb.errors.join("")), JSON.stringify(mb.errors));

  // plain polygon via wrapper still works
  const mw = G.measurePolygon(r.H, [[20,10],[70,10],[70,50],[20,50]].map(([x,y]) => project(x,y)));
  check("wrapper area 2000", mw.valid && Math.abs(mw.area - 2000) < 1e-4, `area=${mw.area}`);
}

console.log(fails ? `\n${fails} FAILURES` : "\nALL JS PARITY TESTS PASSED");
process.exit(fails ? 1 : 0);
