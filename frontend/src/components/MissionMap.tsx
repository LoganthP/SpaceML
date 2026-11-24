import { Eye, Globe2, Minus, Pause, Play, Plus, RotateCcw, RotateCw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/examples/jsm/renderers/CSS2DRenderer.js";
import { useTheme } from "../lib/theme";

export interface MapSat { id: string; name: string; color: string; installed: boolean; path: string }
export interface Hotspot { id: "telemetry" | "exoplanet" | "mars" | "spaceweather" | "workbench" | "webb" | "missions"; title: string; sub: string; path: string; color: string; ready: boolean; sats?: MapSat[] }

function meanLon(planet: "earth" | "mars") {
  const d = (Date.now() - Date.UTC(2000, 0, 1, 12)) / 86400000;
  return THREE.MathUtils.degToRad(((planet === "earth" ? 100.46 + 0.9856474 * d : 355.45 + 0.5240329 * d) % 360 + 360) % 360);
}

/* ---------------------------------------------------------------- spacecraft models (schematic, not to scale) */
const std = (color: number, extra: Partial<THREE.MeshStandardMaterialParameters> = {}) =>
  new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.55, ...extra });
const panelMat = () => std(0x1d4ed8, { emissive: 0x1e3a8a, emissiveIntensity: 0.5, metalness: 0.3 });

function solarWings(span: number, w: number, h: number) {
  const g = new THREE.Group();
  for (const s of [-1, 1]) {
    const p = new THREE.Mesh(new THREE.BoxGeometry(w, h * 0.06, h), panelMat());
    p.position.x = s * (span / 2 + w / 2);
    g.add(p);
    const arm = new THREE.Mesh(new THREE.CylinderGeometry(h * 0.04, h * 0.04, span / 2, 6), std(0xcbd5e1));
    arm.rotation.z = Math.PI / 2;
    arm.position.x = s * span / 4;
    g.add(arm);
  }
  return g;
}

/** Hubble: silver tube with an aperture door and two solar arrays. */
function hubble() {
  const g = new THREE.Group();
  const tube = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.22, 20), std(0xe2e8f0, { metalness: 0.8 }));
  tube.rotation.z = Math.PI / 2;
  g.add(tube);
  const aft = new THREE.Mesh(new THREE.CylinderGeometry(0.056, 0.056, 0.06, 20), std(0x94a3b8));
  aft.rotation.z = Math.PI / 2; aft.position.x = -0.13; g.add(aft);
  const door = new THREE.Mesh(new THREE.CircleGeometry(0.05, 20), std(0x334155, { side: THREE.DoubleSide }));
  door.rotation.y = -Math.PI / 2.6; door.position.set(0.12, 0.03, 0); g.add(door);
  const wings = solarWings(0.12, 0.12, 0.05); wings.rotation.y = Math.PI / 2; wings.position.x = -0.04; g.add(wings);
  return g;
}

/** Webb: 18 gold hexagonal mirror segments, a secondary mirror on struts, and the five-layer sunshield. */
function webb() {
  const g = new THREE.Group();
  const gold = std(0xf5b301, { emissive: 0xb45309, emissiveIntensity: 0.45, metalness: 0.9, roughness: 0.25, side: THREE.DoubleSide });
  const hex = new THREE.CircleGeometry(0.03, 6);
  const mirror = new THREE.Group();
  const r = 0.03 * Math.sqrt(3);
  const cells: [number, number][] = [];
  for (let q = -2; q <= 2; q++) for (let s = -2; s <= 2; s++) {
    const t = -q - s;
    if (Math.max(Math.abs(q), Math.abs(s), Math.abs(t)) <= 2 && !(q === 0 && s === 0)) cells.push([q, s]);
  }
  cells.slice(0, 18).forEach(([q, s]) => {
    const m = new THREE.Mesh(hex, gold);
    m.position.set(r * (q + s / 2), r * (s * Math.sqrt(3) / 2), 0);
    m.rotation.z = Math.PI / 6;
    mirror.add(m);
  });
  mirror.rotation.x = -Math.PI / 2.4;
  mirror.position.y = 0.09;
  g.add(mirror);
  const sec = new THREE.Mesh(new THREE.CircleGeometry(0.014, 6), gold);
  sec.position.set(0, 0.2, 0.08); sec.rotation.x = Math.PI / 2.6; g.add(sec);
  const strut = std(0x94a3b8);
  for (const x of [-0.06, 0.06]) {
    const s = new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.003, 0.16, 4), strut);
    s.position.set(x / 2, 0.15, 0.04); s.rotation.z = -x * 4; g.add(s);
  }
  const shape = new THREE.Shape();
  shape.moveTo(0, 0.2); shape.lineTo(0.32, 0); shape.lineTo(0, -0.2); shape.lineTo(-0.32, 0); shape.closePath();
  const shield = new THREE.ShapeGeometry(shape);
  for (let i = 0; i < 5; i++) {
    const layer = new THREE.Mesh(shield, std(i === 0 ? 0xc4b5fd : 0xe9d5ff, { side: THREE.DoubleSide, metalness: 0.7, roughness: 0.3,
      transparent: true, opacity: 0.85 - i * 0.08 }));
    layer.rotation.x = -Math.PI / 2;
    layer.position.y = -0.005 - i * 0.012;
    layer.scale.setScalar(1 - i * 0.04);
    g.add(layer);
  }
  return g;
}

/** Kepler: photometer barrel with an angled sunshade and a fixed solar array. */
function kepler() {
  const g = new THREE.Group();
  const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.2, 20), std(0x1f2937, { metalness: 0.3 }));
  g.add(barrel);
  const shade = new THREE.Mesh(new THREE.CylinderGeometry(0.062, 0.062, 0.08, 20, 1, true, 0, Math.PI), std(0x9ca3af, { side: THREE.DoubleSide }));
  shade.position.y = 0.13; g.add(shade);
  const array = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.004, 0.14), panelMat());
  array.position.set(0, -0.08, -0.08); array.rotation.x = 0.5; g.add(array);
  return g;
}

/** SMAP: spacecraft bus with the large spinning mesh reflector. */
function smap() {
  const g = new THREE.Group();
  g.add(new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.06, 0.06), std(0xe5e7eb)));
  const dish = new THREE.Mesh(new THREE.TorusGeometry(0.09, 0.006, 6, 32), std(0xfbbf24, { metalness: 0.8 }));
  dish.position.y = 0.09; dish.rotation.x = Math.PI / 2.3; g.add(dish);
  const boom = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 0.09, 4), std(0xcbd5e1));
  boom.position.y = 0.05; g.add(boom);
  const w = solarWings(0.06, 0.08, 0.04); w.position.y = -0.02; g.add(w);
  return g;
}

/** A CubeSat-style bus (OPS-SAT) or a small probe with a dish. */
function cubesat(color: number, dish = false) {
  const g = new THREE.Group();
  g.add(new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.05, 0.07), std(color, { metalness: 0.4 })));
  g.add(solarWings(0.05, 0.06, 0.045));
  if (dish) {
    const d = new THREE.Mesh(new THREE.SphereGeometry(0.035, 16, 8, 0, Math.PI * 2, 0, Math.PI / 3.2), std(0xe5e7eb, { side: THREE.DoubleSide }));
    d.position.y = 0.04; g.add(d);
  }
  return g;
}

const SAT_MODEL: Record<string, () => THREE.Group> = {
  opssat: () => cubesat(0x22c55e), hubble,
};

/**
 * The inner Solar System as a mission map: every module sits where its data is collected. SMAP orbits Earth
 * (telemetry), ACE/DSCOVR sit at L1 (space weather), Kepler trails Earth staring at Cygnus (exoplanets), Curiosity is on
 * Mars, Webb sits at Sun-Earth L2 and the Earth-orbit missions (OPS-SAT, Hubble) circle Earth.
 *
 * The scene is built once per theme; label text updates in place, so polling never resets the camera. Drag / arrow keys
 * rotate, wheel / + - zoom, the buttons pause, reset or fly to Earth orbit, and every spacecraft is clickable.
 */
export default function MissionMap({ spots, height = 460 }: { spots: Hotspot[]; height?: number }) {
  const theme = useTheme();
  const host = useRef<HTMLDivElement>(null);
  const nav = useNavigate();
  const navRef = useRef(nav);
  navRef.current = nav;
  const spotsRef = useRef(spots);
  spotsRef.current = spots;
  const api = useRef<{ rotate: (a: number) => void; zoom: (f: number) => void; reset: () => void; earth: (on: boolean) => void;
    setAuto: (on: boolean) => void; refresh: () => void; focus: (id: string | null) => void } | null>(null);
  const [auto, setAuto] = useState(true);
  const [earthView, setEarthView] = useState(false);
  const [hover, setHover] = useState<string | null>(null);
  const [fleet, setFleet] = useState<{ id: string; name: string; path: string; color: string }[]>([]);
  const [focused, setFocused] = useState<string | null>(null);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const dark = theme === "dark";
    let w = el.clientWidth;
    const h = height;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
    renderer.domElement.style.touchAction = "none";
    renderer.domElement.style.cursor = "grab";
    el.appendChild(renderer.domElement);
    const labels = new CSS2DRenderer();
    labels.setSize(w, h);
    Object.assign(labels.domElement.style, { position: "absolute", inset: "0", pointerEvents: "none" });
    el.appendChild(labels.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, w / h, 0.01, 400);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.minDistance = 1.2;
    controls.maxDistance = 26;
    controls.maxPolarAngle = Math.PI * 0.49;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.18;
    controls.addEventListener("start", () => { renderer.domElement.style.cursor = "grabbing"; });
    controls.addEventListener("end", () => { renderer.domElement.style.cursor = "grab"; });
    scene.add(new THREE.AmbientLight(0xffffff, dark ? 0.55 : 0.9));
    scene.add(new THREE.PointLight(0xfff1d6, dark ? 90 : 60, 0, 1.4));
    const fill = new THREE.DirectionalLight(0xffffff, 0.6); fill.position.set(0, 10, 6); scene.add(fill);

    // hotspot tags: created once, text refreshed in place
    const tags = new Map<string, HTMLButtonElement>();
    const paintTag = (id: string) => {
      const e = tags.get(id);
      const s = spotsRef.current.find((x) => x.id === id);
      if (!e) return;
      e.style.display = s ? "" : "none";
      if (!s) return;
      e.className = `map-tag ${s.ready ? "" : "map-tag-off"}`;
      e.style.setProperty("--c", s.color);
      e.innerHTML = `<span class="map-dot"></span><b></b><i></i>`;
      (e.querySelector("b") as HTMLElement).textContent = s.title;
      (e.querySelector("i") as HTMLElement).textContent = s.sub;
      e.title = `${s.title} - ${s.sub}. Open`;
    };
    const hud = (id: Hotspot["id"]) => {
      const e = document.createElement("button");
      e.style.pointerEvents = "auto";
      e.onclick = () => { const s = spotsRef.current.find((x) => x.id === id); if (s) navRef.current(s.path); };
      e.onpointerenter = () => { controls.autoRotate = false; };
      e.onpointerleave = () => { controls.autoRotate = autoRef.current; };
      tags.set(id, e);
      paintTag(id);
      return new CSS2DObject(e);
    };
    const autoRef = { current: true };
    // round, soft points (square GL points look like random blocks when you zoom in)
    const dotCanvas = document.createElement("canvas"); dotCanvas.width = dotCanvas.height = 32;
    const dc = dotCanvas.getContext("2d")!;
    const dg = dc.createRadialGradient(16, 16, 0, 16, 16, 16);
    dg.addColorStop(0, "rgba(255,255,255,1)"); dg.addColorStop(0.5, "rgba(255,255,255,0.8)"); dg.addColorStop(1, "rgba(255,255,255,0)");
    dc.fillStyle = dg; dc.fillRect(0, 0, 32, 32);
    const dotTex = new THREE.CanvasTexture(dotCanvas);
    const dots = (p: THREE.PointsMaterialParameters) => new THREE.PointsMaterial({ map: dotTex, transparent: true, depthWrite: false, alphaTest: 0.02, ...p });
    const small = (text: string, cls = "scene-label") => { const e = document.createElement("div"); e.className = cls; e.textContent = text; return new CSS2DObject(e); };

    // starfield
    const sp: number[] = [];
    for (let i = 0; i < 1400; i++) { const v = new THREE.Vector3().randomDirection().multiplyScalar(60 + Math.random() * 60); sp.push(v.x, v.y, v.z); }
    const sg = new THREE.BufferGeometry(); sg.setAttribute("position", new THREE.Float32BufferAttribute(sp, 3));
    scene.add(new THREE.Points(sg, dots({ color: dark ? 0xe0e7ff : 0x818cf8, size: dark ? 0.45 : 0.35, opacity: dark ? 0.9 : 0.45 })));

    // Sun
    scene.add(new THREE.Mesh(new THREE.SphereGeometry(0.62, 48, 48), new THREE.MeshBasicMaterial({ color: 0xfbbf24 })));
    const gc = document.createElement("canvas"); gc.width = gc.height = 128;
    const g2 = gc.getContext("2d")!;
    const grd = g2.createRadialGradient(64, 64, 0, 64, 64, 64);
    grd.addColorStop(0, "rgba(253,224,71,1)"); grd.addColorStop(0.25, "rgba(251,146,60,0.45)"); grd.addColorStop(1, "rgba(251,146,60,0)");
    g2.fillStyle = grd; g2.fillRect(0, 0, 128, 128);
    const glowTex = new THREE.CanvasTexture(gc);
    const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, depthWrite: false, blending: dark ? THREE.AdditiveBlending : THREE.NormalBlending }));
    glow.scale.setScalar(4.2); scene.add(glow);
    const sunTag = small("Sun"); sunTag.position.set(0, -0.95, 0); scene.add(sunTag);

    const ring = (r: number, color: number, op: number, dashed = false, parent: THREE.Object3D = scene) => {
      const pts = new THREE.EllipseCurve(0, 0, r, r, 0, Math.PI * 2).getPoints(240).map((v) => new THREE.Vector3(v.x, 0, v.y));
      const mat = dashed ? new THREE.LineDashedMaterial({ color, dashSize: 0.2, gapSize: 0.15, transparent: true, opacity: op })
        : new THREE.LineBasicMaterial({ color, transparent: true, opacity: op });
      const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat);
      if (dashed) l.computeLineDistances();
      parent.add(l);
      return l;
    };
    const RE = 3.4, RM = 5.3;
    ring(RE, 0x60a5fa, dark ? 0.55 : 0.7);
    ring(RM, 0xfb923c, dark ? 0.5 : 0.65);
    ring(1.9, 0x94a3b8, 0.25, true);
    const ap: number[] = [];
    for (let i = 0; i < 900; i++) { const a = Math.random() * Math.PI * 2, r = 6.6 + Math.random() * 1.1; ap.push(Math.cos(a) * r, (Math.random() - 0.5) * 0.25, Math.sin(a) * r); }
    const ag = new THREE.BufferGeometry(); ag.setAttribute("position", new THREE.Float32BufferAttribute(ap, 3));
    scene.add(new THREE.Points(ag, dots({ color: dark ? 0x94a3b8 : 0xa8a29e, size: 0.06, opacity: 0.6 })));

    // clickable spacecraft
    const pickables: { obj: THREE.Object3D; name: string; path: string }[] = [];
    const addPick = (obj: THREE.Object3D, name: string, path: string) => pickables.push({ obj, name, path });

    // Earth system
    const earthPivot = new THREE.Group(); earthPivot.rotation.y = meanLon("earth"); scene.add(earthPivot);
    const earth = new THREE.Group(); earth.position.set(RE, 0, 0); earth.scale.setScalar(1.5); earthPivot.add(earth);
    // every spacecraft, for the fleet bar (fly to it) and for its always-visible name tag
    const craft: { id: string; name: string; path: string; color: string; obj: THREE.Object3D; dist: number }[] = [];
    const nameTags: HTMLElement[] = [];
    const nameTag = (obj: THREE.Object3D, text: string, path: string, y = 0.12) => {
      const lab = small(text, "scene-label sat-label");
      lab.element.style.pointerEvents = "auto";
      lab.element.onclick = () => navRef.current(path);
      lab.position.set(0, y, 0); obj.add(lab); nameTags.push(lab.element);
    };
    earth.add(new THREE.Mesh(new THREE.SphereGeometry(0.26, 40, 40), new THREE.MeshStandardMaterial({ color: 0x3b82f6, roughness: 0.45, emissive: 0x1e40af, emissiveIntensity: 0.35 })));
    earth.add(new THREE.Mesh(new THREE.SphereGeometry(0.3, 32, 32), new THREE.MeshBasicMaterial({ color: 0x93c5fd, transparent: true, opacity: 0.15, side: THREE.BackSide })));
    const wb = hud("workbench"); wb.position.set(-1.1, 0.25, 0.35); earth.add(wb);

    // SMAP (telemetry) on its polar orbit
    const smapTilt = new THREE.Group(); smapTilt.rotation.x = THREE.MathUtils.degToRad(80); earth.add(smapTilt);
    ring(0.55, 0x22d3ee, 0.6, false, smapTilt);
    const smapPivot = new THREE.Group(); smapTilt.add(smapPivot);
    const smapObj = smap(); smapObj.scale.setScalar(0.9); smapObj.position.set(0.55, 0, 0); smapPivot.add(smapObj);
    addPick(smapObj, "SMAP (soil-moisture satellite) - telemetry anomaly detection", "/telemetry");
    nameTag(smapObj, "SMAP", "/telemetry");
    craft.push({ id: "smap", name: "SMAP", path: "/telemetry", color: "#22d3ee", obj: smapObj, dist: 2.2 });
    const tel = hud("telemetry"); tel.position.set(0.2, 0.8, 0.15); earth.add(tel);

    // Earth-orbit missions, each on its own inclined orbit
    const leoPivots: THREE.Group[] = [];
    const satMeshes = new Map<string, THREE.Group>();
    const sats = (spotsRef.current.find((s) => s.id === "missions")?.sats) ?? [];
    sats.forEach((m, i) => {
      const tilt = new THREE.Group();
      tilt.rotation.set(THREE.MathUtils.degToRad(25 + i * 23), THREE.MathUtils.degToRad(i * 47), 0);
      earth.add(tilt);
      const r = 0.46 + i * 0.13;
      ring(r, new THREE.Color(m.color).getHex(), m.installed ? 0.5 : 0.2, true, tilt);
      const pv = new THREE.Group(); pv.rotation.y = i * 1.3; tilt.add(pv); leoPivots.push(pv);
      const model = (SAT_MODEL[m.id] ?? (() => cubesat(0x94a3b8)))();
      model.scale.setScalar(m.id === "hubble" ? 1.05 : 1.45);
      model.position.set(r, 0, 0);
      pv.add(model);
      satMeshes.set(m.id, model);
      addPick(model, m.name, m.path);
      nameTag(model, m.name, m.path, 0.1);
      craft.push({ id: m.id, name: m.name, path: m.path, color: m.color, obj: model, dist: 2.0 });
    });
    if (spotsRef.current.some((s) => s.id === "missions")) { const lt = hud("missions"); lt.position.set(0.15, -0.95, 0.35); earth.add(lt); }

    // Webb at Sun-Earth L2 (anti-sunward)
    const webbObj = webb(); webbObj.scale.setScalar(0.6); webbObj.position.set(1.15, 0.05, 0);
    webbObj.rotation.z = Math.PI / 2;                     // sunshield faces the Sun, mirror faces away
    earth.add(webbObj);
    addPick(webbObj, "James Webb Space Telescope at Sun-Earth L2", "/telemetry?mission=webb");
    craft.push({ id: "webb", name: "James Webb", path: "/telemetry?mission=webb", color: "#facc15", obj: webbObj, dist: 2.4 });
    const webbTag = hud("webb"); webbTag.position.set(1.3, -0.45, -0.1); earth.add(webbTag);

    // L1 + solar wind (space weather)
    const l1 = cubesat(0xf59e0b, true); l1.scale.setScalar(1.5); l1.position.set(-1.0, 0, 0); earth.add(l1);
    addPick(l1, "ACE / DSCOVR at Sun-Earth L1 - solar wind", "/space-weather");
    nameTag(l1, "ACE · DSCOVR", "/space-weather", 0.1);
    craft.push({ id: "l1", name: "ACE · DSCOVR (L1)", path: "/space-weather", color: "#f59e0b", obj: l1, dist: 2.4 });
    const swTag = hud("spaceweather"); swTag.position.set(-0.65, 1.15, -0.9); earth.add(swTag);
    const windN = 260, wp = new Float32Array(windN * 3);
    const seeds = Array.from({ length: windN }, () => ({ t: Math.random(), a: (Math.random() - 0.5) * 0.35, y: (Math.random() - 0.5) * 0.4 }));
    const wg = new THREE.BufferGeometry(); wg.setAttribute("position", new THREE.BufferAttribute(wp, 3));
    earthPivot.add(new THREE.Points(wg, dots({ color: 0xfbbf24, size: 3, sizeAttenuation: false, opacity: 0.7 })));

    // Kepler trailing Earth, staring at Cygnus
    const kepPivot = new THREE.Group(); kepPivot.rotation.y = -0.42; earthPivot.add(kepPivot);
    const kep = new THREE.Group(); kep.position.set(RE * 1.01, 0, 0); kepPivot.add(kep);
    const coneDir = new THREE.Vector3(-0.35, 0.9, 0.25).normalize();
    const kepObj = kepler(); kepObj.scale.setScalar(1.6); kepObj.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), coneDir); kep.add(kepObj);
    addPick(kepObj, "Kepler space telescope - exoplanet transits", "/exoplanet");
    craft.push({ id: "kepler", name: "Kepler", path: "/exoplanet", color: "#a78bfa", obj: kepObj, dist: 3.2 });
    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.75, 3.4, 32, 1, true), new THREE.MeshBasicMaterial({ color: 0xa78bfa, transparent: true, opacity: dark ? 0.14 : 0.11, side: THREE.DoubleSide, depthWrite: false }));
    cone.position.copy(coneDir.clone().multiplyScalar(1.85)); cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, -1, 0), coneDir); kep.add(cone);
    const exo = hud("exoplanet"); exo.position.copy(coneDir.clone().multiplyScalar(3.6)); kep.add(exo);

    // Mars + Curiosity
    const marsPivot = new THREE.Group(); marsPivot.rotation.y = meanLon("mars"); scene.add(marsPivot);
    const mars = new THREE.Group(); mars.position.set(RM, 0, 0); marsPivot.add(mars);
    mars.add(new THREE.Mesh(new THREE.SphereGeometry(0.2, 40, 40), new THREE.MeshStandardMaterial({ color: 0xea580c, roughness: 0.85, emissive: 0x7c2d12, emissiveIntensity: 0.25 })));
    const rover = new THREE.Group();
    rover.add(new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.025, 0.05), std(0xf8fafc)));
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 0.05, 4), std(0xf8fafc)); mast.position.set(0.02, 0.035, 0); rover.add(mast);
    rover.position.set(0, 0.215, 0); mars.add(rover);
    addPick(rover, "Curiosity rover, Gale crater - terrain", "/mars");
    craft.push({ id: "curiosity", name: "Curiosity (Mars)", path: "/mars", color: "#f97316", obj: mars, dist: 3.0 });
    const mt = hud("mars"); mt.position.set(0, 0.85, 0); mars.add(mt);

    // camera
    const home = () => {
      const eAng = earthPivot.rotation.y;
      const camDir = new THREE.Vector3(Math.cos(eAng), 0, -Math.sin(eAng)).applyAxisAngle(new THREE.Vector3(0, 1, 0), 0.5);
      camera.position.copy(camDir.multiplyScalar(w < 640 ? 15 : 10.5)).add(new THREE.Vector3(0, 6.2, 0));
      controls.target.set(0, -0.3, 0);
    };
    home();
    let focusEarth = false;
    let focusObj: THREE.Object3D | null = null;
    const earthWorld = new THREE.Vector3();
    const focusWorld = new THREE.Vector3(), prevFocus = new THREE.Vector3();
    setFleet(craft.map(({ id, name, path, color }) => ({ id, name, path, color })));

    // hover + click on spacecraft
    const ray = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    let downAt = { x: 0, y: 0 };
    const pick = (ev: PointerEvent) => {
      const r = renderer.domElement.getBoundingClientRect();
      ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(ndc, camera);
      const hits = ray.intersectObjects(pickables.map((p) => p.obj), true);
      if (!hits.length) return null;
      let o: THREE.Object3D | null = hits[0].object;
      while (o && !pickables.some((p) => p.obj === o)) o = o.parent;
      return pickables.find((p) => p.obj === o) ?? null;
    };
    const onMove = (ev: PointerEvent) => {
      const p = pick(ev);
      renderer.domElement.style.cursor = p ? "pointer" : "grab";
      setHover(p ? p.name : null);
    };
    const onDown = (ev: PointerEvent) => { downAt = { x: ev.clientX, y: ev.clientY }; };
    const onUp = (ev: PointerEvent) => {
      if (Math.hypot(ev.clientX - downAt.x, ev.clientY - downAt.y) > 5) return;   // a drag, not a click
      const p = pick(ev);
      if (p) navRef.current(p.path);
    };
    renderer.domElement.addEventListener("pointermove", onMove);
    renderer.domElement.addEventListener("pointerdown", onDown);
    renderer.domElement.addEventListener("pointerup", onUp);
    renderer.domElement.addEventListener("pointerleave", () => setHover(null));

    api.current = {
      rotate: (a) => {
        const off = camera.position.clone().sub(controls.target).applyAxisAngle(new THREE.Vector3(0, 1, 0), a);
        camera.position.copy(controls.target).add(off);
      },
      zoom: (f) => {
        const off = camera.position.clone().sub(controls.target);
        const len = THREE.MathUtils.clamp(off.length() * f, controls.minDistance, controls.maxDistance);
        camera.position.copy(controls.target).add(off.setLength(len));
      },
      reset: () => { focusEarth = false; focusObj = null; home(); },
      focus: (id) => {
        const c = craft.find((x) => x.id === id);
        focusObj = c?.obj ?? null;
        focusEarth = false;
        if (!c) { home(); return; }
        c.obj.getWorldPosition(focusWorld);
        prevFocus.copy(focusWorld);
        const off = camera.position.clone().sub(controls.target).setLength(c.dist);
        off.y = Math.max(off.y, c.dist * 0.35);
        controls.target.copy(focusWorld);
        camera.position.copy(focusWorld).add(off);
      },
      earth: (on) => {
        focusEarth = on; focusObj = null;
        if (on) {
          earth.getWorldPosition(earthWorld);
          const off = camera.position.clone().sub(controls.target).setLength(5.6);
          off.y = Math.max(off.y, 2.4);
          controls.target.copy(earthWorld);
          camera.position.copy(earthWorld).add(off);
        } else home();
      },
      setAuto: (on) => { autoRef.current = on; controls.autoRotate = on; },
      refresh: () => {
        tags.forEach((_, id) => paintTag(id));
        const cur = (spotsRef.current.find((s) => s.id === "missions")?.sats) ?? [];
        cur.forEach((m) => satMeshes.get(m.id)?.traverse((o) => {
          const mm = (o as THREE.Mesh).material as THREE.MeshStandardMaterial | undefined;
          if (mm && "opacity" in mm) { mm.transparent = true; mm.opacity = m.installed ? 1 : 0.55; }
        }));
      },
    };
    api.current.refresh();

    // keep labels readable: a label that would cover another one is moved down until it is clear
    let frame = 0;
    const declutter = () => {
      const big = new Set<HTMLElement>(tags.values());
      const els = [...tags.values(), ...nameTags].filter((e) => e.offsetParent !== null && e.style.display !== "none" && e.parentElement?.style.display !== "none");
      const boxes = els.map((e) => {
        const prev = Number(e.dataset.dy || 0);
        const r = e.getBoundingClientRect();
        return { e, x: r.left, y: r.top - prev, w: r.width, h: r.height, big: big.has(e) };
      }).filter((b) => b.w > 0).sort((p, q) => (p.big === q.big ? p.y - q.y : p.big ? -1 : 1));
      const placed: { x: number; y: number; w: number; h: number }[] = [];
      for (const b of boxes) {
        let y = b.y;
        for (let k = 0; k < 12; k++) {
          const hit = placed.find((p) => b.x < p.x + p.w + 2 && b.x + b.w + 2 > p.x && y < p.y + p.h + 2 && y + b.h + 2 > p.y);
          if (!hit) break;
          y = hit.y + hit.h + 3;
        }
        placed.push({ x: b.x, y, w: b.w, h: b.h });
        const dy = Math.round(y - b.y);
        if (String(dy) !== (b.e.dataset.dy || "0")) { b.e.dataset.dy = String(dy); b.e.style.translate = dy ? `0 ${dy}px` : ""; }
      }
    };
    let raf = 0, last = performance.now();
    const prevEarth = new THREE.Vector3();
    earth.getWorldPosition(prevEarth);
    const tick = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      earthPivot.rotation.y += dt * 0.025;
      marsPivot.rotation.y += dt * 0.0133;
      smapPivot.rotation.y += dt * 1.6;
      leoPivots.forEach((p, i) => { p.rotation.y += dt * (1.1 + i * 0.17); });
      webbObj.rotation.y += dt * 0.15;
      seeds.forEach((s, i) => {
        s.t = (s.t + dt * 0.18) % 1;
        const r = 0.7 + s.t * (RE - 0.75), ang = s.a * (1 - s.t * 0.7);
        wp.set([Math.cos(ang) * r, s.y * s.t, Math.sin(ang) * r], i * 3);
      });
      wg.attributes.position.needsUpdate = true;
      earth.getWorldPosition(earthWorld);
      if (focusEarth) {                                   // follow Earth along its orbit
        const d = earthWorld.clone().sub(prevEarth);
        controls.target.add(d); camera.position.add(d);
      }
      prevEarth.copy(earthWorld);
      if (focusObj) {                                     // follow the chosen spacecraft
        focusObj.getWorldPosition(focusWorld);
        const d = focusWorld.clone().sub(prevFocus);
        controls.target.add(d); camera.position.add(d);
        prevFocus.copy(focusWorld);
      }
      controls.update();
      renderer.render(scene, camera);
      labels.render(scene, camera);
      if (++frame % 3 === 0) declutter();
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    const ro = new ResizeObserver(() => {
      w = el.clientWidth;
      renderer.setSize(w, h); labels.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    });
    ro.observe(el);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      api.current = null;
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose?.();
        const mat = m.material as THREE.Material | THREE.Material[] | undefined;
        (Array.isArray(mat) ? mat : mat ? [mat] : []).forEach((x) => x.dispose());
      });
      glowTex.dispose();
      dotTex.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      el.removeChild(labels.domElement);
    };
  }, [height, theme, spots.some((s) => s.id === "missions") ? (spots.find((s) => s.id === "missions")?.sats ?? []).map((s) => s.id).join(",") : ""]); // eslint-disable-line react-hooks/exhaustive-deps

  // new data (catalog polling) only repaints labels - the scene and the camera stay where the user put them
  useEffect(() => { api.current?.refresh(); }, [spots]);

  const key = (e: React.KeyboardEvent) => {
    const a = api.current;
    if (!a) return;
    const k = e.key;
    if (k === "ArrowLeft") a.rotate(-0.15); else if (k === "ArrowRight") a.rotate(0.15);
    else if (k === "+" || k === "=" || k === "ArrowUp") a.zoom(0.85); else if (k === "-" || k === "ArrowDown") a.zoom(1.18);
    else if (k === "0" || k === "Home") { a.reset(); setEarthView(false); setFocused(null); } else return;
    e.preventDefault();
    a.setAuto(false); setAuto(false);
  };
  const btn = "grid h-8 w-8 place-items-center rounded-lg text-ink hover:bg-white/70 focus-visible:outline focus-visible:outline-2 focus-visible:outline-violet-500 dark:hover:bg-white/10";
  return (
    <div className="relative w-full">
      <div ref={host} tabIndex={0} onKeyDown={key} role="application"
        aria-label="Mission map. Drag or use the arrow keys to rotate, scroll or plus and minus to zoom, 0 to reset. Click a spacecraft or a label to open its data."
        className="relative w-full outline-none focus-visible:ring-2 focus-visible:ring-violet-500" style={{ height }} />
      {focused && (() => { const f = fleet.find((x) => x.id === focused); return f && (
        <div className="absolute right-3 top-3 z-10 flex items-center gap-2 rounded-xl border border-white/60 bg-white/85 px-3 py-1.5 text-xs shadow backdrop-blur dark:border-white/10 dark:bg-slate-900/85" role="status">
          <span className="h-2 w-2 rounded-full" style={{ background: f.color }} aria-hidden="true" /><b>{f.name}</b>
          <button type="button" className="btn-primary px-2 py-0.5 text-[11px]" onClick={() => nav(f.path)}>Open</button>
          <button type="button" className="text-ink-soft hover:underline" onClick={() => { api.current?.focus(null); setFocused(null); }}>Back</button>
        </div>); })()}
      {hover && <div className="pointer-events-none absolute left-1/2 top-2 -translate-x-1/2 rounded-full bg-slate-900/85 px-3 py-1 text-xs font-semibold text-white" role="status">{hover} - click to open</div>}
      <div className="relative z-10 flex flex-wrap items-start gap-2 px-3 pb-3 pt-1">
        <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1 rounded-xl border border-white/60 bg-white/80 p-1 shadow-sm backdrop-blur dark:border-white/10 dark:bg-slate-900/80" role="toolbar" aria-label="Fly to a spacecraft">
          {fleet.map((f) => (
            <button key={f.id} type="button" aria-pressed={focused === f.id}
              onClick={() => { const id = focused === f.id ? null : f.id; api.current?.focus(id); api.current?.setAuto(false); setAuto(false); setFocused(id); setEarthView(false); }}
              className={`flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] font-semibold transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-violet-500 ${focused === f.id ? "bg-violet-600 text-white" : "text-ink hover:bg-white/70 dark:hover:bg-white/10"}`}>
              <span className="h-2 w-2 rounded-full" style={{ background: f.color }} aria-hidden="true" />{f.name}
            </button>))}
        </div>
        <div className="flex shrink-0 items-center gap-0.5 rounded-xl border border-white/60 bg-white/80 p-1 shadow-sm backdrop-blur dark:border-white/10 dark:bg-slate-900/80" role="toolbar" aria-label="Map controls">
          <button className={btn} title="Rotate left (←)" aria-label="Rotate left" onClick={() => { api.current?.rotate(-0.35); api.current?.setAuto(false); setAuto(false); }}><RotateCcw className="h-4 w-4" /></button>
          <button className={btn} title="Rotate right (→)" aria-label="Rotate right" onClick={() => { api.current?.rotate(0.35); api.current?.setAuto(false); setAuto(false); }}><RotateCw className="h-4 w-4" /></button>
          <button className={btn} title="Zoom in (+)" aria-label="Zoom in" onClick={() => api.current?.zoom(0.8)}><Plus className="h-4 w-4" /></button>
          <button className={btn} title="Zoom out (−)" aria-label="Zoom out" onClick={() => api.current?.zoom(1.25)}><Minus className="h-4 w-4" /></button>
          <button className={btn} title={auto ? "Pause rotation" : "Resume rotation"} aria-label={auto ? "Pause rotation" : "Resume rotation"} aria-pressed={!auto}
            onClick={() => { const on = !auto; api.current?.setAuto(on); setAuto(on); }}>{auto ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}</button>
          <button className={`${btn} ${earthView ? "bg-violet-600 text-white hover:bg-violet-700" : ""}`} title="Close-up of Earth orbit (Hubble, OPS-SAT, SMAP, Webb)" aria-label="Earth orbit close-up" aria-pressed={earthView}
            onClick={() => { const on = !earthView; api.current?.earth(on); setEarthView(on); setFocused(null); }}><Globe2 className="h-4 w-4" /></button>
          <button className={btn} title="Reset view (0)" aria-label="Reset view" onClick={() => { api.current?.reset(); setEarthView(false); setFocused(null); }}><Eye className="h-4 w-4" /></button>
        </div>
      </div>
    </div>
  );
}
