import { Pause, Play, Crosshair } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/examples/jsm/renderers/CSS2DRenderer.js";
import { useTheme } from "../lib/theme";

export interface GlobeMarker { lat: number; lon: number; label: string; color?: string; primary?: boolean }
export interface GlobeOrbit { label: string; inclinationDeg: number; altitudeKm: number; periodMin: number; color?: string; eccentric?: boolean }

const BODY = {
  earth: { radiusKm: 6371, name: "Earth" },
  mars: { radiusKm: 3389.5, name: "Mars" },
} as const;

/** lat/lon (deg) -> point on a sphere of radius r (y = north). */
export function ll(lat: number, lon: number, r = 1): THREE.Vector3 {
  const phi = THREE.MathUtils.degToRad(90 - lat), th = THREE.MathUtils.degToRad(lon + 180);
  return new THREE.Vector3(-r * Math.sin(phi) * Math.cos(th), r * Math.cos(phi), r * Math.sin(phi) * Math.sin(th));
}

let landCache: Promise<number[][]> | null = null;
const landPoints = () => (landCache ??= fetch("/earth-land-points.json").then((r) => r.json()).then((j) => j.points as number[][]).catch(() => []));

/** Procedural Mars surface: rust base, dark albedo regions, craters, polar caps (deterministic). */
function marsTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 1024; c.height = 512;
  const g = c.getContext("2d")!;
  const base = g.createLinearGradient(0, 0, 0, 512);
  base.addColorStop(0, "#d9a27a"); base.addColorStop(0.5, "#c1643a"); base.addColorStop(1, "#b5704c");
  g.fillStyle = base; g.fillRect(0, 0, 1024, 512);
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < 70; i++) { // dark albedo features (Syrtis-like), mostly mid-latitudes
    const x = rnd() * 1024, y = 160 + rnd() * 220, rx = 30 + rnd() * 120, ry = 10 + rnd() * 40;
    g.fillStyle = `rgba(90,40,25,${0.08 + rnd() * 0.18})`;
    g.beginPath(); g.ellipse(x, y, rx, ry, rnd() * Math.PI, 0, Math.PI * 2); g.fill();
  }
  for (let i = 0; i < 260; i++) { // craters
    const x = rnd() * 1024, y = 40 + rnd() * 432, r = 1 + rnd() ** 3 * 14;
    g.strokeStyle = `rgba(255,220,190,${0.12 + rnd() * 0.15})`; g.lineWidth = 1;
    g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.stroke();
    g.fillStyle = `rgba(80,35,20,${0.12 + rnd() * 0.12})`;
    g.beginPath(); g.arc(x + r * 0.2, y + r * 0.2, r * 0.8, 0, Math.PI * 2); g.fill();
  }
  // Valles Marineris-ish canyon near the equator
  g.strokeStyle = "rgba(80,30,18,0.45)"; g.lineWidth = 5;
  g.beginPath(); g.moveTo(560, 268); g.bezierCurveTo(610, 262, 660, 280, 720, 272); g.stroke();
  const cap = (y0: number, y1: number) => { const gr = g.createLinearGradient(0, y0, 0, y1); gr.addColorStop(0, "rgba(255,255,255,0.95)"); gr.addColorStop(1, "rgba(255,255,255,0)"); g.fillStyle = gr; g.fillRect(0, Math.min(y0, y1), 1024, Math.abs(y1 - y0)); };
  cap(0, 34); cap(512, 486);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function tag(text: string, cls: string) {
  const el = document.createElement("div");
  el.className = cls;
  el.textContent = text;
  return new CSS2DObject(el);
}

/**
 * Where an observation comes from: Earth (dotted continents) or Mars, with surface markers
 * (rover sites, observatories, image footprints) and spacecraft orbits drawn to scale.
 */
export default function ContextGlobe({ body, markers = [], orbits = [], footprint, height = 360, caption }: {
  body: "earth" | "mars"; markers?: GlobeMarker[]; orbits?: GlobeOrbit[];
  footprint?: [number, number, number, number] | null; height?: number; caption?: string;
}) {
  const theme = useTheme();
  const host = useRef<HTMLDivElement>(null);
  const [spin, setSpin] = useState(true);
  const spinRef = useRef(spin);
  spinRef.current = spin;
  const focusRef = useRef<() => void>(() => {});

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const dark = theme === "dark";
    let w = el.clientWidth;
    const h = height;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
    el.appendChild(renderer.domElement);
    const labels = new CSS2DRenderer();
    labels.setSize(w, h);
    Object.assign(labels.domElement.style, { position: "absolute", inset: "0", pointerEvents: "none" });
    el.appendChild(labels.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(36, w / h, 0.01, 100);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.minDistance = 1.5;
    controls.maxDistance = 8;
    scene.add(new THREE.AmbientLight(0xffffff, body === "mars" ? 1.1 : 0.9));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(4, 2, 3);
    scene.add(sun);

    const globe = new THREE.Group();
    scene.add(globe);

    // surface
    let marsTex: THREE.CanvasTexture | null = null;
    if (body === "earth") {
      globe.add(new THREE.Mesh(new THREE.SphereGeometry(0.995, 64, 64),
        new THREE.MeshPhongMaterial({ color: dark ? 0x101850 : 0xeef2ff, emissive: dark ? 0x0b1240 : 0xc7d2fe, emissiveIntensity: dark ? 0.6 : 0.5, transparent: true, opacity: 0.92, shininess: 25 })));
      landPoints().then((pts) => {
        if (!pts.length) return;
        const pos = new Float32Array(pts.length * 3);
        pts.forEach(([la, lo], i) => { const v = ll(la, lo, 1.002); pos.set([v.x, v.y, v.z], i * 3); });
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
        globe.add(new THREE.Points(g, new THREE.PointsMaterial({ color: dark ? 0x7dd3fc : 0x4f46e5, size: 0.032, transparent: true, opacity: 0.95 })));
      });
    } else {
      marsTex = marsTexture();
      globe.add(new THREE.Mesh(new THREE.SphereGeometry(1, 64, 64), new THREE.MeshStandardMaterial({ map: marsTex, roughness: 0.95 })));
    }
    // graticule
    const gm = new THREE.LineBasicMaterial({ color: dark ? 0x4f5bb5 : 0xa5b4fc, transparent: true, opacity: dark ? 0.35 : 0.45 });
    for (let lat = -60; lat <= 60; lat += 30) {
      const p = []; for (let lo = -180; lo <= 180; lo += 4) p.push(ll(lat, lo, 1.004));
      globe.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(p), gm));
    }
    for (let lo = -180; lo < 180; lo += 30) {
      const p = []; for (let la = -90; la <= 90; la += 4) p.push(ll(la, lo, 1.004));
      globe.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(p), gm));
    }
    // atmosphere rim
    const atm = new THREE.Mesh(new THREE.SphereGeometry(1.08, 48, 48), new THREE.MeshBasicMaterial({
      color: body === "mars" ? 0xfdba74 : 0x60a5fa, transparent: true, opacity: dark ? 0.12 : 0.1, side: THREE.BackSide }));
    scene.add(atm);

    // markers
    const pulses: THREE.Mesh[] = [];
    markers.forEach((m) => {
      const col = new THREE.Color(m.color ?? "#7c3aed");
      const p = ll(m.lat, m.lon, 1.01);
      const dot = new THREE.Mesh(new THREE.SphereGeometry(m.primary ? 0.022 : 0.016, 16, 16), new THREE.MeshBasicMaterial({ color: col }));
      dot.position.copy(p);
      globe.add(dot);
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.03, 0.04, 32), new THREE.MeshBasicMaterial({ color: col, side: THREE.DoubleSide, transparent: true }));
      ring.position.copy(p);
      ring.lookAt(p.clone().multiplyScalar(2));
      globe.add(ring);
      pulses.push(ring);
      const spike = new THREE.Line(new THREE.BufferGeometry().setFromPoints([p, ll(m.lat, m.lon, m.primary ? 1.16 : 1.1)]), new THREE.LineBasicMaterial({ color: col }));
      globe.add(spike);
      const t = tag(m.label, `scene-label ${m.primary ? "scene-label-sel" : ""}`);
      t.position.copy(ll(m.lat, m.lon, m.primary ? 1.2 : 1.13));
      globe.add(t);
    });

    // footprint (lat/lon box)
    if (footprint) {
      const [w0, s0, e0, n0] = footprint;
      const pad = Math.max(0.15, Math.min(e0 - w0, n0 - s0) * 0.0);
      const ring: THREE.Vector3[] = [];
      const N = 24;
      for (let i = 0; i <= N; i++) ring.push(ll(s0 - pad, w0 + (e0 - w0) * (i / N), 1.012));
      for (let i = 0; i <= N; i++) ring.push(ll(s0 + (n0 - s0) * (i / N), e0 + pad, 1.012));
      for (let i = 0; i <= N; i++) ring.push(ll(n0 + pad, e0 - (e0 - w0) * (i / N), 1.012));
      for (let i = 0; i <= N; i++) ring.push(ll(n0 - (n0 - s0) * (i / N), w0 - pad, 1.012));
      globe.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(ring), new THREE.LineBasicMaterial({ color: 0xec4899, linewidth: 2 })));
    }

    // orbits (altitude to scale; eccentric ones are drawn as a stretched ellipse)
    const sats: { pivot: THREE.Group; speed: number }[] = [];
    const R = BODY[body].radiusKm;
    orbits.forEach((o) => {
      const r = 1 + o.altitudeKm / R;
      const tilt = new THREE.Group();
      tilt.rotation.x = THREE.MathUtils.degToRad(o.inclinationDeg);
      scene.add(tilt);
      const col = new THREE.Color(o.color ?? "#06b6d4");
      const a = o.eccentric ? r * 1.6 : r, b = r;
      const curve = new THREE.EllipseCurve(o.eccentric ? r * 0.6 : 0, 0, a, b, 0, Math.PI * 2).getPoints(180).map((v) => new THREE.Vector3(v.x, 0, v.y));
      tilt.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve), new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.8 })));
      const pivot = new THREE.Group();
      tilt.add(pivot);
      const sat = new THREE.Mesh(new THREE.OctahedronGeometry(0.03), new THREE.MeshBasicMaterial({ color: col }));
      sat.position.set(a + (o.eccentric ? r * 0.6 : 0), 0, 0);
      pivot.add(sat);
      const t = tag(`${o.label} · ${Math.round(o.altitudeKm).toLocaleString()} km · i ${o.inclinationDeg}°`, "scene-label");
      t.position.set(0, 0.08, 0);
      sat.add(t);
      sats.push({ pivot, speed: (2 * Math.PI) / Math.max(o.periodMin, 1) * 20 });
    });

    const target = markers.find((m) => m.primary) ?? markers[0];
    const focus = () => {
      const dir = target ? ll(target.lat, target.lon, 1) : new THREE.Vector3(0, 0.3, 1);
      globe.rotation.set(0, 0, 0);
      camera.position.copy(dir.normalize().multiplyScalar(orbits.length ? 4.2 : 3.2));
      controls.target.set(0, 0, 0);
    };
    focusRef.current = focus;
    focus();

    let raf = 0, last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      if (spinRef.current) {
        globe.rotation.y += dt * 0.06;
        sats.forEach((s) => { s.pivot.rotation.y += dt * s.speed * 0.05; });
      }
      pulses.forEach((p) => { const k = 1 + 0.5 * ((now / 900) % 1); p.scale.setScalar(k); (p.material as THREE.MeshBasicMaterial).opacity = 1.4 - k; });
      controls.update();
      renderer.render(scene, camera);
      labels.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    const ro = new ResizeObserver(() => {
      w = el.clientWidth;
      renderer.setSize(w, h); labels.setSize(w, h);
      camera.aspect = w / h; camera.updateProjectionMatrix();
    });
    ro.observe(el);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose?.();
        (m.material as THREE.Material | undefined)?.dispose?.();
      });
      marsTex?.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      el.removeChild(labels.domElement);
    };
  }, [body, markers, orbits, footprint, height, theme]);

  return (
    <div className="scene-surface relative overflow-hidden rounded-2xl border">
      <div ref={host} className="relative w-full" style={{ height }} />
      <div className="absolute left-3 top-3 flex gap-1.5">
        <button className="scene-btn" onClick={() => setSpin((s) => !s)}>{spin ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}{spin ? "Pause" : "Spin"}</button>
        <button className="scene-btn" onClick={() => focusRef.current()}><Crosshair className="h-3.5 w-3.5" /> Locate</button>
      </div>
      <div className="pointer-events-none absolute right-3 top-3 rounded-full bg-white/80 px-3 py-1 font-mono text-[10px] font-semibold uppercase tracking-wider text-ink-soft">
        {BODY[body].name}
      </div>
      {caption && <div className="pointer-events-none absolute bottom-2 left-3 right-3 text-[10px] text-ink-faint">{caption}</div>}
    </div>
  );
}
