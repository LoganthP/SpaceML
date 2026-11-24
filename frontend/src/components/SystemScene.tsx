import { Eye, Pause, Play, RotateCcw, Ruler, View } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/examples/jsm/renderers/CSS2DRenderer.js";
import { palette, useTheme } from "../lib/theme";
import type { SystemInfo } from "../lib/types";

/** Black-body-ish colour for a stellar effective temperature (K). */
export function starColor(teff: number): THREE.Color {
  const t = Math.min(Math.max(teff, 2500), 12000) / 100;
  let r: number, g: number, b: number;
  if (t <= 66) {
    r = 255;
    g = 99.47 * Math.log(t) - 161.12;
    b = t <= 19 ? 0 : 138.52 * Math.log(t - 10) - 305.04;
  } else {
    r = 329.7 * Math.pow(t - 60, -0.1332);
    g = 288.12 * Math.pow(t - 60, -0.0755);
    b = 255;
  }
  const c = (v: number) => Math.min(Math.max(v, 0), 255) / 255;
  return new THREE.Color(c(r), c(g), c(b));
}

const PLANET_COLORS = ["#7c3aed", "#0891b2", "#ea580c", "#db2777", "#16a34a", "#ca8a04", "#2563eb", "#9333ea"];

type Mode = "visual" | "true";

function label(text: string, cls: string) {
  const el = document.createElement("div");
  el.className = cls;
  el.textContent = text;
  return new CSS2DObject(el);
}

/**
 * Orbital view of a star and its planets, built from the derived system parameters.
 * The observer (Earth / Kepler) sits on the +X axis; each orbit is tilted by its inclination
 * so an edge-on orbit (i ≈ 90°) carries the planet across the stellar disc as seen from Earth.
 */
export default function SystemScene({ system, height = 440 }: { system: SystemInfo; height?: number }) {
  const theme = useTheme();
  const host = useRef<HTMLDivElement>(null);
  const api = useRef<{ setView: (v: "earth" | "top" | "free") => void; reset: () => void } | null>(null);
  const [playing, setPlaying] = useState(true);
  const [mode, setMode] = useState<Mode>("visual");
  const [transiting, setTransiting] = useState(false);
  const playRef = useRef(playing);
  playRef.current = playing;

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const w = el.clientWidth, h = height;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
    el.appendChild(renderer.domElement);
    const labels = new CSS2DRenderer();
    labels.setSize(w, h);
    labels.domElement.style.position = "absolute";
    labels.domElement.style.inset = "0";
    labels.domElement.style.pointerEvents = "none";
    el.appendChild(labels.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 2000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const sun = new THREE.PointLight(0xffffff, 60, 0, 1.6);
    scene.add(sun);

    const { star, planets } = system;
    // scale to the planets; the habitable zone is drawn only if it fits near the orbits
    const aMax = Math.max(...planets.map((p) => p.a_au), 0.005) * 1.15;
    // Visual mode compresses distances (sqrt) and inflates bodies so every planet is visible;
    // true mode uses one scale for everything (star and planets become tiny, as in reality).
    const R = 6; // scene radius for the outermost orbit
    const dist = (au: number) => (mode === "visual" ? R * Math.sqrt(au / aMax) : (R * au) / aMax);
    const starR = mode === "visual" ? Math.min(0.55, 0.22 + 0.08 * Math.sqrt(star.radius_rsun))
      : Math.max((R * star.radius_rsun * 0.00465) / aMax, 0.004);
    const planetR = (rpEarth: number) => mode === "visual" ? Math.min(0.32, 0.05 + 0.035 * Math.sqrt(rpEarth))
      : Math.max((R * rpEarth * 4.26e-5) / aMax, 0.0015);

    // star + glow
    const sc = starColor(star.teff);
    const starMesh = new THREE.Mesh(new THREE.SphereGeometry(starR, 48, 48), new THREE.MeshBasicMaterial({ color: sc }));
    scene.add(starMesh);
    const glowTex = (() => {
      const c = document.createElement("canvas");
      c.width = c.height = 128;
      const g = c.getContext("2d")!;
      const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
      grd.addColorStop(0, `rgba(${sc.r * 255},${sc.g * 255},${sc.b * 255},0.9)`);
      grd.addColorStop(0.35, `rgba(${sc.r * 255},${sc.g * 255},${sc.b * 255},0.25)`);
      grd.addColorStop(1, "rgba(255,255,255,0)");
      g.fillStyle = grd;
      g.fillRect(0, 0, 128, 128);
      return new THREE.CanvasTexture(c);
    })();
    const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, depthWrite: false }));
    glow.scale.setScalar(Math.max(starR * 7, 1.2));
    scene.add(glow);
    const starLabel = label(`${Math.round(star.teff)} K · ${star.radius_rsun.toFixed(2)} R☉`, "scene-label scene-label-star");
    starLabel.position.set(0, -starR - 0.25, 0);
    scene.add(starLabel);

    // habitable zone (in the reference plane)
    const hzIn = dist(star.hz_inner_au), hzOut = dist(star.hz_outer_au);
    if (hzIn < R * 1.6) {
      const hz = new THREE.Mesh(new THREE.RingGeometry(hzIn, hzOut, 128),
        new THREE.MeshBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false }));
      hz.rotation.x = -Math.PI / 2;
      scene.add(hz);
      const hzl = label("habitable zone", "scene-label scene-label-hz");
      hzl.position.set(0, 0, -Math.min((hzIn + hzOut) / 2, R * 1.5));
      scene.add(hzl);
    } else {
      const hzl = label(`habitable zone starts at ${star.hz_inner_au.toFixed(2)} AU (off-scale, outside view)`, "scene-label scene-label-hz");
      hzl.position.set(0, 0, -R * 1.2);
      scene.add(hzl);
    }

    // reference grid (faint)
    const pal = palette(theme);
    const grid = new THREE.PolarGridHelper(R * 1.15, 12, 6, 96, pal.sceneGrid, pal.sceneGrid2);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.55;
    scene.add(grid);

    // observer: line of sight to Earth along +X
    const losLen = R * 1.4;
    const los = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(losLen, 0, 0)]),
      new THREE.LineDashedMaterial({ color: pal.dark ? 0x94a3b8 : 0x64748b, dashSize: 0.18, gapSize: 0.12 }));
    los.computeLineDistances();
    scene.add(los);
    const earth = new THREE.Mesh(new THREE.SphereGeometry(0.12, 24, 24), new THREE.MeshStandardMaterial({ color: 0x3b82f6, roughness: 0.5 }));
    earth.position.set(losLen, 0, 0);
    scene.add(earth);
    const earthLabel = label("to Earth / Kepler", "scene-label");
    earthLabel.position.set(losLen, 0.32, 0);
    scene.add(earthLabel);

    // planets
    const bodies: { mesh: THREE.Mesh; pivot: THREE.Group; r: number; period: number; phase: number; sel: boolean }[] = [];
    const minP = Math.min(...planets.map((p) => p.period_days));
    planets.forEach((p, i) => {
      const r = dist(p.a_au);
      const color = p.selected ? "#7c3aed" : PLANET_COLORS[(i + 1) % PLANET_COLORS.length];
      const tilt = new THREE.Group();
      tilt.rotation.z = THREE.MathUtils.degToRad(90 - (p.inclination_deg || 90)); // edge-on when i = 90°
      scene.add(tilt);
      const orbit = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(new THREE.EllipseCurve(0, 0, r, r, 0, Math.PI * 2).getPoints(160)
          .map((v) => new THREE.Vector3(v.x, 0, v.y))),
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: p.selected ? 0.95 : 0.45 }));
      tilt.add(orbit);
      const pivot = new THREE.Group();
      tilt.add(pivot);
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(planetR(p.rp_earth), 32, 32),
        new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.05, emissive: color, emissiveIntensity: 0.12 }));
      mesh.position.set(r, 0, 0);
      pivot.add(mesh);
      const lab = label(`${p.label ?? p.id} · ${p.period_days.toFixed(p.period_days < 10 ? 2 : 1)} d · ${p.rp_earth.toFixed(1)} R⊕`,
        `scene-label ${p.selected ? "scene-label-sel" : ""}`);
      lab.position.set(0, planetR(p.rp_earth) + 0.18, 0);
      mesh.add(lab);
      // stagger phases; the selected planet starts just before transit
      bodies.push({ mesh, pivot, r, period: p.period_days, phase: p.selected ? -0.5 : i * 1.7, sel: p.selected });
    });

    const views = {
      earth: () => { camera.position.set(R * 2.4, 0.15, 0.001); controls.target.set(0, 0, 0); },
      top: () => { camera.position.set(0.001, R * 2.6, 0.001); controls.target.set(0, 0, 0); },
      free: () => { camera.position.set(R * 1.3, R * 0.95, R * 1.55); controls.target.set(0, 0, 0); },
    };
    views.free();
    api.current = { setView: (v) => views[v](), reset: () => views.free() };

    let raf = 0, last = performance.now(), wasTransit = false;
    const world = new THREE.Vector3();
    const tick = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      if (playRef.current) {
        for (const b of bodies) {
          b.phase += (dt * 2 * Math.PI * 0.12 * minP) / b.period; // innermost planet ~8 s per orbit
          b.pivot.rotation.y = b.phase;
        }
      }
      const s = bodies.find((b) => b.sel);
      if (s) {
        s.mesh.getWorldPosition(world);
        const inFront = world.x > 0 && Math.hypot(world.y, world.z) < starR + planetR(1) * 1.2;
        if (inFront !== wasTransit) { wasTransit = inFront; setTransiting(inFront); }
      }
      controls.update();
      renderer.render(scene, camera);
      labels.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    const ro = new ResizeObserver(() => {
      const nw = el.clientWidth;
      renderer.setSize(nw, h);
      labels.setSize(nw, h);
      camera.aspect = nw / h;
      camera.updateProjectionMatrix();
    });
    ro.observe(el);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose?.();
        const mat = m.material as THREE.Material | THREE.Material[] | undefined;
        (Array.isArray(mat) ? mat : mat ? [mat] : []).forEach((x) => x.dispose());
      });
      glowTex.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      el.removeChild(labels.domElement);
    };
  }, [system, mode, height, theme]);

  return (
    <div className="scene-surface relative overflow-hidden rounded-2xl border">
      <div ref={host} className="relative w-full" style={{ height }} />
      <div className="absolute left-3 top-3 flex flex-wrap gap-1.5">
        <button className="scene-btn" onClick={() => setPlaying((p) => !p)}>{playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}{playing ? "Pause" : "Play"}</button>
        <button className="scene-btn" onClick={() => api.current?.setView("earth")}><Eye className="h-3.5 w-3.5" /> View from Earth</button>
        <button className="scene-btn" onClick={() => api.current?.setView("top")}><View className="h-3.5 w-3.5" /> Top</button>
        <button className="scene-btn" onClick={() => api.current?.reset()}><RotateCcw className="h-3.5 w-3.5" /></button>
        <button className="scene-btn" onClick={() => setMode((m) => (m === "visual" ? "true" : "visual"))}>
          <Ruler className="h-3.5 w-3.5" /> {mode === "visual" ? "Visual scale" : "True scale"}
        </button>
      </div>
      <div className={`absolute right-3 top-3 rounded-full px-3 py-1 text-[11px] font-bold transition ${transiting ? "bg-violet-600 text-white shadow-lg" : "bg-white/80 text-ink-soft"}`}>
        {transiting ? "● Transit: flux dips now" : "Out of transit"}
      </div>
      <div className="pointer-events-none absolute bottom-2 left-3 text-[10px] text-ink-faint">
        Drag to rotate · scroll to zoom · {mode === "visual" ? "distances √-compressed, bodies enlarged" : "distances and sizes on one scale"}
      </div>
    </div>
  );
}
