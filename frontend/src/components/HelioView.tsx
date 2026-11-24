import { Pause, Play } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/examples/jsm/renderers/CSS2DRenderer.js";
import { palette, useTheme } from "../lib/theme";

/** Mean heliocentric ecliptic longitude (deg) for a date - good to ~1-2° for Earth and Mars. */
function meanLongitude(planet: "earth" | "mars", date: Date): number {
  const d = (date.getTime() - Date.UTC(2000, 0, 1, 12)) / 86400000;
  const L = planet === "earth" ? 100.46 + 0.9856474 * d : 355.45 + 0.5240329 * d;
  return ((L % 360) + 360) % 360;
}
/** RA/Dec (deg) -> unit vector in the ecliptic frame (x: vernal equinox, z: ecliptic north -> scene y). */
function eclipticDir(ra: number, dec: number): THREE.Vector3 {
  const e = THREE.MathUtils.degToRad(23.439), a = THREE.MathUtils.degToRad(ra), d = THREE.MathUtils.degToRad(dec);
  const x = Math.cos(d) * Math.cos(a), y = Math.cos(d) * Math.sin(a), z = Math.sin(d);
  const ye = y * Math.cos(e) + z * Math.sin(e), ze = -y * Math.sin(e) + z * Math.cos(e);
  return new THREE.Vector3(x, ze, -ye).normalize();
}
const AU = 3.2; // scene units per AU
const pos = (lonDeg: number, rAU: number) => {
  const l = THREE.MathUtils.degToRad(lonDeg);
  return new THREE.Vector3(Math.cos(l) * rAU * AU, 0, -Math.sin(l) * rAU * AU);
};
function tag(text: string, cls = "scene-label") {
  const el = document.createElement("div");
  el.className = cls;
  el.textContent = text;
  return new CSS2DObject(el);
}

/**
 * Sun-centred view of where an observation was made: Earth and Mars at their positions on
 * `date`, Kepler trailing Earth with its fixed field of view (Cygnus-Lyra), the Sun-Earth L1
 * point where ACE/Wind/DSCOVR sample the solar wind, and the solar-wind stream.
 */
export default function HelioView({ date, show, target, height = 380, caption }: {
  date?: string | null;
  show: { mars?: boolean; kepler?: boolean; l1?: boolean; wind?: boolean; tess?: boolean };
  target?: { ra: number; dec: number; label: string } | null; height?: number; caption?: string;
}) {
  const theme = useTheme();
  const host = useRef<HTMLDivElement>(null);
  const [run, setRun] = useState(true);
  const runRef = useRef(run);
  runRef.current = run;

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const pal = palette(theme);
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
    const camera = new THREE.PerspectiveCamera(40, w / h, 0.05, 200);
    camera.position.set(2.5, 6.5, 8.5);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    scene.add(new THREE.PointLight(0xffffff, 40, 0, 1.2));

    const when = date ? new Date(date) : new Date();
    // Sun
    const sun = new THREE.Mesh(new THREE.SphereGeometry(0.42, 48, 48), new THREE.MeshBasicMaterial({ color: 0xfbbf24 }));
    scene.add(sun);
    const glowC = document.createElement("canvas");
    glowC.width = glowC.height = 128;
    const gg = glowC.getContext("2d")!;
    const grd = gg.createRadialGradient(64, 64, 0, 64, 64, 64);
    grd.addColorStop(0, "rgba(253,224,71,0.9)"); grd.addColorStop(0.3, "rgba(251,146,60,0.35)"); grd.addColorStop(1, "rgba(251,146,60,0)");
    gg.fillStyle = grd; gg.fillRect(0, 0, 128, 128);
    const glowTex = new THREE.CanvasTexture(glowC);
    const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, depthWrite: false }));
    glow.scale.setScalar(2.6);
    scene.add(glow);
    const sunTag = tag("Sun"); sunTag.position.set(0, -0.62, 0); scene.add(sunTag);

    const ring = (rAU: number, color: number, op = 0.6) => {
      const pts = new THREE.EllipseCurve(0, 0, rAU * AU, rAU * AU, 0, Math.PI * 2).getPoints(200).map((v) => new THREE.Vector3(v.x, 0, v.y));
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color, transparent: true, opacity: op })));
    };
    const grid = new THREE.PolarGridHelper(AU * 1.8, 12, 4, 96, pal.sceneGrid, pal.sceneGrid2);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.45;
    scene.add(grid);

    // Earth
    ring(1, 0x3b82f6);
    const eLon = meanLongitude("earth", when);
    const earthPos = pos(eLon, 1);
    const earth = new THREE.Mesh(new THREE.SphereGeometry(0.14, 32, 32), new THREE.MeshStandardMaterial({ color: 0x3b82f6, roughness: 0.5, emissive: 0x1d4ed8, emissiveIntensity: 0.3 }));
    earth.position.copy(earthPos);
    scene.add(earth);
    const eTag = tag(`Earth · ${when.toISOString().slice(0, 10)}`); eTag.position.copy(earthPos).add(new THREE.Vector3(0, 0.32, 0)); scene.add(eTag);

    const movers: (() => void)[] = [];
    if (show.mars) {
      ring(1.524, 0xf97316);
      const mp = pos(meanLongitude("mars", when), 1.524);
      const mars = new THREE.Mesh(new THREE.SphereGeometry(0.1, 32, 32), new THREE.MeshStandardMaterial({ color: 0xea580c, roughness: 0.8 }));
      mars.position.copy(mp);
      scene.add(mars);
      const t = tag("Mars · Curiosity (Gale crater)", "scene-label scene-label-sel"); t.position.copy(mp).add(new THREE.Vector3(0, 0.3, 0)); scene.add(t);
      const link = new THREE.Line(new THREE.BufferGeometry().setFromPoints([earthPos, mp]),
        new THREE.LineDashedMaterial({ color: 0xf97316, dashSize: 0.15, gapSize: 0.1 }));
      link.computeLineDistances();
      scene.add(link);
      const dist = earthPos.distanceTo(mp) / AU;
      const lt = tag(`${dist.toFixed(2)} AU · light time ${(dist * 8.317).toFixed(1)} min`);
      lt.position.copy(earthPos.clone().lerp(mp, 0.5)).add(new THREE.Vector3(0, 0.18, 0));
      scene.add(lt);
    }
    if (show.kepler) {
      // Kepler: Earth-trailing heliocentric orbit (372.5 d), drifting ~0.13 AU/yr behind Earth after 2009 launch.
      const yrs = Math.max(0.5, Math.min((when.getTime() - Date.UTC(2009, 2, 7)) / 3.156e10, 9.6));
      const drift = THREE.MathUtils.radToDeg((0.13 * Math.min(yrs, 4)) / 1.0);
      const kp = pos(eLon - drift, 1.01);
      const kep = new THREE.Mesh(new THREE.OctahedronGeometry(0.09), new THREE.MeshStandardMaterial({ color: 0x8b5cf6, emissive: 0x7c3aed, emissiveIntensity: 0.4 }));
      kep.position.copy(kp);
      scene.add(kep);
      const t = tag("Kepler space telescope", "scene-label scene-label-sel"); t.position.copy(kp).add(new THREE.Vector3(0, 0.3, 0)); scene.add(t);
      const dir = eclipticDir(target?.ra ?? 290.67, target?.dec ?? 44.5);
      const len = 3.4;
      const cone = new THREE.Mesh(new THREE.ConeGeometry(0.55, len, 32, 1, true),
        new THREE.MeshBasicMaterial({ color: 0xa78bfa, transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false }));
      cone.position.copy(kp).add(dir.clone().multiplyScalar(len / 2));
      cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, -1, 0), dir);
      scene.add(cone);
      const ft = tag(target ? `field of view → ${target.label}` : "fixed field: Cygnus–Lyra (115 deg²)");
      ft.position.copy(kp).add(dir.clone().multiplyScalar(len + 0.2));
      scene.add(ft);
    }
    if (show.tess) {
      const tp = earthPos.clone().add(new THREE.Vector3(0.35, 0.12, 0));
      const tess = new THREE.Mesh(new THREE.OctahedronGeometry(0.06), new THREE.MeshBasicMaterial({ color: 0x8b5cf6 }));
      tess.position.copy(tp); scene.add(tess);
      const t = tag("TESS · 13.7-day lunar-resonant Earth orbit", "scene-label scene-label-sel"); t.position.copy(tp).add(new THREE.Vector3(0.2, 0.55, 0)); scene.add(t);
    }
    if (show.l1) {
      // L1 is 0.01 AU sunward of Earth; drawn 6x farther out so it is visible.
      const l1 = earthPos.clone().multiplyScalar(1 - 0.06);
      const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.06), new THREE.MeshBasicMaterial({ color: 0x06b6d4 }));
      m.position.copy(l1); scene.add(m);
      const t = tag("L1 monitors · ACE / Wind / DSCOVR", "scene-label scene-label-sel"); t.position.copy(l1).add(new THREE.Vector3(0, 0.42, 0)); scene.add(t);
      // magnetosphere bow
      const bow = new THREE.Mesh(new THREE.SphereGeometry(0.32, 32, 16, 0, Math.PI), new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.14, side: THREE.DoubleSide, depthWrite: false }));
      bow.position.copy(earthPos);
      bow.lookAt(new THREE.Vector3(0, 0, 0));
      bow.rotateY(Math.PI / 2);
      scene.add(bow);
    }
    if (show.wind) {
      const n = 380;
      const p = new Float32Array(n * 3);
      const seeds = Array.from({ length: n }, () => ({ t: Math.random(), a: (Math.random() - 0.5) * 0.5, y: (Math.random() - 0.5) * 0.35 }));
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(p, 3));
      scene.add(new THREE.Points(geo, new THREE.PointsMaterial({ color: 0xfbbf24, size: 0.035, transparent: true, opacity: 0.8 })));
      movers.push(() => {
        seeds.forEach((s, i) => {
          if (runRef.current) s.t = (s.t + 0.004) % 1;
          const r = 0.45 + s.t * AU * 1.05;
          const ang = THREE.MathUtils.degToRad(eLon) + s.a * (1 - s.t * 0.6);
          p.set([Math.cos(ang) * r, s.y * s.t, -Math.sin(ang) * r], i * 3);
        });
        geo.attributes.position.needsUpdate = true;
      });
    }
    if (show.l1 && !show.mars && !show.kepler) {
      // frame the Sun-Earth line: Earth in front, Sun behind, L1 between
      const side = new THREE.Vector3(-earthPos.z, 0, earthPos.x).normalize();
      controls.target.copy(earthPos.clone().multiplyScalar(0.62));
      camera.position.copy(earthPos.clone().multiplyScalar(1.25)).add(side.multiplyScalar(2.8)).add(new THREE.Vector3(0, 1.9, 0));
    } else {
      controls.target.set(earthPos.x * 0.45, 0, earthPos.z * 0.45);
    }

    let raf = 0;
    const tick = () => {
      movers.forEach((m) => m());
      if (runRef.current) earth.rotation.y += 0.02;
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
      glowTex.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      el.removeChild(labels.domElement);
    };
  }, [date, show.mars, show.kepler, show.l1, show.wind, show.tess, target?.ra, target?.dec, target?.label, height, theme]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="scene-surface relative overflow-hidden rounded-2xl border">
      <div ref={host} className="relative w-full" style={{ height }} />
      <div className="absolute left-3 top-3"><button className="scene-btn" onClick={() => setRun((r) => !r)}>{run ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}{run ? "Pause" : "Play"}</button></div>
      <div className="pointer-events-none absolute right-3 top-3 rounded-full bg-white/80 px-3 py-1 font-mono text-[10px] font-semibold uppercase tracking-wider text-ink-soft">Heliocentric · ecliptic</div>
      {caption && <div className="pointer-events-none absolute bottom-2 left-3 right-3 text-[10px] text-ink-faint">{caption}</div>}
    </div>
  );
}
