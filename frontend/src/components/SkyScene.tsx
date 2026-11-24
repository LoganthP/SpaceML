import { Crosshair, Globe2 } from "lucide-react";
import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/examples/jsm/renderers/CSS2DRenderer.js";
import { palette, useTheme } from "../lib/theme";

export type SkyStar = [number, number, number, number, number]; // kepid, ra, dec, label(2=PC,1=AFP,0=other), cached

const D2R = Math.PI / 180;
/** RA/Dec (deg) -> point on a sphere; looking out from the centre, as on the real sky. */
export function radec(ra: number, dec: number, r = 10): THREE.Vector3 {
  return new THREE.Vector3(
    -r * Math.cos(dec * D2R) * Math.cos(ra * D2R),
    r * Math.sin(dec * D2R),
    r * Math.cos(dec * D2R) * Math.sin(ra * D2R),
  );
}

const KEPLER_FIELD = { ra: 290.67, dec: 44.5 };

function tag(text: string, cls: string) {
  const el = document.createElement("div");
  el.className = cls;
  el.textContent = text;
  return new CSS2DObject(el);
}

/**
 * Celestial sphere seen from Earth (camera at the centre). Every Kepler target in the catalog is
 * placed at its real right ascension / declination; the analysed star is ringed. RA/Dec grid
 * lines every 30° / 15° give orientation; the Kepler field sits between Cygnus and Lyra.
 */
export default function SkyScene({ stars, selected, height = 440 }: {
  stars: SkyStar[]; selected: { kepid?: number; ra: number; dec: number; name: string } | null; height?: number;
}) {
  const theme = useTheme();
  const host = useRef<HTMLDivElement>(null);
  const focus = useRef<() => void>(() => {});
  const overview = useRef<() => void>(() => {});

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
    Object.assign(labels.domElement.style, { position: "absolute", inset: "0", pointerEvents: "none" });
    el.appendChild(labels.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 100);
    camera.position.set(0, 0, 0.01);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableZoom = false;
    controls.enablePan = false;
    controls.rotateSpeed = -0.35; // drag the sky, not the camera
    controls.enableDamping = true;

    // RA/Dec grid
    const pal = palette(theme);
    const gridMat = new THREE.LineBasicMaterial({ color: pal.sceneGrid, transparent: true, opacity: 0.7 });
    for (let ra = 0; ra < 360; ra += 30) {
      const pts = [];
      for (let d = -80; d <= 80; d += 4) pts.push(radec(ra, d, 10.2));
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), gridMat));
      const t = tag(`${ra / 15}h`, "scene-label scene-label-grid");
      t.position.copy(radec(ra, 0, 10.2));
      scene.add(t);
    }
    for (let dec = -75; dec <= 75; dec += 15) {
      const pts = [];
      for (let ra = 0; ra <= 360; ra += 3) pts.push(radec(ra, dec, 10.2));
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
        dec === 0 ? new THREE.LineBasicMaterial({ color: 0x818cf8 }) : gridMat));
      if (dec !== 0) {
        const t = tag(`${dec > 0 ? "+" : ""}${dec}°`, "scene-label scene-label-grid");
        t.position.copy(radec(KEPLER_FIELD.ra + 25, dec, 10.2));
        scene.add(t);
      }
    }

    // stars
    const groups: Record<number, number[]> = { 0: [], 1: [], 2: [] };
    for (const [, ra, dec, lab] of stars) {
      const v = radec(ra, dec);
      groups[lab]?.push(v.x, v.y, v.z);
    }
    const colors = { 0: pal.dark ? 0xe2e8f0 : 0x94a3b8, 1: 0xf43f5e, 2: 0x10b981 };
    const sizes = { 0: 0.035, 1: 0.045, 2: 0.055 };
    for (const k of [0, 1, 2] as const) {
      if (!groups[k].length) continue;
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(groups[k], 3));
      scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: colors[k], size: sizes[k], transparent: true, opacity: 0.85 })));
    }
    const fieldTag = tag("Kepler field · Cygnus–Lyra", "scene-label");
    fieldTag.position.copy(radec(KEPLER_FIELD.ra, KEPLER_FIELD.dec + 9, 9.8));
    scene.add(fieldTag);

    // selected star
    let ring: THREE.Mesh | null = null;
    if (selected) {
      const p = radec(selected.ra, selected.dec, 9.9);
      ring = new THREE.Mesh(new THREE.RingGeometry(0.16, 0.22, 48),
        new THREE.MeshBasicMaterial({ color: 0x7c3aed, side: THREE.DoubleSide, transparent: true }));
      ring.position.copy(p);
      ring.lookAt(0, 0, 0);
      scene.add(ring);
      const dot = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 16), new THREE.MeshBasicMaterial({ color: 0x7c3aed }));
      dot.position.copy(p);
      scene.add(dot);
      const t = tag(`${selected.name} · RA ${selected.ra.toFixed(2)}° Dec ${selected.dec >= 0 ? "+" : ""}${selected.dec.toFixed(2)}°`,
        "scene-label scene-label-sel");
      t.position.copy(radec(selected.ra, selected.dec + 0.9, 9.9));
      scene.add(t);
    }

    const look = (ra: number, dec: number, fov: number) => {
      const target = radec(ra, dec, 1);
      camera.position.copy(target.clone().multiplyScalar(-0.01));
      controls.target.set(0, 0, 0);
      camera.fov = fov;
      camera.updateProjectionMatrix();
    };
    focus.current = () => selected ? look(selected.ra, selected.dec, 22) : look(KEPLER_FIELD.ra, KEPLER_FIELD.dec, 40);
    overview.current = () => look(KEPLER_FIELD.ra, KEPLER_FIELD.dec, 62);
    focus.current();

    let raf = 0;
    const tick = (t: number) => {
      if (ring) (ring.material as THREE.MeshBasicMaterial).opacity = 0.55 + 0.45 * Math.sin(t / 300);
      controls.update();
      renderer.render(scene, camera);
      labels.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      camera.fov = Math.min(Math.max(camera.fov + e.deltaY * 0.03, 6), 90);
      camera.updateProjectionMatrix();
    };
    renderer.domElement.addEventListener("wheel", onWheel, { passive: false });
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
      renderer.domElement.removeEventListener("wheel", onWheel);
      controls.dispose();
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose?.();
        const mat = m.material as THREE.Material | undefined;
        mat?.dispose?.();
      });
      renderer.dispose();
      el.removeChild(renderer.domElement);
      el.removeChild(labels.domElement);
    };
  }, [stars, selected, height, theme]);

  return (
    <div className="scene-surface relative overflow-hidden rounded-2xl border">
      <div ref={host} className="relative w-full" style={{ height }} />
      <div className="absolute left-3 top-3 flex gap-1.5">
        <button className="scene-btn" onClick={() => focus.current()}><Crosshair className="h-3.5 w-3.5" /> Find star</button>
        <button className="scene-btn" onClick={() => overview.current()}><Globe2 className="h-3.5 w-3.5" /> Whole field</button>
      </div>
      <div className="absolute right-3 top-3 flex gap-2 rounded-full bg-white/80 px-3 py-1 text-[10px] font-semibold text-ink-soft">
        <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-emerald-500" /> planet candidate</span>
        <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-rose-500" /> false positive</span>
        <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-slate-400" /> other</span>
      </div>
      <div className="pointer-events-none absolute bottom-2 left-3 text-[10px] text-ink-faint">
        View from Earth · drag to look around · scroll to zoom · {stars.length.toLocaleString()} Kepler targets at their catalog RA/Dec
      </div>
    </div>
  );
}
