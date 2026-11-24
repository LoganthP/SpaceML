import { useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import * as THREE from "three";
import { CSS2DObject, CSS2DRenderer } from "three/examples/jsm/renderers/CSS2DRenderer.js";

export interface OrreryBody { id: string; label: string; sub: string; color: string; path: string; radius: number; size: number; speed: number; ready: boolean }

/** Bright, clickable orrery: the platform core with one orbiting world per module. */
export default function MissionOrrery({ bodies, height = 360 }: { bodies: OrreryBody[]; height?: number }) {
  const host = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const navRef = useRef(navigate);
  navRef.current = navigate;

  useEffect(() => {
    const el = host.current;
    if (!el) return;
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
    const camera = new THREE.PerspectiveCamera(38, w / h, 0.1, 200);
    camera.position.set(0, 7.5, 15);
    camera.lookAt(0, -0.4, 0);
    scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    const key = new THREE.PointLight(0xffffff, 80, 0, 1.5);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.8);
    fill.position.set(-5, 8, 6);
    scene.add(fill);

    // core: iridescent sphere + rings
    const coreMat = new THREE.MeshPhysicalMaterial({ color: 0xa78bfa, roughness: 0.25, metalness: 0.1, clearcoat: 1, iridescence: 1, iridescenceIOR: 1.6, emissive: 0x7c3aed, emissiveIntensity: 0.25 });
    const core = new THREE.Mesh(new THREE.IcosahedronGeometry(1.05, 6), coreMat);
    scene.add(core);
    const halo = new THREE.Mesh(new THREE.TorusGeometry(1.6, 0.02, 12, 120), new THREE.MeshBasicMaterial({ color: 0xc4b5fd }));
    halo.rotation.x = Math.PI / 2.3;
    scene.add(halo);
    const halo2 = halo.clone();
    halo2.rotation.set(Math.PI / 2, 0.6, 0);
    scene.add(halo2);

    // sparkles
    const pts: number[] = [];
    for (let i = 0; i < 380; i++) {
      const r = 5 + Math.random() * 10, a = Math.random() * Math.PI * 2;
      pts.push(Math.cos(a) * r, (Math.random() - 0.5) * 6, Math.sin(a) * r);
    }
    const sg = new THREE.BufferGeometry();
    sg.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    scene.add(new THREE.Points(sg, new THREE.PointsMaterial({ color: 0xa5b4fc, size: 0.06, transparent: true, opacity: 0.8 })));

    const meshes: { mesh: THREE.Mesh; pivot: THREE.Group; speed: number; body: OrreryBody; ring: THREE.Mesh }[] = [];
    bodies.forEach((b, i) => {
      const orbit = new THREE.Mesh(new THREE.TorusGeometry(b.radius, 0.012, 8, 200),
        new THREE.MeshBasicMaterial({ color: b.color, transparent: true, opacity: 0.45 }));
      orbit.rotation.x = Math.PI / 2;
      scene.add(orbit);
      const pivot = new THREE.Group();
      pivot.rotation.y = (i / bodies.length) * Math.PI * 2 + 0.6;
      scene.add(pivot);
      const mat = new THREE.MeshStandardMaterial({ color: b.color, roughness: 0.35, metalness: 0.1, emissive: b.color, emissiveIntensity: b.ready ? 0.25 : 0.05 });
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(b.size, 40, 40), mat);
      mesh.position.set(b.radius, 0, 0);
      mesh.userData.path = b.path;
      pivot.add(mesh);
      const ring = new THREE.Mesh(new THREE.RingGeometry(b.size * 1.35, b.size * 1.5, 48),
        new THREE.MeshBasicMaterial({ color: b.color, side: THREE.DoubleSide, transparent: true, opacity: b.ready ? 0.7 : 0.25 }));
      ring.rotation.x = Math.PI / 2.4;
      mesh.add(ring);
      const tagEl = document.createElement("div");
      tagEl.className = "orrery-tag";
      tagEl.innerHTML = `<b>${b.label}</b><span>${b.sub}</span>`;
      tagEl.style.setProperty("--c", b.color);
      const tag = new CSS2DObject(tagEl);
      tag.position.set(0, b.size + 0.45, 0);
      mesh.add(tag);
      meshes.push({ mesh, pivot, speed: b.speed, body: b, ring });
    });

    const ray = new THREE.Raycaster();
    const mouse = new THREE.Vector2(-9, -9);
    let hovered: THREE.Mesh | null = null;
    const onMove = (e: PointerEvent) => {
      const r = renderer.domElement.getBoundingClientRect();
      mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    };
    const onLeave = () => mouse.set(-9, -9);
    const onClick = () => { if (hovered) navRef.current(hovered.userData.path); };
    renderer.domElement.addEventListener("pointermove", onMove);
    renderer.domElement.addEventListener("pointerleave", onLeave);
    renderer.domElement.addEventListener("click", onClick);

    let raf = 0, t0 = performance.now();
    const tick = (now: number) => {
      const dt = Math.min((now - t0) / 1000, 0.05);
      t0 = now;
      core.rotation.y += dt * 0.25;
      halo.rotation.z += dt * 0.15;
      halo2.rotation.z -= dt * 0.1;
      ray.setFromCamera(mouse, camera);
      const hit = ray.intersectObjects(meshes.map((m) => m.mesh))[0]?.object as THREE.Mesh | undefined;
      hovered = hit ?? null;
      renderer.domElement.style.cursor = hovered ? "pointer" : "grab";
      for (const m of meshes) {
        const isHover = m.mesh === hovered;
        if (!isHover) m.pivot.rotation.y += dt * m.speed;
        const s = THREE.MathUtils.lerp(m.mesh.scale.x, isHover ? 1.35 : 1, 0.15);
        m.mesh.scale.setScalar(s);
        m.mesh.rotation.y += dt * 0.6;
      }
      renderer.render(scene, camera);
      labels.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    const ro = new ResizeObserver(() => {
      w = el.clientWidth;
      renderer.setSize(w, h);
      labels.setSize(w, h);
      camera.aspect = w / h;
      camera.position.set(0, 7.5, w < 560 ? 21 : 15);
      camera.lookAt(0, -0.4, 0);
      camera.updateProjectionMatrix();
    });
    ro.observe(el);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.domElement.removeEventListener("pointermove", onMove);
      renderer.domElement.removeEventListener("pointerleave", onLeave);
      renderer.domElement.removeEventListener("click", onClick);
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose?.();
        (m.material as THREE.Material | undefined)?.dispose?.();
      });
      renderer.dispose();
      el.removeChild(renderer.domElement);
      el.removeChild(labels.domElement);
    };
  }, [bodies, height]);

  return <div ref={host} className="relative w-full" style={{ height }} />;
}
