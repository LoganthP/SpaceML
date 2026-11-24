import { useEffect, useRef } from "react";
import { useTheme } from "../lib/theme";

interface StarP { x: number; y: number; r: number; a: number; tw: number; d: number; hue: number }

/**
 * Full-screen starfield behind the glass UI. Light theme: faint indigo specks over pastel
 * nebulae. Dark theme: bright twinkling stars, coloured nebulae and the occasional meteor.
 * Stars drift slightly with the pointer (depth parallax). Static if reduced motion is preferred.
 */
export default function SpaceBackdrop() {
  const theme = useTheme();
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current!;
    const ctx = cv.getContext("2d")!;
    const dark = theme === "dark";
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let w = 0, h = 0, stars: StarP[] = [], raf = 0, last = 0;
    const mouse = { x: 0, y: 0, tx: 0, ty: 0 };
    let meteor: { x: number; y: number; vx: number; vy: number; life: number } | null = null;

    const build = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = window.innerWidth; h = window.innerHeight;
      cv.width = w * dpr; cv.height = h * dpr;
      cv.style.width = `${w}px`; cv.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const n = Math.round((w * h) / (dark ? 2600 : 4200));
      stars = Array.from({ length: n }, () => ({
        x: Math.random() * w, y: Math.random() * h, d: Math.random(),
        r: Math.random() < 0.06 ? 1.3 + Math.random() * 1.1 : 0.35 + Math.random() * 0.8,
        a: 0.25 + Math.random() * 0.75, tw: Math.random() * Math.PI * 2, hue: Math.random(),
      }));
    };

    const draw = (t: number) => {
      ctx.clearRect(0, 0, w, h);
      mouse.x += (mouse.tx - mouse.x) * 0.04;
      mouse.y += (mouse.ty - mouse.y) * 0.04;
      for (const s of stars) {
        const tw = reduce ? 1 : 0.65 + 0.35 * Math.sin(t / 900 + s.tw * 7);
        const px = s.x + mouse.x * (s.d * 14), py = s.y + mouse.y * (s.d * 10);
        const alpha = s.a * tw * (dark ? 1 : 0.55);
        ctx.beginPath();
        ctx.arc(px, py, s.r, 0, Math.PI * 2);
        ctx.fillStyle = dark
          ? (s.hue < 0.15 ? `rgba(255,214,170,${alpha})` : s.hue < 0.3 ? `rgba(170,200,255,${alpha})` : `rgba(240,244,255,${alpha})`)
          : `rgba(99,102,241,${alpha * 0.55})`;
        ctx.fill();
        if (dark && s.r > 1.3) {
          ctx.beginPath();
          ctx.arc(px, py, s.r * 3.2, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(190,205,255,${alpha * 0.08})`;
          ctx.fill();
        }
      }
      if (dark && !reduce) {
        if (!meteor && Math.random() < 0.002) {
          meteor = { x: Math.random() * w * 0.8, y: Math.random() * h * 0.3, vx: 9 + Math.random() * 5, vy: 3 + Math.random() * 2, life: 1 };
        }
        if (meteor) {
          const m = meteor;
          const g = ctx.createLinearGradient(m.x, m.y, m.x - m.vx * 12, m.y - m.vy * 12);
          g.addColorStop(0, `rgba(255,255,255,${0.85 * m.life})`);
          g.addColorStop(1, "rgba(255,255,255,0)");
          ctx.strokeStyle = g;
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.moveTo(m.x, m.y);
          ctx.lineTo(m.x - m.vx * 12, m.y - m.vy * 12);
          ctx.stroke();
          m.x += m.vx; m.y += m.vy; m.life -= 0.012;
          if (m.life <= 0 || m.x > w + 50) meteor = null;
        }
      }
    };

    const loop = (t: number) => {
      raf = requestAnimationFrame(loop);
      if (t - last < 33) return; // ~30 fps is plenty for a backdrop
      last = t;
      if (!document.hidden) draw(t);
    };
    const onMove = (e: PointerEvent) => { mouse.tx = e.clientX / w - 0.5; mouse.ty = e.clientY / h - 0.5; };
    build();
    if (reduce) draw(0); else raf = requestAnimationFrame(loop);
    window.addEventListener("resize", build);
    window.addEventListener("pointermove", onMove);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", build);
      window.removeEventListener("pointermove", onMove);
    };
  }, [theme]);

  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      {theme === "dark" ? (
        <>
          <div className="absolute -left-40 top-[-10rem] h-[38rem] w-[38rem] rounded-full bg-violet-700/25 blur-[120px]" />
          <div className="absolute right-[-12rem] top-[20%] h-[34rem] w-[34rem] rounded-full bg-cyan-600/15 blur-[120px]" />
          <div className="absolute bottom-[-14rem] left-[30%] h-[36rem] w-[36rem] rounded-full bg-fuchsia-700/15 blur-[130px]" />
        </>
      ) : (
        <>
          <div className="absolute -left-32 -top-32 h-[28rem] w-[28rem] animate-drift rounded-full bg-cyan-300/40 blur-3xl" />
          <div className="absolute right-[-8rem] top-10 h-[32rem] w-[32rem] animate-drift rounded-full bg-violet-300/40 blur-3xl [animation-delay:-6s]" />
          <div className="absolute bottom-[-10rem] left-1/3 h-[30rem] w-[30rem] animate-drift rounded-full bg-orange-200/50 blur-3xl [animation-delay:-11s]" />
        </>
      )}
      <canvas ref={ref} className="absolute inset-0" />
      {/* faint orbital rings */}
      <svg className="absolute left-1/2 top-[38%] h-[140vmax] w-[140vmax] -translate-x-1/2 -translate-y-1/2 opacity-[0.22] dark:opacity-[0.16]" viewBox="0 0 100 100">
        {[18, 28, 39, 49].map((r, i) => (
          <ellipse key={r} cx="50" cy="50" rx={r} ry={r * 0.36} fill="none" stroke={i % 2 ? "#a78bfa" : "#67e8f9"} strokeWidth="0.05" strokeDasharray={i === 3 ? "0.4 0.6" : undefined} />
        ))}
      </svg>
    </div>
  );
}
