"""Dataset insights for any image collection: classify every image and summarise the set.

Per image (no model needed):
  * region map: terrain / rover hardware / shadow / sky. Hardware is found from dense, strong
    edges (machined parts, cables, panels) with enclosed panels filled in; shadow = large very dark
    areas; sky = the smooth band above the horizon. Saved as a colour overlay.
  * scene type from the region shares: rover hardware close-up / terrain with hardware in view /
    landscape with horizon / ground close view / blank frame - with the numbers that decided it
  * terrain-only measurements: texture (fine / textured / blocky), a roughness heat map, and the
    number of rocks/pebbles (small high-contrast blobs on the ground) per 10 000 terrain pixels
  * exposure, contrast, sharpness, perceptual hash (near-duplicates), NASA product-id parsing
With the Mars terrain model installed, the terrain pixels are also segmented into soil / bedrock /
sand / big rock (hardware, shadow and sky are excluded, as in the AI4Mars training labels).

The set is grouped (k-means on the image features, k chosen by silhouette) and summarised in
plain-language findings. Heuristic results are labelled as such; nothing here is ground truth.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .profile import parse_name
from .store import WorkbenchError, get, load_image

MAX_IMAGES = 60
WORK_SIDE = 512
TERRAIN = ["soil", "bedrock", "sand", "big rock"]
TERRAIN_RGB = np.array([[250, 204, 21], [59, 130, 246], [244, 63, 94], [16, 185, 129]], np.uint8)
REGION_RGB = {"hardware": (99, 102, 241), "shadow": (30, 27, 75), "sky": (56, 189, 248)}
REGIONS = ["terrain", "hardware", "shadow", "sky"]
SCENES = {"hardware": "Rover hardware close-up", "mixed": "Terrain with rover hardware in view",
          "landscape": "Landscape with horizon", "ground": "Ground / terrain close view",
          "blank": "Dark, blank or calibration frame"}


# ------------------------------------------------------------------ image → arrays
def _prep(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """(gray 0-1, rgb 0-1 or None) downscaled so the long side is WORK_SIDE."""
    a = np.nan_to_num(arr.astype(np.float64))
    hi = a.max() if a.size else 1
    if hi > 255:
        lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
        a = (a - lo) / max(hi - lo, 1e-9)
    elif hi > 1.0:
        a = a / 255.0
    a = np.clip(a, 0, 1)
    rgb = None
    if a.shape[0] >= 3 and not (np.allclose(a[0], a[1], atol=2e-3) and np.allclose(a[1], a[2], atol=2e-3)):
        rgb = np.moveaxis(a[:3], 0, -1)
    g = a[:3].mean(0) if a.shape[0] >= 3 else a[0]
    h, w = g.shape
    f = WORK_SIDE / max(h, w)
    if f < 1:
        size = (max(16, int(w * f)), max(16, int(h * f)))
        g = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize(size, Image.BILINEAR), np.float64) / 255
        if rgb is not None:
            rgb = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).resize(size, Image.BILINEAR),
                             np.float64) / 255
    return g, rgb


def _dhash(g: np.ndarray) -> int:
    small = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((9, 8), Image.BOX), np.int16)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


# ------------------------------------------------------------------ regions
def region_map(g: np.ndarray) -> tuple[np.ndarray, dict]:
    """Label every pixel 0 terrain, 1 hardware, 2 shadow, 3 sky."""
    h, w = g.shape
    gx, gy = ndimage.sobel(g, 1), ndimage.sobel(g, 0)
    mag = np.hypot(gx, gy)
    strong = mag > 0.35
    dens = ndimage.uniform_filter(strong.astype(float), 15)
    extreme = (g > 0.62) | (g < 0.07)
    near_structure = ndimage.uniform_filter(strong.astype(float), 41) > 0.03
    hw = (dens > 0.06) | ((ndimage.uniform_filter(extreme.astype(float), 15) > 0.5) & near_structure)
    hw = ndimage.binary_closing(hw, iterations=4)
    hw = ndimage.binary_opening(hw, iterations=2)
    hw = _keep_large(hw, 0.01)
    # smooth panels enclosed by hardware edges belong to the rover too
    holes = ndimage.binary_fill_holes(hw) & ~hw
    lab, n = ndimage.label(holes)
    if n:
        sizes = ndimage.sum(holes, lab, range(1, n + 1))
        hw |= np.isin(lab, 1 + np.where(sizes <= 0.08 * hw.size)[0])
    # deep shadow (under the rover, inside the chassis)
    dark = ndimage.uniform_filter((g < 0.08).astype(float), 11) > 0.7
    shadow = _keep_large(dark & ~hw, 0.01)
    # sky: smooth band connected to the top edge
    sky = np.zeros_like(hw)
    hz = _horizon_row(g)
    if hz:
        sky[:hz] = True
        sky &= ~hw
    lab_map = np.zeros((h, w), np.uint8)
    lab_map[hw] = 1
    lab_map[shadow] = 2
    lab_map[sky] = 3
    shares = {r: float((lab_map == i).mean()) for i, r in enumerate(REGIONS)}
    return lab_map, shares


def _keep_large(mask: np.ndarray, frac: float) -> np.ndarray:
    lab, n = ndimage.label(mask)
    if not n:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.where(sizes >= frac * mask.size)[0])


def _horizon_row(g: np.ndarray) -> int:
    grad = np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))
    row_tex = ndimage.uniform_filter1d(grad.mean(1), 5)
    row_mean = g.mean(1)
    med_tex = np.median(row_tex)
    ground_mean = float(np.median(row_mean[len(g) // 2:]))
    # sky is very smooth AND brighter than the ground below it (distant smooth ground is darker)
    sky = (row_tex < 0.3 * med_tex) & (row_mean > 1.12 * ground_mean)
    n = 0
    for s in sky:
        if not s:
            break
        n += 1
    return n if n >= max(3, int(0.04 * len(g))) else 0


# ------------------------------------------------------------------ measurements
def measure(g: np.ndarray, lab: np.ndarray) -> dict:
    lap = ndimage.laplace(g)
    mag = np.hypot(ndimage.sobel(g, 1), ndimage.sobel(g, 0))
    hist = np.histogram(g, bins=64, range=(0, 1))[0].astype(float)
    p = hist / max(hist.sum(), 1)
    terr = lab == 0
    terr[:5], terr[-5:], terr[:, :5], terr[:, -5:] = False, False, False, False
    loc = np.sqrt(np.maximum(ndimage.uniform_filter(g ** 2, 9) - ndimage.uniform_filter(g, 9) ** 2, 0))
    t = loc[terr] if terr.sum() > 200 else np.array([0.0])
    return {"brightness": float(g.mean()), "contrast": float(g.std()), "sharpness": float(lap.var() * 1e3),
            "edge_density": float((mag > 0.25).mean()), "entropy": float(-(p[p > 0] * np.log2(p[p > 0])).sum()),
            "clipped_high": float((g > 0.985).mean()), "clipped_low": float((g < 0.015).mean()),
            "fine": float((t < 0.02).mean()), "blocky": float((t > 0.07).mean()),
            "roughness": float(np.median(t)), "terrain_brightness": float(g[terr].mean()) if terr.any() else 0.0}, loc


def rocks(g: np.ndarray, lab: np.ndarray) -> list[tuple[int, int, int]]:
    """Small high-contrast blobs on terrain (pebbles, rocks): (row, col, radius) at work scale."""
    terr = ndimage.binary_erosion(lab == 0, iterations=3)
    if terr.sum() < 500:
        return []
    dog = ndimage.gaussian_filter(g, 1.2) - ndimage.gaussian_filter(g, 4)
    thr = max(0.035, 4 * np.median(np.abs(dog[terr])))
    blobs = (np.abs(dog) > thr) & terr
    lb, n = ndimage.label(blobs)
    if not n:
        return []
    out = []
    for i, sl in enumerate(ndimage.find_objects(lb), 1):
        if sl is None:
            continue
        size = int((lb[sl] == i).sum())
        if 4 <= size <= 600:
            r0, c0 = (sl[0].start + sl[0].stop) // 2, (sl[1].start + sl[1].stop) // 2
            out.append((r0, c0, max(2, int(np.sqrt(size / np.pi)) + 1)))
    return out[:400]


def scene_of(f: dict, reg: dict) -> tuple[str, list[str]]:
    if f["contrast"] < 0.012 or f["brightness"] < 0.03 or f["entropy"] < 1.5:
        return "blank", [f"very low contrast ({f['contrast']:.3f}) / information ({f['entropy']:.1f} bits)"]
    why = [f"hardware {reg['hardware']:.0%} · terrain {reg['terrain']:.0%}"
           + (f" · shadow {reg['shadow']:.0%}" if reg["shadow"] > 0.02 else "")
           + (f" · sky {reg['sky']:.0%}" if reg["sky"] > 0.02 else "")]
    if reg["hardware"] + reg["shadow"] >= 0.5:
        return "hardware", why + ["dense machined edges, panels and cables cover most of the frame"]
    if reg["hardware"] >= 0.12:
        return "mixed", why + ["rover parts in the foreground, open ground behind"]
    if reg["sky"] > 0.05:
        return "landscape", why + [f"smooth sky band over the top {reg['sky']:.0%}"]
    return "ground", why + ["natural texture fills the frame"]


# ------------------------------------------------------------------ pictures
def _save_region_png(g, rgb, lab, rk, out: Path) -> None:
    base = (rgb if rgb is not None else np.repeat(g[..., None], 3, -1)) * 255
    vis = base.copy()
    for i, r in ((1, "hardware"), (2, "shadow"), (3, "sky")):
        m = lab == i
        vis[m] = 0.45 * vis[m] + 0.55 * np.array(REGION_RGB[r])
    im = Image.fromarray(vis.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for r0, c0, rad in rk:
        d.ellipse([c0 - rad - 1, r0 - rad - 1, c0 + rad + 1, r0 + rad + 1], outline=(251, 146, 60), width=1)
    im.thumbnail((420, 420))
    im.save(out)


def _save_texture_png(g, lab, loc, out: Path) -> None:
    terr = lab == 0
    terr[:4], terr[-4:], terr[:, :4], terr[:, -4:] = False, False, False, False   # filter edge effects
    top = max(0.03, float(np.percentile(loc[terr], 98))) if terr.sum() > 200 else 0.1
    v = np.clip(loc / top, 0, 1) ** 0.7
    # perceptual-ish ramp: deep blue (smooth) → teal → yellow → red (rough)
    stops = np.array([[30, 41, 99], [20, 150, 160], [250, 204, 21], [220, 38, 38]], float)
    x = v * (len(stops) - 1)
    i0 = np.clip(x.astype(int), 0, len(stops) - 2)
    t = (x - i0)[..., None]
    col = stops[i0] * (1 - t) + stops[i0 + 1] * t
    gray = np.repeat(g[..., None] * 255, 3, -1)
    vis = np.where(terr[..., None], 0.35 * gray + 0.65 * col, 0.25 * gray)
    im = Image.fromarray(vis.clip(0, 255).astype(np.uint8))
    im.thumbnail((420, 420))
    im.save(out)


# ------------------------------------------------------------------ run
def run(dataset_id: str, p: dict, run_dir: Path, version_dir: Path | None) -> dict:
    meta = get(dataset_id)
    imgs = meta.get("images", [])
    if not imgs:
        raise WorkbenchError("This dataset has no images")
    sel = imgs[:MAX_IMAGES]
    model = None
    if version_dir is not None and p.get("use_model", True):
        try:
            from ..mars.infer import _model
            ck = version_dir / "model.pt"
            model = _model(str(version_dir), ck.stat().st_mtime)[0]
        except Exception as e:  # noqa: BLE001 - insights still work without the model
            print(f"Mars model not usable: {e}")
    (run_dir / "img").mkdir(parents=True, exist_ok=True)
    rows = []
    for k, e in enumerate(sel):
        try:
            _, arr, geo = load_image(dataset_id, e["id"])
            g, rgb = _prep(arr)
        except Exception as ex:  # noqa: BLE001
            rows.append({"id": e["id"], "file": e["file"], "error": str(ex)[:200]})
            continue
        lab, reg = region_map(g)
        f, loc = measure(g, lab)
        scene, why = scene_of(f, reg)
        rk = rocks(g, lab) if scene != "blank" else []
        terr_px = int((lab == 0).sum())
        th = Image.fromarray(((rgb if rgb is not None else np.repeat(g[..., None], 3, -1)) * 255).astype(np.uint8))
        th.thumbnail((420, 420))
        th.save(run_dir / "img" / f"{e['id']}.jpg", quality=85)
        _save_region_png(g, rgb, lab, rk, run_dir / "img" / f"{e['id']}_regions.png")
        _save_texture_png(g, lab, loc, run_dir / "img" / f"{e['id']}_texture.png")
        row = {"id": e["id"], "file": e["file"], "width": e.get("width"), "height": e.get("height"),
               "bands": e.get("bands"), "color": rgb is not None, "features": f, "regions": reg,
               "scene": scene, "scene_label": SCENES[scene], "why": why,
               "texture": {"fine": f["fine"], "textured": max(0.0, 1 - f["fine"] - f["blocky"]), "blocky": f["blocky"]},
               "rocks": len(rk), "rock_density": round(len(rk) / max(terr_px, 1) * 1e4, 2),
               "hash": _dhash(g), "nasa": parse_name(e["file"]), "thumb": f"img/{e['id']}.jpg",
               "region_map": f"img/{e['id']}_regions.png", "texture_map": f"img/{e['id']}_texture.png",
               "georeferenced": bool((geo or {}).get("crs"))}
        if model is not None and scene != "blank" and reg["terrain"] > 0.05:
            row["terrain"] = _segment(model, g, rgb, lab, run_dir / "img" / f"{e['id']}_terrain.png")
            row["overlay"] = f"img/{e['id']}_terrain.png"
        rows.append(row)
    ok = [r for r in rows if "features" in r]
    if not ok:
        raise WorkbenchError("None of the images could be read")
    groups = _group(ok)
    dups = _duplicates(ok)
    for r in ok:
        r["hash"] = f"{r['hash']:016x}"
    summary = _summary(ok, groups, dups, model is not None, len(imgs))
    return {"dataset_id": dataset_id, "n_images": len(imgs), "n_analysed": len(ok), "images": rows,
            "groups": groups, "duplicates": dups, "summary": summary["stats"], "insights": summary["insights"],
            "model_used": model is not None, "settings": p,
            "scene_counts": dict(Counter(r["scene"] for r in ok)), "scene_labels": SCENES,
            "terrain_classes": TERRAIN, "region_classes": REGIONS,
            "region_colors": {"terrain": None, **{k: list(v) for k, v in REGION_RGB.items()}}}


def _segment(model, g, rgb, lab, out: Path) -> dict:
    """Terrain classes on terrain pixels only (hardware/shadow/sky excluded)."""
    from ..mars.model import segment
    src = rgb if rgb is not None else np.repeat(g[..., None], 3, -1)
    img = Image.fromarray((src * 255).astype(np.uint8))
    pred = segment(model, img, 256 if max(img.size) <= 600 else 384)
    if pred.shape != lab.shape:
        pred = np.asarray(Image.fromarray(pred.astype(np.uint8)).resize(lab.shape[::-1], Image.NEAREST))
    terr = lab == 0
    cnt = np.bincount(pred[terr].ravel(), minlength=4)[:4]
    shares = cnt / max(cnt.sum(), 1)
    base = src * 255
    col = TERRAIN_RGB[np.clip(pred, 0, 3)].astype(np.float64)
    vis = np.where(terr[..., None], 0.5 * base + 0.5 * col, 0.3 * base)
    o = Image.fromarray(vis.clip(0, 255).astype(np.uint8))
    o.thumbnail((420, 420))
    o.save(out)
    return {c: float(s) for c, s in zip(TERRAIN, shares)}


FEATS = ["brightness", "contrast", "sharpness", "edge_density", "entropy", "fine", "blocky", "roughness"]


def _vec(r: dict) -> list[float]:
    f = r["features"]
    return [f[k] for k in FEATS] + [r["regions"][k] for k in REGIONS]


def _group(rows: list[dict]) -> list[dict]:
    n = len(rows)
    if n < 4:
        for r in rows:
            r["group"] = 0
        return [{"id": 0, "label": "All images", "members": [r["id"] for r in rows]}]
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    X = np.array([_vec(r) for r in rows], float)
    X[:, 2] = np.log1p(X[:, 2])
    X = (X - X.mean(0)) / np.where(X.std(0) > 1e-9, X.std(0), 1)
    best, labels = -1.0, np.zeros(n, int)
    for k in range(2, min(5, n - 1) + 1):
        lab = KMeans(k, n_init=10, random_state=42).fit_predict(X)
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(X, lab)
        if s > best:
            best, labels = s, lab
    if best < 0.15:
        labels = np.zeros(n, int)
    out = []
    for k in sorted(set(labels)):
        mem = [rows[i] for i in range(n) if labels[i] == k]
        sc = Counter(r["scene"] for r in mem).most_common(1)[0][0]
        reg = {c: float(np.mean([r["regions"][c] for r in mem])) for c in REGIONS}
        label = SCENES[sc] + f" · {reg['terrain']:.0%} terrain"
        out.append({"id": int(k), "label": label, "members": [r["id"] for r in mem], "regions": reg,
                    "silhouette": round(best, 3) if best > 0 else None})
    out.sort(key=lambda g: -len(g["members"]))
    for r in rows:
        r["group"] = next(g["id"] for g in out if r["id"] in g["members"])
    return out


def _duplicates(rows: list[dict]) -> list[dict]:
    out = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            d = bin(rows[i]["hash"] ^ rows[j]["hash"]).count("1")
            if d <= 6:
                out.append({"a": rows[i]["id"], "b": rows[j]["id"], "a_file": rows[i]["file"],
                            "b_file": rows[j]["file"], "distance": d})
    return out[:50]


def _summary(rows: list[dict], groups: list[dict], dups: list[dict], model: bool, total: int) -> dict:
    ins: list[dict] = []
    n = len(rows)
    nasa = [r["nasa"] for r in rows if r.get("nasa")]
    if nasa:
        from .profile import _stereo_pairs
        mis = Counter(x["mission"] for x in nasa).most_common(1)[0][0]
        cams = Counter(x["camera"] for x in nasa)
        sols = sorted({x["sol"] for x in nasa if x.get("sol") is not None})
        pairs = _stereo_pairs([r["file"] for r in rows if r.get("nasa")])
        txt = (f"{len(nasa)} of {n} images are {mis} products ("
               + ", ".join(f"{c} ×{k}" for c, k in cams.most_common(3)) + ")")
        if sols:
            txt += f", sol {sols[0]}" + (f"–{sols[-1]}" if len(sols) > 1 else "")
        if pairs:
            txt += f"; {pairs} left/right stereo pair(s) - depth could be reconstructed"
        ins.append({"kind": "info", "title": "What the files are", "text": txt + "."})
    sc = Counter(r["scene"] for r in rows)
    ins.append({"kind": "classification", "title": "Scene types",
                "text": "; ".join(f"{SCENES[k]}: {v}" for k, v in sc.most_common()) + "."})
    reg = {c: float(np.mean([r["regions"][c] for r in rows])) for c in REGIONS}
    ins.append({"kind": "classification", "title": "What the pixels show",
                "text": f"On average {reg['terrain']:.0%} of each frame is Martian terrain, {reg['hardware']:.0%} rover "
                        f"hardware, {reg['shadow']:.0%} deep shadow and {reg['sky']:.0%} sky."})
    hw = sorted([r for r in rows if r["scene"] == "hardware"], key=lambda r: -r["regions"]["hardware"])
    if hw:
        ins.append({"kind": "warn", "title": "Mostly rover hardware",
                    "text": f"{len(hw)} of {n} image(s) mainly show the rover itself (deck, arm, wheels) - e.g. "
                            f"{hw[0]['file']} is {hw[0]['regions']['hardware']:.0%} hardware. Hardware pixels are excluded "
                            "from every terrain statistic below."})
    terr_imgs = [r for r in rows if r["regions"]["terrain"] > 0.25 and r["scene"] != "blank"]
    if terr_imgs:
        best = max(terr_imgs, key=lambda r: r["regions"]["terrain"])
        fine = np.mean([r["texture"]["fine"] for r in terr_imgs])
        blk = np.mean([r["texture"]["blocky"] for r in terr_imgs])
        rough = max(terr_imgs, key=lambda r: r["features"]["roughness"])
        smooth = min(terr_imgs, key=lambda r: r["features"]["roughness"])
        ins.append({"kind": "terrain", "title": "Ground texture",
                    "text": f"On the terrain pixels of {len(terr_imgs)} image(s), {fine:.0%} is fine-grained (sand/soil-like) "
                            f"and {blk:.0%} blocky (rock-like). Roughest ground: {rough['file']}; smoothest: {smooth['file']}. "
                            "See the roughness heat maps."})
        rk = sorted(terr_imgs, key=lambda r: -r["rock_density"])
        ins.append({"kind": "terrain", "title": "Rocks and pebbles",
                    "text": f"{sum(r['rocks'] for r in terr_imgs)} small rocks/pebbles detected on the ground (orange circles "
                            f"on the region maps). Rockiest: {rk[0]['file']} ({rk[0]['rock_density']:.1f} per 10k terrain px)."
                            + (" The ground is mostly smooth and fine-grained: easy driving." if fine > 0.6 else "")})
        ins.append({"kind": "info", "title": "Best terrain view",
                    "text": f"{best['file']} has the most open ground ({best['regions']['terrain']:.0%} of the frame) - the "
                            "best choice for terrain segmentation."})
    else:
        ins.append({"kind": "warn", "title": "Little open ground",
                    "text": "No image shows more than 25% terrain, so terrain statistics would be unreliable."})
    if model:
        seg = [r for r in rows if r.get("terrain")]
        if seg:
            w = np.array([r["regions"]["terrain"] for r in seg])
            avg = {c: float(np.average([r["terrain"][c] for r in seg], weights=w)) for c in TERRAIN}
            top = max(avg, key=avg.get)
            ins.append({"kind": "terrain", "title": "Terrain model (DeepLabV3+)",
                        "text": f"On terrain pixels the model finds {avg['soil']:.0%} soil, {avg['bedrock']:.0%} bedrock, "
                                f"{avg['sand']:.0%} sand and {avg['big rock']:.0%} big rock; dominant: {top}."})
            sandy = max(seg, key=lambda r: r["terrain"]["sand"] * r["regions"]["terrain"])
            if sandy["terrain"]["sand"] > 0.15:
                ins.append({"kind": "warn", "title": "Sand hazard",
                            "text": f"{sandy['file']}: {sandy['terrain']['sand']:.0%} of its ground is sand - soft sand is "
                                    "what trapped the Spirit rover. Review this mask in Review Studio."})
    else:
        ins.append({"kind": "info", "title": "Terrain classes",
                    "text": "When the Mars terrain model is installed (automatic setup), run again to label soil, bedrock, "
                            "sand and big rock on the terrain pixels."})
    med = np.median([x["features"]["sharpness"] for x in rows])
    blur = [r for r in rows if r["features"]["sharpness"] < 0.25 * med]
    if blur:
        ins.append({"kind": "warn", "title": "Possibly blurred",
                    "text": f"{len(blur)} image(s) are much less sharp than the rest (e.g. {blur[0]['file']})."})
    over = [r for r in rows if r["features"]["clipped_high"] > 0.05]
    under = [r for r in rows if r["features"]["clipped_low"] > 0.2]
    if over or under:
        ins.append({"kind": "warn", "title": "Exposure",
                    "text": (f"{len(over)} over-exposed " if over else "") + ("and " if over and under else "")
                            + (f"{len(under)} under-exposed " if under else "") + "image(s)."})
    if dups:
        ins.append({"kind": "warn", "title": "Near-duplicates",
                    "text": f"{len(dups)} pair(s) of almost identical images (e.g. {dups[0]['a_file']} ≈ "
                            f"{dups[0]['b_file']}). Keep one of each for statistics."})
    if len(groups) > 1:
        ins.append({"kind": "classification", "title": f"{len(groups)} visual groups",
                    "text": "; ".join(f"{g['label']} ({len(g['members'])})" for g in groups) + "."})
    if total > n:
        ins.append({"kind": "info", "title": "Sampled", "text": f"The first {n} of {total} images were analysed."})
    stats = {"images": n, "scenes": dict(sc), "groups": len(groups), "duplicates": len(dups), "regions": reg,
             "rocks": int(sum(r["rocks"] for r in rows)),
             "mean_brightness": float(np.mean([r["features"]["brightness"] for r in rows]))}
    return {"stats": stats, "insights": ins}
