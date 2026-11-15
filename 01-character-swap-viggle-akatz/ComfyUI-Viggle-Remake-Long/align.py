# SPDX-License-Identifier: Apache-2.0
"""Line a painted keyframe up with its source frame: scale + shift search, colour match on the unchanged background,
and a fill for any edge the alignment uncovers (source background there, never the replaced person). numpy only."""
import numpy as np
from PIL import Image


def box(a, r):
    """Box blur of a 2-D array with radius r (cumulative sums)."""
    if r <= 0:
        return a
    p = np.pad(a, r + 1, mode="edge")
    c = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s[: a.shape[0], : a.shape[1]] / (k * k)


def warp(img, s, dx, dy):
    """Scale by s about the centre, then shift by (dx, dy) pixels; bilinear; outside the image mirrors back in."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sx = (xx - dx - w / 2) / s + w / 2
    sy = (yy - dy - h / 2) / s + h / 2
    sx = np.where(sx < 0, -sx, np.where(sx > w - 1, 2 * (w - 1) - sx, sx))
    sy = np.where(sy < 0, -sy, np.where(sy > h - 1, 2 * (h - 1) - sy, sy))
    x0 = np.clip(np.floor(sx).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(sy).astype(int), 0, h - 2)
    fx = np.clip(sx - x0, 0, 1)[..., None]
    fy = np.clip(sy - y0, 0, 1)[..., None]
    a, b = img[y0, x0], img[y0, x0 + 1]
    c, d = img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def shift_by_phase(a, b):
    """Translation that moves b onto a (phase correlation)."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa, fb = np.fft.fft2(a * win), np.fft.fft2(b * win)
    r = fa * np.conj(fb)
    r /= np.abs(r) + 1e-6
    corr = np.fft.ifft2(r).real
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = a.shape
    return (x - w if x > w // 2 else x), (y - h if y > h // 2 else y)


def align_keyframe(s_img: np.ndarray, p_img: np.ndarray, keep_unchanged: bool = False):
    """s_img: source frame, p_img: painting, both HxWx3 float 0-255. Returns (aligned painting at the source size,
    report)."""
    h, w = s_img.shape[:2]
    if p_img.shape[:2] != (h, w):
        p_img = np.asarray(Image.fromarray(np.clip(p_img, 0, 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
                           dtype=np.float32)

    # 1. alignment, searched at quarter size on grey images; the unchanged background dominates the median
    q = 4
    gs, gp = s_img[::q, ::q].mean(2), p_img[::q, ::q].mean(2)
    best = (1e9, 1.0, 0.0, 0.0)
    for s in np.linspace(0.92, 1.08, 33):
        g = warp(gp[..., None], s, 0, 0)[..., 0]
        dx, dy = shift_by_phase(gs, g)
        score = np.median(np.abs(warp(g[..., None], 1.0, dx, dy)[..., 0] - gs)[8:-8, 8:-8])
        if score < best[0]:
            best = (score, s, dx * q, dy * q)
    _, s, dx, dy = best
    aligned = warp(p_img, s, dx, dy)

    # 2. colour match, measured where the two images already agree (mostly the unchanged background)
    diff = box(np.abs(aligned - s_img).mean(2), 12)
    bg = diff < np.percentile(diff, 50)
    fixed = aligned.copy()
    for ch in range(3):
        a, b = aligned[..., ch][bg], s_img[..., ch][bg]
        fixed[..., ch] = (aligned[..., ch] - a.mean()) * (b.std() / (a.std() + 1e-6)) + b.mean()

    # 3. edges the alignment uncovers (the painting was zoomed in): source background there; where the new person
    #    runs into that edge, the painted person mirrored on (the source still shows the old person there)
    yy, xx = np.mgrid[0:h, 0:w]
    sx = (xx - dx - w / 2) / s + w / 2
    sy = (yy - dy - h / 2) / s + h / 2
    covered = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    result = fixed
    edge_note = "no uncovered edge"
    if (~covered).any():
        ys, xs = np.where(covered)
        y0, y1, x0, x1 = ys.min() + 2, ys.max() - 2, xs.min() + 2, xs.max() - 2
        person = np.clip((box(np.abs(fixed - s_img).mean(2), 10) - 25) / 20, 0, 1)
        person_ext = person[np.clip(yy, y0, y1), np.clip(xx, x0, x1)]
        person_ext = np.clip(box((box(person_ext, 4) > 0.15).astype(np.float32), 30) * 2.5, 0, 1)
        fill = fixed * person_ext[..., None] + s_img * (1 - person_ext[..., None])
        soft = np.clip(box((~covered).astype(np.float32), 4) * 2, 0, 1)[..., None]
        result = fixed * (1 - soft) + fill * soft
        edge_note = f"filled an uncovered edge of {(~covered).mean() * 100:.1f}% of the frame"
    if keep_unchanged:
        # only what was edited: where the aligned painting differs clearly from the source, widened and feathered
        d = box(np.abs(result - s_img).mean(2), 6)
        m = (d > 28).astype(np.float32)
        m = np.clip(box(np.clip(box(m, 12) * 4, 0, 1), 6), 0, 1)[..., None]   # widened ~18 px, feathered ~6 px
        result = s_img * (1 - m) + result * m
        edge_note += f"; kept the source everywhere except the edited {float(m.mean()) * 100:.0f}% of the frame"
    before = np.abs(p_img - s_img).mean(2)[bg].mean()
    after = np.abs(result - s_img).mean(2)[bg].mean()
    warn = ""
    if (~covered).mean() > 0.01:   # the painting was zoomed in / moved: part of the frame had to be filled in
        warn = (f" WARNING: {(~covered).mean() * 100:.0f}% of the frame was missing from the painting and had to be "
                "filled; check the edges, or paint again from Viggle Pick Frame with editor_margin 7.")
    report = (f"scale {s:.3f}, shift ({dx:+.0f}, {dy:+.0f}) px, {edge_note}, background difference "
              f"{before:.1f} -> {after:.1f}.{warn}")
    return result, report
