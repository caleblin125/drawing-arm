"""
vector_input.py - turn SVG files and raster images into pen strokes.

Every loader returns a list of strokes; a stroke is an (n, 2) float array of
points in the source's own units with y pointing DOWN (SVG / image
convention).  fit_to_box() then scales them onto the paper (y up, mm).
"""
import math
import re
import xml.etree.ElementTree as ET

import numpy as np


# =============================================================== SVG
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


class _Scanner:
    """Path-data scanner that copes with compact numbers ("1.5.5", "0 01")."""

    def __init__(self, s):
        self.s, self.i = s, 0

    def _skip(self):
        while self.i < len(self.s) and self.s[self.i] in " \t\r\n,":
            self.i += 1

    def done(self):
        self._skip()
        return self.i >= len(self.s)

    def peek_cmd(self):
        self._skip()
        if self.i < len(self.s) and self.s[self.i].isalpha():
            return self.s[self.i]
        return None

    def cmd(self):
        c = self.peek_cmd()
        self.i += 1
        return c

    def num(self):
        self._skip()
        m = _NUM.match(self.s, self.i)
        if not m:
            raise ValueError(f"bad path data near: {self.s[self.i:self.i + 20]!r}")
        self.i = m.end()
        return float(m.group())

    def flag(self):
        self._skip()
        c = self.s[self.i]
        self.i += 1
        return c == "1"

    def has_num(self):
        self._skip()
        return self.i < len(self.s) and (self.s[self.i].isdigit() or self.s[self.i] in "+-.")


def _mat_mul(a, b):
    return a @ b


def _parse_transform(s):
    M = np.eye(3)
    for name, args in re.findall(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)", s or ""):
        v = [float(x) for x in _NUM.findall(args)]
        T = np.eye(3)
        if name == "matrix" and len(v) == 6:
            T = np.array([[v[0], v[2], v[4]], [v[1], v[3], v[5]], [0, 0, 1]])
        elif name == "translate":
            T[0, 2] = v[0]
            T[1, 2] = v[1] if len(v) > 1 else 0
        elif name == "scale":
            T[0, 0] = v[0]
            T[1, 1] = v[1] if len(v) > 1 else v[0]
        elif name == "rotate":
            a = math.radians(v[0])
            R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
            if len(v) == 3:
                cx, cy = v[1], v[2]
                T = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]]) @ R @ np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
            else:
                T = R
        elif name == "skewX":
            T[0, 1] = math.tan(math.radians(v[0]))
        elif name == "skewY":
            T[1, 0] = math.tan(math.radians(v[0]))
        M = M @ T
    return M


def _bez(pts, n):
    t = np.linspace(0, 1, n)[:, None]
    p = [np.asarray(q, float) for q in pts]
    if len(p) == 3:
        return (1 - t) ** 2 * p[0] + 2 * (1 - t) * t * p[1] + t ** 2 * p[2]
    return (1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3]


def _nseg(pts, tol):
    L = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    return int(min(200, max(4, L / tol)))


def _arc(p0, rx, ry, phi, large, sweep, p1, tol):
    """SVG endpoint arc -> points (F.6.5 of the SVG spec)."""
    if rx == 0 or ry == 0 or p0 == p1:
        return np.array([p1])
    rx, ry = abs(rx), abs(ry)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (p0[0] - p1[0]) / 2, (p0[1] - p1[1]) / 2
    x1, y1 = cp * dx + sp * dy, -sp * dx + cp * dy
    lam = x1 ** 2 / rx ** 2 + y1 ** 2 / ry ** 2
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num = rx ** 2 * ry ** 2 - rx ** 2 * y1 ** 2 - ry ** 2 * x1 ** 2
    den = rx ** 2 * y1 ** 2 + ry ** 2 * x1 ** 2
    co = math.sqrt(max(0.0, num / den)) * (-1 if large == sweep else 1)
    cx1, cy1 = co * rx * y1 / ry, -co * ry * x1 / rx
    cx = cp * cx1 - sp * cy1 + (p0[0] + p1[0]) / 2
    cy = sp * cx1 + cp * cy1 + (p0[1] + p1[1]) / 2
    t1 = math.atan2((y1 - cy1) / ry, (x1 - cx1) / rx)
    t2 = math.atan2((-y1 - cy1) / ry, (-x1 - cx1) / rx)
    dt = t2 - t1
    if sweep and dt < 0:
        dt += 2 * math.pi
    elif not sweep and dt > 0:
        dt -= 2 * math.pi
    n = int(min(200, max(4, abs(dt) * max(rx, ry) / tol)))
    t = t1 + dt * np.linspace(0, 1, n + 1)[1:]
    x, y = rx * np.cos(t), ry * np.sin(t)
    return np.stack([cp * x - sp * y + cx, sp * x + cp * y + cy], axis=1)


def parse_path(d, tol=0.5):
    sc = _Scanner(d)
    strokes, cur = [], []
    pos = start = (0.0, 0.0)
    last_ctrl, last_cmd = None, ""
    cmd = None

    def flush():
        nonlocal cur
        if len(cur) > 1:
            strokes.append(np.array(cur))
        cur = []

    while not sc.done():
        c = sc.peek_cmd()
        if c:
            cmd = sc.cmd()
        elif cmd is None:
            break
        rel = cmd.islower()
        C = cmd.upper()
        ox, oy = pos if rel else (0.0, 0.0)

        if C == "Z":
            if cur:
                cur.append(start)
            flush()
            pos = start
            cur = [start]
            last_ctrl = None
            last_cmd = "Z"
            cmd = None
            continue
        if C == "M":
            flush()
            pos = (ox + sc.num(), oy + sc.num())
            start = pos
            cur = [pos]
            cmd = "l" if rel else "L"            # implicit lineto after moveto
            last_ctrl, last_cmd = None, "M"
            continue
        if not cur:
            cur = [pos]
        if C == "L":
            pos = (ox + sc.num(), oy + sc.num())
            cur.append(pos)
            last_ctrl = None
        elif C == "H":
            pos = ((ox if rel else 0) + sc.num(), pos[1])
            cur.append(pos)
            last_ctrl = None
        elif C == "V":
            pos = (pos[0], (oy if rel else 0) + sc.num())
            cur.append(pos)
            last_ctrl = None
        elif C in "CS":
            if C == "C":
                c1 = (ox + sc.num(), oy + sc.num())
            else:
                c1 = (2 * pos[0] - last_ctrl[0], 2 * pos[1] - last_ctrl[1]) if last_cmd in "CS" and last_ctrl else pos
            c2 = (ox + sc.num(), oy + sc.num())
            p1 = (ox + sc.num(), oy + sc.num())
            pts = [pos, c1, c2, p1]
            cur.extend(map(tuple, _bez(pts, _nseg(pts, tol))[1:]))
            pos, last_ctrl = p1, c2
        elif C in "QT":
            if C == "Q":
                c1 = (ox + sc.num(), oy + sc.num())
            else:
                c1 = (2 * pos[0] - last_ctrl[0], 2 * pos[1] - last_ctrl[1]) if last_cmd in "QT" and last_ctrl else pos
            p1 = (ox + sc.num(), oy + sc.num())
            pts = [pos, c1, p1]
            cur.extend(map(tuple, _bez(pts, _nseg(pts, tol))[1:]))
            pos, last_ctrl = p1, c1
        elif C == "A":
            rx, ry, rot = sc.num(), sc.num(), math.radians(sc.num())
            large, sweep = sc.flag(), sc.flag()
            p1 = (ox + sc.num(), oy + sc.num())
            cur.extend(map(tuple, _arc(pos, rx, ry, rot, large, sweep, p1, tol)))
            pos, last_ctrl = p1, None
        else:
            raise ValueError(f"unknown path command {cmd!r}")
        last_cmd = C
    flush()
    return strokes


def _f(el, k, default=0.0):
    v = el.get(k)
    if v is None:
        return default
    m = _NUM.match(v.strip())
    return float(m.group()) if m else default


def _ellipse(cx, cy, rx, ry, n=96):
    t = np.linspace(0, 2 * math.pi, n + 1)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], axis=1)


SKIP_TAGS = {"defs", "clipPath", "mask", "symbol", "marker", "pattern", "style", "title", "desc", "metadata", "text"}


def load_svg(path, tol=0.5):
    root = ET.parse(path).getroot()
    out = []

    def walk(el, M):
        tag = el.tag.split("}")[-1]
        if tag in SKIP_TAGS:
            return
        if (el.get("display") == "none") or ("display:none" in (el.get("style") or "").replace(" ", "")):
            return
        M = M @ _parse_transform(el.get("transform"))
        local = []
        if tag == "path" and el.get("d"):
            local = parse_path(el.get("d"), tol)
        elif tag == "line":
            local = [np.array([[_f(el, "x1"), _f(el, "y1")], [_f(el, "x2"), _f(el, "y2")]])]
        elif tag in ("polyline", "polygon"):
            v = [float(x) for x in _NUM.findall(el.get("points", ""))]
            pts = np.array(v[: len(v) // 2 * 2]).reshape(-1, 2)
            if tag == "polygon" and len(pts):
                pts = np.vstack([pts, pts[:1]])
            if len(pts) > 1:
                local = [pts]
        elif tag == "rect":
            x, y, w, h = _f(el, "x"), _f(el, "y"), _f(el, "width"), _f(el, "height")
            if w > 0 and h > 0:
                local = [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]])]
        elif tag == "circle":
            r = _f(el, "r")
            if r > 0:
                local = [_ellipse(_f(el, "cx"), _f(el, "cy"), r, r)]
        elif tag == "ellipse":
            rx, ry = _f(el, "rx"), _f(el, "ry")
            if rx > 0 and ry > 0:
                local = [_ellipse(_f(el, "cx"), _f(el, "cy"), rx, ry)]
        for s in local:
            h = np.hstack([s, np.ones((len(s), 1))]) @ M.T
            out.append(h[:, :2])
        for ch in el:
            walk(ch, M)

    walk(root, np.eye(3))
    return out


# ============================================================== raster images
def _load_gray(path, max_px):
    from PIL import Image, ImageOps
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("L")
    s = max_px / max(im.size)
    if s < 1:
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    return np.asarray(im, dtype=float) / 255.0          # 0 = black, 1 = white


def _blur(a, sigma):
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(a, sigma)
    except ImportError:
        return a


def image_outline(path, max_px=320, levels=(0.5,), blur=1.0):
    """Iso-contours of the (blurred) brightness -> outlines of dark shapes."""
    import contourpy
    g = _blur(_load_gray(path, max_px), blur)
    gen = contourpy.contour_generator(z=g, line_type=contourpy.LineType.Separate)
    strokes = []
    for lv in levels:
        for line in gen.lines(lv):
            if len(line) > 2:
                strokes.append(np.asarray(line))           # (x=col, y=row): y already down
    return strokes


def image_hatch(path, max_px=180, spacing=3.0, tones=((0.80, 45), (0.55, -45), (0.30, 0))):
    """Tone-dependent hatching: each (threshold, angle) adds a family of
    parallel lines (spacing in pixels) wherever brightness < threshold."""
    g = _blur(_load_gray(path, max_px), 0.7)
    H, W = g.shape
    strokes = []
    diag = math.hypot(W, H)
    for thr, angle in tones:
        a = math.radians(angle)
        d = np.array([math.cos(a), math.sin(a)])          # along the line
        nrm = np.array([-d[1], d[0]])                      # across lines
        c = np.array([W / 2, H / 2])
        ts = np.arange(-diag / 2, diag / 2, 0.5)
        flip = False
        for off in np.arange(-diag / 2, diag / 2, spacing):
            pts = c + off * nrm + ts[:, None] * d
            inside = (pts[:, 0] >= 0) & (pts[:, 0] <= W - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= H - 1)
            if not inside.any():
                continue
            xi = np.clip(pts[:, 0].round().astype(int), 0, W - 1)
            yi = np.clip(pts[:, 1].round().astype(int), 0, H - 1)
            dark = inside & (g[yi, xi] < thr)
            edges = np.flatnonzero(np.diff(np.r_[0, dark.astype(int), 0]))
            segs = [pts[s:e] for s, e in zip(edges[::2], edges[1::2]) if e - s >= 3]
            if flip:                                       # boustrophedon -> short travels
                segs = [s[::-1] for s in segs[::-1]]
            strokes += [np.array([s[0], s[-1]]) for s in segs]
            flip = not flip
    return strokes


def load_image(path, mode="outline", **kw):
    if mode == "outline":
        return image_outline(path, **kw)
    if mode == "hatch":
        return image_hatch(path, **kw)
    if mode == "both":
        return image_outline(path) + image_hatch(path)
    raise ValueError(mode)


# ================================================================ utilities
def rdp(pts, eps):
    """Ramer-Douglas-Peucker simplification (iterative)."""
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        a, b = pts[i], pts[j]
        ab = b - a
        L = math.hypot(*ab)
        seg = pts[i + 1:j]
        if len(seg) == 0:
            continue
        if L < 1e-12:
            dd = np.hypot(*(seg - a).T)
        else:
            dd = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / L
        k = int(np.argmax(dd))
        if dd[k] > eps:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return pts[keep]


def fit_to_box(strokes, box, margin=8.0, simplify=0.15, min_len=1.0):
    """Scale/translate strokes (y-down source) into box=(x0,y0,x1,y1) mm, y up."""
    strokes = [np.asarray(s, float) for s in strokes if len(s) > 1]
    if not strokes:
        return []
    allp = np.vstack(strokes)
    lo, hi = allp.min(0), allp.max(0)
    span = np.maximum(hi - lo, 1e-9)
    x0, y0, x1, y1 = box
    s = min((x1 - x0 - 2 * margin) / span[0], (y1 - y0 - 2 * margin) / span[1])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    mid = (lo + hi) / 2
    out = []
    for st in strokes:
        p = np.empty_like(st)
        p[:, 0] = cx + (st[:, 0] - mid[0]) * s
        p[:, 1] = cy - (st[:, 1] - mid[1]) * s            # flip y
        p = rdp(p, simplify)
        if np.sum(np.hypot(*np.diff(p, axis=0).T)) >= min_len:
            out.append(p)
    return out
