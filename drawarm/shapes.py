"""Built-in test drawings (strokes in a 0..200 box, y down like SVG/images)."""
import math

import numpy as np


def builtin_shape(name):
    """Strokes in a 0..200 box, y down (same convention as SVG/images)."""
    c = np.array([100.0, 100.0])
    t = np.linspace(0, 2 * math.pi, 241)
    if name == "circle":
        return [c + 90 * np.stack([np.cos(t), np.sin(t)], 1)]
    if name == "star":
        k = np.arange(11)
        r = np.where(k % 2 == 0, 95, 38)
        a = -math.pi / 2 + k * math.pi / 5
        return [c + np.stack([r * np.cos(a), r * np.sin(a)], 1)]
    if name == "spiral":
        tt = np.linspace(0, 6 * math.pi, 600)
        r = 4 + 4.5 * tt
        return [c + np.stack([r * np.cos(tt), r * np.sin(tt)], 1)]
    if name == "heart":
        x = 16 * np.sin(t) ** 3
        y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        return [c + 5.5 * np.stack([x, y], 1)]
    if name == "lissajous":
        tt = np.linspace(0, 2 * math.pi, 800)
        return [c + np.stack([95 * np.sin(3 * tt), 80 * np.sin(4 * tt)], 1)]
    if name == "flower":
        r = 50 + 40 * np.cos(5 * t)
        return [c + np.stack([r * np.cos(t), r * np.sin(t)], 1)]
    if name == "grid":
        s = []
        for v in range(10, 191, 30):
            s += [np.array([[v, 10.0], [v, 190.0]]), np.array([[10.0, v], [190.0, v]])]
        return s
    raise KeyError(name)


SHAPES = ["circle", "star", "spiral", "heart", "lissajous", "flower", "grid"]


def load_drawing(src, image_mode="outline"):
    """Strokes from a built-in shape name, an .svg file or a raster image."""
    from .inputs import load_image, load_svg
    low = src.lower()
    if low.endswith(".svg"):
        return load_svg(src)
    if low.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
        return load_image(src, image_mode)
    if src not in SHAPES:
        raise SystemExit(f"unknown drawing {src!r}: use one of {', '.join(SHAPES)}, an .svg or an image")
    return builtin_shape(src)


def drawing_name(src):
    import os
    return os.path.splitext(os.path.basename(src))[0]
