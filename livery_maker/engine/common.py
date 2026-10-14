"""Shared building blocks for Livery Maker templates."""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field, asdict
from io import BytesIO
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fonts')

BUILTIN_FONTS = {
    'Inter Display ExtraBold Italic': 'InterDisplay-ExtraBoldItalic.otf',
    'Inter Display ExtraBold': 'InterDisplay-ExtraBold.otf',
    'Inter Display Black Italic': 'InterDisplay-BlackItalic.otf',
    'Inter Display Bold Italic': 'InterDisplay-BoldItalic.otf',
    'Inter Display Bold': 'InterDisplay-Bold.otf',
    'Inter Display SemiBold': 'InterDisplay-SemiBold.otf',
    'Inter Bold': 'Inter-Bold.otf',
}


def font_path(name_or_path: str) -> str:
    if name_or_path in BUILTIN_FONTS:
        return os.path.join(FONT_DIR, BUILTIN_FONTS[name_or_path])
    return name_or_path


def hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32)


# ---------------------------------------------------------------- data model
@dataclass
class Design:
    airline: str = 'FlyFurball'          # used for folder names, LiveryStat, spreadsheet
    title_text: str = 'FlyFurball'       # fuselage titles
    title_font: str = 'Inter Display ExtraBold Italic'
    title_scale: float = 1.0             # 0.5 .. 1.0 of the largest size that fits
    body_colour: str = '#FAFBFA'         # forward fuselage
    accent_colour: str = '#005EB8'       # tail, rear fuselage, engines
    title_colour: str = '#005EB8'
    symbol_colour: str = '#F6F6F6'       # saltire
    reg_colour: str = '#141414'
    reg_font: str = 'Inter Display Bold'
    fade_shift: float = 0.0              # -0.5 (earlier) .. +0.5 (later), fraction of the default fade length
    fade_length: float = 1.0             # 0.4 (sharp) .. 2.0 (long)
    tail_symbol: str = 'saltire'         # 'saltire' | 'image' | 'none'
    tail_image: str = ''                 # PNG with transparency, used when tail_symbol == 'image'
    tail_image_size: float = 0.55        # image height as a fraction of fin height
    saltire_width: float = 0.14          # arm width as a fraction of fin height
    winglet_symbol: bool = True          # small saltire on winglets/sharklets (saltire mode only)
    engine_colour: str = 'accent'        # 'accent' | 'body'
    tailplane_colour: str = 'accent'     # 'accent' | 'original'
    home_base: str = 'EGPH'
    # per aircraft type: {'dx': px, 'dy': px, 'size': factor} moving the fuselage titles from their default spot
    title_pos: dict = field(default_factory=dict)
    reg_pos: dict = field(default_factory=dict)
    tail_pos: dict = field(default_factory=dict)   # per aircraft type: {'dx', 'dy' (fractions of fin height), 'size'}    # per aircraft type: {'dx', 'dy', 'size'} for the registration
    title_outline: float = 0.0             # outside stroke round the titles, as a fraction of capital height (0 = none)
    title_outline_colour: str = '#FFFFFF'
    # 0 = clean, 0.5 = the paintkit's own dirt layers as supplied, 1 = heavily weathered (paintkit aircraft only)
    weathering: float = 0.5

    def dirt_amount(self) -> float:
        """Weathering slider -> strength of the paintkit's dirt layers: 0 -> none, 0.5 -> as supplied, 1 -> 5x."""
        w = max(0.0, min(1.0, float(self.weathering)))
        return 2 * w if w <= 0.5 else 1 + (w - 0.5) * 8

    def rgb(self, which: str) -> np.ndarray:
        return hex_rgb(getattr(self, which))


@dataclass
class FleetEntry:
    template: str
    reg: str
    variant: str
    selcal: str = ''

    @property
    def label(self) -> str:
        return f'{self.reg} ({self.variant})'


@dataclass
class Project:
    design: Design = field(default_factory=Design)
    fleet: list = field(default_factory=list)          # list[FleetEntry]
    sources: dict = field(default_factory=dict)        # template id -> folder path
    output_dir: str = ''
    models: dict = field(default_factory=dict)         # template id -> folder with the add-on's .obj files and .acf

    def to_json(self) -> str:
        return json.dumps({'design': asdict(self.design), 'fleet': [asdict(f) for f in self.fleet],
                           'sources': self.sources, 'output_dir': self.output_dir,
                           'models': self.models}, indent=2)

    @classmethod
    def from_json(cls, text: str) -> 'Project':
        d = json.loads(text)
        known = set(Design.__dataclass_fields__)
        return cls(design=Design(**{k: v for k, v in d.get('design', {}).items() if k in known}),
                   fleet=[FleetEntry(**f) for f in d.get('fleet', [])],
                   sources=d.get('sources', {}), output_dir=d.get('output_dir', ''),
                   models=d.get('models', {}))


Progress = Callable[[str], None]


def noop(_msg: str) -> None:
    pass


# ---------------------------------------------------------------- fade
def fade_ramp(x: np.ndarray, x0: float, x1: float, design: Design) -> np.ndarray:
    """Smoothstep 0->1 between the template's default fade points, adjusted by the design."""
    span = (x1 - x0)
    mid = (x0 + x1) / 2 + design.fade_shift * span
    half = span / 2 * max(design.fade_length, 0.05)
    t = np.clip((np.asarray(x, np.float32) - (mid - half)) / (2 * half), 0, 1)
    return t * t * (3 - 2 * t)


_STATIONS: dict = {}


def station_map(template_id: str, texture: str, size: int) -> np.ndarray:
    """Distance along the fuselage (metres, nose -> tail) for every texel of one of the aircraft's fuselage textures,
    NaN where the texel isn't on the fuselage. Measured once from the aircraft's 3D model and shipped with the app,
    so fades can follow the real aircraft across texture seams."""
    import cv2
    key = (template_id, texture, size)
    if key not in _STATIONS:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stations', f'{template_id}__{texture}.npz')
        z = np.load(p)['z'].astype(np.float32)
        ok = np.isfinite(z).astype(np.float32)
        zf = np.nan_to_num(z) * ok
        # fill small gaps (window cut-outs, seams) from their surroundings
        bz, bo = cv2.GaussianBlur(zf, (0, 0), 3), cv2.GaussianBlur(ok, (0, 0), 3)
        gap = (ok == 0) & (bo > 0.15)
        zf[gap] = bz[gap] / bo[gap]; ok[gap] = 1
        if z.shape[0] != size:
            zf = cv2.resize(zf, (size, size), interpolation=cv2.INTER_LINEAR)
            ok = cv2.resize(ok, (size, size), interpolation=cv2.INTER_LINEAR)
        with np.errstate(invalid='ignore', divide='ignore'):
            out = np.where(ok > 0.5, zf / np.maximum(ok, 1e-6), np.nan).astype(np.float32)
        while len(_STATIONS) >= 4:
            _STATIONS.pop(next(iter(_STATIONS)))
        _STATIONS[key] = out
    return _STATIONS[key]


def fade_map(template_id: str, texture: str, size: int, zfade, design: 'Design'):
    """(t, ok): the fade 0..1 for every texel by its position along the fuselage, and where that is known."""
    z = station_map(template_id, texture, size)
    ok = np.isfinite(z)
    t = fade_ramp(np.nan_to_num(z), zfade[0], zfade[1], design).astype(np.float32)
    t[~ok] = 0
    return t, ok


# ---------------------------------------------------------------- text
def text_mask(txt: str, font: str, cap: float):
    """Render text white-on-black; cap = capital height in px. Returns (mask L image, font, bbox)."""
    fnt = ImageFont.truetype(font_path(font), size=max(4, int(cap / 0.727)))
    bb = fnt.getbbox(txt)
    im = Image.new('L', (bb[2] - bb[0] + 10, bb[3] - bb[1] + 10), 0)
    ImageDraw.Draw(im).text((5 - bb[0], 5 - bb[1]), txt, font=fnt, fill=255)
    return im, fnt, bb


def fit_title(txt: str, font: str, max_cap: int, max_width: int, scale: float = 1.0):
    """Largest cap height <= max_cap whose ink fits max_width, then scaled by `scale`."""
    cap = max_cap
    while cap > 8:
        tm, fnt, bb = text_mask(txt, font, cap)
        ink = tm.getbbox() or (0, 0, 1, 1)
        if ink[2] - ink[0] <= max_width:
            break
        cap -= 4
    cap = max(8, int(cap * max(0.3, min(1.0, scale))))
    tm, fnt, bb = text_mask(txt, font, cap)
    return tm, fnt, bb, cap


def title_left(txt: str, font: str, x_a: int, x_b: int, win_y: int, max_cap: int, scale: float,
               margin: int = 80, dx: int = 0, dy: int = 0, size: float = 1.0):
    """The left-side title: (mask image, x, y, cap). Fitted between x_a..x_b with capitals centred on the window
    line win_y, then resized by `size` and moved by (dx, dy) texture pixels. None if there is no text."""
    if not (txt or '').strip():
        return None
    tm, fnt, bb, cap = fit_title(txt, font, max_cap, (x_b - x_a) - 2 * margin, scale)
    if abs(size - 1.0) > 1e-3:
        cap = max(8, int(round(cap * size)))
        tm, fnt, bb = text_mask(txt, font, cap)
    ink = tm.getbbox() or (0, 0, 1, 1)
    base_in = 5 - bb[1] + fnt.getmetrics()[0]
    tx = int((x_a + x_b) / 2 - (ink[0] + ink[2]) / 2) + int(dx)
    ty = int(win_y + cap / 2 - base_in) + int(dy)
    return tm, tx, ty, cap


def stroke_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow an anti-aliased text mask outwards by radius px (a round 'outside stroke'); uint8 in, uint8 out."""
    import cv2
    ys, xs = np.nonzero(mask)
    out = np.zeros_like(mask)
    if not len(ys) or radius < 1:
        return out
    y0, y1 = max(0, ys.min() - radius - 2), min(mask.shape[0], ys.max() + radius + 3)
    x0, x1 = max(0, xs.min() - radius - 2), min(mask.shape[1], xs.max() + radius + 3)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    out[y0:y1, x0:x1] = cv2.dilate(np.ascontiguousarray(mask[y0:y1, x0:x1]), k)
    return out


def title_pair(layer: Image.Image, txt: str, font: str, x_a: int, x_b: int, win_y: int, y_sum: int,
               max_cap: int, scale: float, margin: int = 80, dx: int = 0, dy: int = 0, size: float = 1.0) -> int:
    """Draw the left-side title (see title_left) and a 180-degree rotated copy on the other side at y' = y_sum - y,
    so moving the left title moves the right one to the matching place. Returns the cap height used."""
    t = title_left(txt, font, x_a, x_b, win_y, max_cap, scale, margin, dx, dy, size)
    if t is None:
        return 0
    tm, tx, ty, cap = t
    layer.paste(tm, (tx, ty), tm)
    r = tm.rotate(180)
    layer.paste(r, (tx, y_sum - (ty + tm.size[1])), r)
    return cap


def reg_left(reg: str, font: str, cap: int, cx: int, base_y: int, anchor: str = 'base',
             dx: int = 0, dy: int = 0, size: float = 1.0):
    """The left-side registration: (mask, x, y) in texture px, centred on cx with its baseline at base_y
    (or its middle at base_y when anchor == 'centre'), resized by `size` and moved by (dx, dy). None if blank."""
    if not (reg or '').strip():
        return None
    rm, _, _ = text_mask(reg, font, max(6, int(round(cap * size))))
    bb = rm.getbbox()
    if not bb:
        return None
    rm = rm.crop(bb)
    rw, rh = rm.size
    y = base_y - rh // 2 if anchor == 'centre' else base_y - rh
    return rm, cx - rw // 2 + int(dx), y + int(dy)


def reg_pair(layer: Image.Image, reg: str, font: str, cap: int, cx: int, base_y: int, y_sum: int,
             anchor: str = 'base', dx: int = 0, dy: int = 0, size: float = 1.0) -> None:
    """Registration on the left side (see reg_left) and a rotated copy on the other side at y' = y_sum - y."""
    t = reg_left(reg, font, cap, cx, base_y, anchor, dx, dy, size)
    if t is None:
        return
    rm, x, y = t
    layer.paste(rm, (x, y), rm)
    r = rm.rotate(180)
    layer.paste(r, (x, y_sum - (y + rm.size[1])), r)


def blend(out: np.ndarray, mask_img, colour: np.ndarray, shade: Optional[np.ndarray] = None) -> np.ndarray:
    a = (np.asarray(mask_img).astype(np.float32) / 255)[..., None] if not isinstance(mask_img, np.ndarray) else mask_img
    if a.ndim == 2:
        a = a[..., None]
    c = colour[None, None, :] if shade is None else colour[None, None, :] * shade[..., None]
    return out * (1 - a) + c * a


# ---------------------------------------------------------------- fin symbols
def _frame(rle, rte, tle, tte):
    rle, rte, tle, tte = (np.array(p, np.float64) for p in (rle, rte, tle, tte))
    ctr = (rle + rte + tle + tte) / 4
    es = (tle + tte) / 2 - (rle + rte) / 2
    H = np.linalg.norm(es)
    es /= H
    ex = rte - rle
    ex /= np.linalg.norm(ex)
    return ctr, ex, es, H


def fin_frame(quad):
    """(centre, right, up, height) of a fin side: right = root chord direction (towards the trailing edge),
    up = perpendicular to it towards the tip."""
    ctr, ex, es, H = _frame(*quad)
    up = np.array([-ex[1], ex[0]])
    if np.dot(up, es) < 0:
        up = -up
    return ctr, ex, up, H, es


def fin_symbol(shape, quad, clip_ref: np.ndarray, mapping, clip_other: np.ndarray, design: Design,
               mapping_is_mirror: bool = False, pos: Optional[dict] = None, flip_ref: bool = False):
    """Paint the tail symbol on the reference fin side and copy it to the other side.
    quad = (root LE, root TE, tip LE, tip TE) on the reference side.
    mapping(xs, ys) -> (x2, y2) maps reference pixels onto the other side (UV correspondence).
    The reference side is the aircraft's left side: an image is drawn with its right edge towards the
    trailing edge. When the correspondence is a mirror, the image is flipped first so it still reads
    correctly from outside on the other side; for a 180-degree rotation it is copied as is.
    Returns (rgb float HxWx3, alpha float HxW)."""
    Hh, Ww = shape
    rgb = np.zeros((Hh, Ww, 3), np.float32)
    alpha = np.zeros((Hh, Ww), np.float32)
    if design.tail_symbol == 'none':
        return rgb, alpha
    ctr, ex, es, H = _frame(*quad)
    _c, _r, up_, _h, _e = fin_frame(quad)
    pos = pos or {}
    ctr = ctr + (float(pos.get('dx', 0)) * ex + float(pos.get('dy', 0)) * up_) * H   # moved by the user
    size = float(pos.get('size', 1.0))
    ang = np.radians(float(pos.get('angle', 0.0)))           # + = top of the logo leans towards the nose
    ca, sa = np.cos(ang), np.sin(ang)

    def turn(v):
        """Rotate a texture-space direction about the fin normal, as seen on the reference (left) side."""
        a, b = float(v @ ex), float(v @ up_)
        return (a * ca - b * sa) * ex + (a * sa + b * ca) * up_

    def copy_over(src_a, src_rgb, flip_rgb=None, flip_a=None):
        ys, xs = np.nonzero((flip_a if flip_a is not None else src_a) > 0.004)
        y2, x2 = mapping(xs, ys)
        x2 = np.asarray(x2).astype(int); y2 = np.asarray(y2).astype(int)
        ok = (y2 >= 0) & (y2 < Hh) & (x2 >= 0) & (x2 < Ww)
        a_src = flip_a if flip_a is not None else src_a
        c_src = flip_rgb if flip_rgb is not None else src_rgb
        ys, xs, y2, x2 = ys[ok], xs[ok], y2[ok], x2[ok]
        keep = clip_other[y2, x2]
        alpha[y2[keep], x2[keep]] = a_src[ys[keep], xs[keep]]
        rgb[y2[keep], x2[keep]] = c_src[ys[keep], xs[keep]]

    if design.tail_symbol == 'saltire':
        im = Image.new('L', (Ww, Hh), 0)
        d = ImageDraw.Draw(im)
        arm = max(4, int(design.saltire_width * H * size))
        for v in (ex + es, -ex + es):
            v = turn(v / np.linalg.norm(v))
            d.line([tuple(ctr - v * 3 * H), tuple(ctr + v * 3 * H)], fill=255, width=arm)
        im = im.filter(ImageFilter.GaussianBlur(0.8))
        a_ref = (np.asarray(im).astype(np.float32) / 255) * clip_ref
        col = np.broadcast_to(design.rgb('symbol_colour'), (Hh, Ww, 3)).astype(np.float32)
        alpha[:] = a_ref
        rgb[:] = col
        copy_over(a_ref, col)
        return rgb, alpha

    if design.tail_symbol == 'image' and design.tail_image and os.path.exists(design.tail_image):
        logo = Image.open(design.tail_image).convert('RGBA')

        def render(img):
            w, h = img.size
            s = design.tail_image_size * size * H / h           # px per logo px
            up = turn(up_)                                        # 'up' = perpendicular to the root chord, towards the tip
            right = turn(ex)
            c = (-ctr @ right) / s + w / 2
            f_ = (ctr @ up) / s + h / 2
            coeffs = (right[0] / s, right[1] / s, c, -up[0] / s, -up[1] / s, f_)
            return np.asarray(img.transform((Ww, Hh), Image.AFFINE, coeffs, resample=Image.BICUBIC)).astype(np.float32)

        # flip_ref: the reference side's texture is stored mirror-image (the ERJ), so it gets the flipped logo
        mirrored = logo.transpose(Image.FLIP_LEFT_RIGHT)
        ref = render(mirrored if flip_ref else logo)
        a_ref = ref[..., 3] / 255 * clip_ref
        rgb[:] = ref[..., :3]
        alpha[:] = a_ref
        if mapping_is_mirror:
            flipped = render(logo if flip_ref else mirrored)
            copy_over(a_ref, ref[..., :3], flipped[..., :3], flipped[..., 3] / 255 * clip_ref)
        else:
            copy_over(a_ref, ref[..., :3])
        return rgb, alpha
    return rgb, alpha


def apply_symbol(out: np.ndarray, sym, shade: Optional[np.ndarray] = None) -> np.ndarray:
    rgb, a = sym
    if shade is not None:
        rgb = rgb * np.clip(shade, 0.85, 1.05)[..., None]
    return out * (1 - a[..., None]) + rgb * a[..., None]


def mini_saltire(draw: ImageDraw.ImageDraw, box, width: int):
    x0, y0, x1, y1 = box
    draw.line([(x0, y0), (x1, y1)], fill=255, width=width)
    draw.line([(x0, y1), (x1, y0)], fill=255, width=width)


# ---------------------------------------------------------------- PSD compositing
def layer_full(layer, size):
    im = layer.topil()
    if im is None:
        return None
    c = Image.new('RGBA', size, (0, 0, 0, 0))
    c.paste(im.convert('RGBA'), (layer.left, layer.top))
    return np.asarray(c).astype(np.float32) / 255


def composite(psd, replace=None, hide=(), show=()):
    """Flatten a paintkit PSD (top-level layers, normal + multiply), replacing named layers with RGBA float arrays."""
    from psd_tools import PSDImage
    replace = replace or {}
    p = psd if not isinstance(psd, str) else PSDImage.open(psd)
    W, H = p.size
    out = np.zeros((H, W, 3), np.float32)
    for l in p:
        vis = (l.visible or l.name in show) and l.name not in hide
        if not vis:
            continue
        a = replace.get(l.name)
        if a is None:
            a = layer_full(l, (W, H))
            if a is None:
                continue
        rgb, al = a[..., :3], a[..., 3:4] * (l.opacity / 255.0)
        src = out * rgb if 'MULTIPLY' in str(l.blend_mode) else rgb
        out = out * (1 - al) + src * al
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


_digest_cache: dict = {}


def file_digest(path: str) -> str:
    """Content hash (cached by path+mtime+size) so identical files, e.g. left/right wing kits, share one cache entry."""
    import hashlib
    st = os.stat(path)
    k = (os.path.abspath(path), st.st_mtime, st.st_size)
    if k not in _digest_cache:
        h = hashlib.md5()
        with open(path, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b''):
                h.update(chunk)
        _digest_cache[k] = h.hexdigest()
    return _digest_cache[k]


_mask_cache: dict = {}


def paint_mask_labels(path: str, layer: str = 'MASK'):
    """Label the paintable islands of a paintkit (transparent parts of its MASK layer)."""
    key = (os.path.abspath(path), layer, os.path.getmtime(path))
    if key not in _mask_cache:
        from psd_tools import PSDImage
        from scipy import ndimage
        p = PSDImage.open(path)
        L = {l.name: l for l in p}
        m = layer_full(L[layer], p.size)[..., 3] < 0.5
        _mask_cache.clear()
        _mask_cache[key] = ndimage.label(m)[0]
    return _mask_cache[key]


class PaintkitSheet:
    """A paintkit PSD reduced to: below (composite under the paint layer) and an affine 'above' stack,
    so result = (below*(1-a) + paint*a) * M + C. Rendering a new paint layer is then instant.

    Layers named 'dirt' (wherever they sit) are gathered into their own affine pair (Kd, Cd) and applied last, so
    the amount of weathering can be chosen when rendering: 0 = none, 1 = as the paintkit supplies it, 2 = double."""
    _cache: dict = {}

    def __init__(self, path: str, layer: str):
        from psd_tools import PSDImage
        p = PSDImage.open(path)
        W, H = p.size
        self.size = (W, H)
        below = np.zeros((H, W, 3), np.float32)
        M = None; C = None
        Kd = Cd = None
        self.layer_alpha = None
        seen = False
        for l in p:
            if not l.visible:
                continue
            if 'dirt' in l.name.lower() and l.name != layer:
                a = layer_full(l, (W, H))
                if a is None:
                    continue
                rgb, al = a[..., :3], a[..., 3:4] * (l.opacity / 255.0)
                if Kd is None:
                    Kd = np.ones((H, W, 3), np.float32); Cd = np.zeros((H, W, 3), np.float32)
                if 'MULTIPLY' in str(l.blend_mode):
                    k = (1 - al) + rgb * al
                    Kd *= k; Cd *= k
                else:
                    Kd *= (1 - al); Cd = Cd * (1 - al) + rgb * al
                del a
                continue
            if l.name == layer and not seen:
                seen = True
                la = layer_full(l, (W, H))
                self.layer_alpha = la[..., 3] if la is not None else np.ones((H, W), np.float32)
                M = np.ones((H, W, 3), np.float32); C = np.zeros((H, W, 3), np.float32)
                continue
            a = layer_full(l, (W, H))
            if a is None:
                continue
            rgb, al = a[..., :3], a[..., 3:4] * (l.opacity / 255.0)
            mult = 'MULTIPLY' in str(l.blend_mode)
            if not seen:
                src = below * rgb if mult else rgb
                below = below * (1 - al) + src * al
            else:
                if mult:
                    k = (1 - al) + rgb * al
                    M *= k; C *= k
                else:
                    M *= (1 - al); C = C * (1 - al) + rgb * al
            del a
        if not seen:
            raise ValueError(f'Layer {layer!r} not found in {os.path.basename(path)}')
        self.below = below.astype(np.float16)
        self.M = M.astype(np.float16); self.C = C.astype(np.float16)
        self.Kd = Kd.astype(np.float16) if Kd is not None else None
        self.Cd = Cd.astype(np.float16) if Cd is not None and Cd.any() else None

    @classmethod
    def get(cls, path: str, layer: str) -> 'PaintkitSheet':
        key = (file_digest(path), layer)
        if key not in cls._cache:
            while len(cls._cache) >= 4:
                cls._cache.pop(next(iter(cls._cache)))
            cls._cache[key] = cls(path, layer)
        return cls._cache[key]

    def render(self, paint_rgb01: np.ndarray, paint_alpha: Optional[np.ndarray] = None, dirt: float = 1.0) -> np.ndarray:
        a = (self.layer_alpha if paint_alpha is None else paint_alpha)[..., None]
        out = self.below.astype(np.float32) * (1 - a) + paint_rgb01.astype(np.float32) * a
        out = out * self.M + self.C
        if self.Kd is not None and dirt > 0:
            out *= np.clip(1 + (self.Kd.astype(np.float32) - 1) * dirt, 0, 1)
            if self.Cd is not None:
                out += self.Cd.astype(np.float32) * dirt
        return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- DDS
def write_dds(img: Image.Image, path: str, fmt: str = 'DXT5', mipmaps: bool = True) -> None:
    """Write a BC1 (DXT1) / BC3 (DXT5) DDS with a full mip chain, using Pillow's block encoder per level."""
    img = img.convert('RGBA')
    w, h = img.size
    levels = [img]
    if mipmaps:
        while max(w, h) > 1:
            w, h = max(1, w // 2), max(1, h // 2)
            levels.append(levels[-1].resize((w, h), Image.LANCZOS))
    payloads = []
    for lv in levels:
        bio = BytesIO()
        if min(lv.size) < 4:
            lv = lv.resize((max(4, lv.size[0]), max(4, lv.size[1])), Image.LANCZOS)
        lv.save(bio, format='DDS', pixel_format=fmt)
        data = bio.getvalue()
        payloads.append(data[128:])
    w0, h0 = img.size
    block = 8 if fmt == 'DXT1' else 16
    pitch = max(1, (w0 + 3) // 4) * max(1, (h0 + 3) // 4) * block
    DDSD_CAPS, DDSD_HEIGHT, DDSD_WIDTH, DDSD_PIXELFORMAT, DDSD_MIPMAPCOUNT, DDSD_LINEARSIZE = 0x1, 0x2, 0x4, 0x1000, 0x20000, 0x80000
    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE | (DDSD_MIPMAPCOUNT if mipmaps else 0)
    caps = 0x1000 | ((0x8 | 0x400000) if mipmaps else 0)
    pf = struct.pack('<II4sIIIII', 32, 0x4, fmt.encode(), 0, 0, 0, 0, 0)
    header = struct.pack('<4sIIIIIII', b'DDS ', 124, flags, h0, w0, pitch, 0, len(levels)) + b'\0' * 44 + pf + \
        struct.pack('<IIIII', caps, 0, 0, 0, 0)
    with open(path, 'wb') as fh:
        fh.write(header)
        for p_ in payloads:
            fh.write(p_)


def dds_format(path: str) -> str:
    with open(path, 'rb') as fh:
        h = fh.read(128)
    return h[84:88].decode('ascii', 'replace')


def load_rgba(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert('RGBA'))


def save_png(arr: np.ndarray, path: str) -> None:
    Image.fromarray(arr).save(path, compress_level=6)


def preview(arr: np.ndarray, size: int = 768) -> Image.Image:
    im = Image.fromarray(arr[..., :3] if arr.ndim == 3 else arr)
    return im.resize((size, size), Image.BILINEAR) if im.size[0] == im.size[1] else im.resize((size * im.size[0] // im.size[1], size))


# ---------------------------------------------------------------- SELCAL placard (ToLiss 256x256)
def selcal_placard(reg: str, selcal: str, path: str) -> None:
    im = Image.new('RGB', (256, 256), (255, 255, 255))
    d = ImageDraw.Draw(im)
    f = font_path('Inter Bold')
    for t, (y0, y1) in ((reg, (19, 80)), (selcal, (157, 229))):
        if not t:
            continue
        sz = 40
        while True:
            F = ImageFont.truetype(f, sz)
            b = F.getbbox(t)
            if (b[2] - b[0] <= 220 and b[3] - b[1] <= (y1 - y0) * 0.62) or sz < 8:
                break
            sz -= 1
        w_, h_ = b[2] - b[0], b[3] - b[1]
        d.text(((256 - w_) // 2 - b[0], (y0 + y1 - h_) // 2 - b[1]), t, font=F, fill=(0, 0, 0))
    im.save(path)


def safe_name(s: str) -> str:
    return ''.join(c for c in s if c not in '/\\:*?"<>|').strip()
