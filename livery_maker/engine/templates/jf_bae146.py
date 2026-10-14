"""JustFlight / Thranda BAe 146 Professional (-100, -200, -300), painted over the QantasLink -200 livery.

The three lengths share one set of textures, but each length uses different parts of them (the fuselage plugs), so
titles, registration and tail logo can't be placed by texture position. Instead they are drawn on a flat side view of
the aircraft in metres and projected onto every texel through position maps measured from the 3D model
(engine/stations/jf_bae146*.npz)."""
from __future__ import annotations

import os
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..common import Design, FleetEntry, fade_ramp, fin_symbol, font_path, preview, title_left, reg_left, write_dds
from .base import Template

S = 4096
STATIONS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stations')

# Side view ("canvas") used for titles and registration: u = (z - Z0) * PPM, v = (YTOP - y) * PPM.
# Metres: z runs nose -> tail, y up. The titles use the -100's nose as reference, so they keep the same place
# relative to the nose on every length; the registration (on the rear fuselage) is measured from the tail.
PPM = 400
Z0, YTOP = -14.0, 2.4
CANVAS_W, CANVAS_H = 12800, 2080
NOSE = {'100': -11.12, '200': -12.24, '300': -13.50}
# Fin canvas: u = (z - FZ0) * FPPM, v = (FYTOP - y) * FPPM
FPPM, FZ0, FYTOP = 300, 9.0, 6.3
FIN_W, FIN_H = int((18.0 - FZ0) * FPPM), int((FYTOP - 1.6) * FPPM)
EDITOR_SCALE = 0.2           # the side-view pictures for the Positions tab are kept at 80 px per metre


def _cu(z):
    return (z - Z0) * PPM


def _cv(y):
    return (YTOP - y) * PPM


def _fc(z, y):
    return ((z - FZ0) * FPPM, (FYTOP - y) * FPPM)


_MAPS: dict = {}


def load_map(name: str):
    """Position map for one texture at full size: (x, y, z, nx, ok) float32 arrays (ok bool)."""
    if name in _MAPS:
        return _MAPS[name]
    d = np.load(os.path.join(STATIONS, f'{name}.npz'))
    p = d['p']
    ok = p[..., 0] != -32768
    okf = ok.astype(np.float32)
    ch = [np.where(ok, p[..., i].astype(np.float32) / 1000, 0) for i in range(3)]
    ch.append(d['nx'].astype(np.float32) / 127)
    # fill one-texel gaps at island edges so painting reaches the seams
    bo = cv2.GaussianBlur(okf, (0, 0), 1.0)
    gap = (~ok) & (bo > 0.2)
    for i in range(4):
        b = cv2.GaussianBlur(ch[i] * okf, (0, 0), 1.0)
        ch[i][gap] = b[gap] / bo[gap]
    okf[gap] = 1
    up = cv2.resize(okf, (S, S), interpolation=cv2.INTER_LINEAR)
    out = []
    for c in ch:
        v = cv2.resize(c * okf, (S, S), interpolation=cv2.INTER_LINEAR)
        out.append((v / np.maximum(up, 1e-4)).astype(np.float32))
    res = (*out, up > 0.5)
    while len(_MAPS) >= 2:
        _MAPS.pop(next(iter(_MAPS)))
    _MAPS[name] = res
    return res


def project(item: np.ndarray, x0: float, y0: float, u: np.ndarray, v: np.ndarray, right: np.ndarray,
            use: np.ndarray, mirror: bool = True) -> np.ndarray:
    """Sample a small canvas image `item` (float32, HxW or HxWxC) placed at canvas (x0, y0) for every texel at canvas
    position (u, v). On the right side (`right`) the item is mirrored about its own centre so it reads correctly."""
    h, w = item.shape[:2]
    lu = u - x0
    if mirror:
        lu = np.where(right, (w - 1) - lu, lu)
    lv = (v - y0).astype(np.float32)
    out = cv2.remap(item, lu.astype(np.float32), lv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m = use.astype(np.float32)
    return out * (m if out.ndim == 2 else m[..., None])


def classes(f: np.ndarray):
    """Colour classes of the QantasLink paint: red, gold, bright neutral, mid neutral."""
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    lum = f.mean(2)
    red = (r - np.maximum(g, b)) > 40
    gold = (r > g) & (g > b) & (r - b > 50) & (g - b > 25) & ~red
    neu = ~red & ~gold
    bright = neu & (lum >= 200)
    mid = neu & (lum < 200) & (lum >= 110)
    return red, gold, bright, mid, lum


def flat_shade(f: np.ndarray, lum=None, cls=None) -> np.ndarray:
    """Surface shading (panel lines, grime, baked light) separated from the old paint colours: lum / the local
    average of the same colour class. Class borders take the shading of their surroundings, so no outline of the old
    design is left behind."""
    if cls is None:
        red, gold, bright, mid, lum = classes(f)
        cls = (red, gold, bright, mid)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = [cv2.morphologyEx(c.astype(np.uint8), cv2.MORPH_OPEN, k) for c in cls]
    F = np.zeros(lum.shape, np.float32)
    best = np.zeros(lum.shape, np.float32)
    lab = np.zeros(lum.shape, np.uint8)
    for i, o in enumerate(opened, 1):
        m = o.astype(np.float32)
        den = cv2.GaussianBlur(m, (0, 0), 25)
        Fc = cv2.GaussianBlur(lum * m, (0, 0), 25) / np.maximum(den, 1e-3)
        own = o > 0
        take = own | ((lab == 0) & (den > best))
        F = np.where(take, Fc, F)
        best = np.where(take & ~own, den, np.where(own, 2.0, best))
        lab[own] = i
    s = np.clip(lum / np.maximum(F, 1), 0.35, 1.3).astype(np.float32)
    edge = cv2.morphologyEx(lab, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    edge &= cv2.dilate((lab > 0).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    bd = cv2.dilate(edge.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
    w = (~bd).astype(np.float32)
    fill = cv2.GaussianBlur(s * w, (0, 0), 6) / np.maximum(cv2.GaussianBlur(w, (0, 0), 6), 1e-3)
    s[bd] = np.clip(fill[bd], 0.35, 1.3)
    return s


def splat(img: np.ndarray, cu: np.ndarray, cv_: np.ndarray, sel: np.ndarray, acc, cnt):
    """Add the selected texels' colours to a side-view accumulator at canvas positions (cu, cv_)."""
    H, W = cnt.shape
    xi = np.round(cu[sel]).astype(np.int64); yi = np.round(cv_[sel]).astype(np.int64)
    good = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    idx = yi[good] * W + xi[good]
    col = img[sel][good].astype(np.float64)
    for c in range(3):
        acc[..., c] += np.bincount(idx, col[:, c], minlength=H * W).reshape(H, W)
    cnt += np.bincount(idx, minlength=H * W).reshape(H, W)


def finish_splat(acc, cnt, bg=(43, 45, 49)) -> Image.Image:
    have = (cnt > 0).astype(np.float32)
    col = acc / np.maximum(cnt, 1)[..., None]
    b = np.dstack([cv2.GaussianBlur((col[..., c] * have).astype(np.float32), (0, 0), 1.2) for c in range(3)])
    bo = cv2.GaussianBlur(have, (0, 0), 1.2)
    fill = b / np.maximum(bo, 1e-3)[..., None]
    out = np.where(have[..., None] > 0, col, np.where(bo[..., None] > 0.25, fill, np.array(bg, np.float32)))
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


class JFBAe146(Template):
    id = 'jf_bae146'
    name = 'JustFlight BAe 146 (100/200/300)'
    model = 'B462'
    make = 'BAe'
    source_kind = 'livery'
    source_hint = 'BAe 146 QantasLink livery folder (contains objects/E1.dds … E7.dds)'
    base_note = 'QantasLink BAe 146-200 livery (VH-NJJ)'
    required = ['E1.dds', 'E2.dds', 'E3.dds', 'E4.dds', 'E5.dds', 'E6.dds', 'E7.dds']
    fingerprints = {'E1.dds': 'b28483b286984db396b1859ab7160678', 'E7.dds': '284ddc054d2d1149ff329079ba2817e0'}
    variants = {'100': {'model': 'B461', 'acf': 'JF_BAe_146_100.acf'},
                '200': {'model': 'B462', 'acf': 'JF_BAe_146_200.acf'},
                '300': {'model': 'B463', 'acf': 'JF_BAe_146_300.acf'}}
    # titles: between the forward door and the wing on the -100, capitals centred on the window line
    TITLE = {'x_a': int(_cu(NOSE['100'] + 5.8)), 'x_b': int(_cu(NOSE['100'] + 14.0)), 'win_y': int(_cv(0.35)),
             'y_sum': 0, 'max_cap': 300, 'margin': 120, 'canvas': '_editor_title.png',
             'band': (0, CANVAS_H), 'cols': (1000, 12400)}
    REG_SPEC = dict(cap=120, cx=int(_cu(7.74)), base=int(_cv(-0.47)), y_sum=0, anchor='centre',
                    canvas='_editor_reg.png', band=(0, CANVAS_H), cols=(1000, 12400))
    FIN_SPEC = dict(file='_editor_fin.png', quad=(_fc(9.74, 2.0), _fc(14.22, 2.0), _fc(13.36, 5.5), _fc(16.85, 5.5)))
    FADE_Z = (3.5, 9.5)
    model3d = {
        'hint': 'JustFlight BAe 146 aircraft folder on your X-Plane PC (or a copy of its objects folder and the three .acf files)',
        'acf': 'JF_BAe_146_300.acf',
        'objects': {'E1.obj': 'livery', 'E2.obj': 'livery', 'E3.obj': 'livery', 'E4.obj': 'livery',
                    'E5.obj': 'livery', 'E6.obj': 'livery', 'E7.obj': 'livery', 'E8.obj': 'livery',
                    'RegNum.obj': 'livery', 'Glass.obj': 'glass', 'E9.obj': 'glass'},
        'datarefs': {'thranda/aircraft/Variant': 0, 'thranda/view_is_internal': 0,
                     'thranda/anim/noseGearDeploy': 1, 'thranda/anim/leftGearDeploy': 1,
                     'thranda/anim/rightGearDeploy': 1},
        'variant_datarefs': {'100': {'sim/aircraft/parts/acf_gear_znodef[0]': -9.47},
                             '200': {'sim/aircraft/parts/acf_gear_znodef[0]': -10.59},
                             '300': {'sim/aircraft/parts/acf_gear_znodef[0]': -11.85}},
    }

    # old QantasLink markings: (texture, box) where the ink is cleaned off
    OLD_MARKS = {'E1': [(1940, 1775, 2955, 1885), (1960, 3365, 2960, 3478)],
                 'E2': [(0, 1780, 680, 1885), (0, 3368, 690, 3475)],
                 'E3': [(170, 1780, 690, 1905), (170, 3318, 690, 3470)]}
    PLATE = (392, 3345, 642, 3427)          # the small registration plate on E1 (RegNum)

    # ---------------- geometry per length
    def length(self, entry: FleetEntry) -> str:
        return entry.variant if entry.variant in NOSE else '200'

    def map_name(self, tex: str, var: str) -> str:
        return f'jf_bae146_{var}__{tex}' if tex in ('E1', 'E2') else f'jf_bae146__{tex}'

    # ---------------- painting
    def clean(self, f: np.ndarray, tex: str) -> np.ndarray:
        boxes = self.OLD_MARKS.get(tex, [])
        if not boxes:
            return f
        img = f.astype(np.uint8)
        lum = f.mean(2); sat = f.max(2) - f.min(2)
        for x0, y0, x1, y1 in boxes:
            X0, Y0, X1, Y1 = max(0, x0 - 40), max(0, y0 - 40), min(S, x1 + 40), min(S, y1 + 40)
            m = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
            ink = (lum[y0:y1, x0:x1] < 175) | (sat[y0:y1, x0:x1] > 45)
            m[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = ink.astype(np.uint8) * 255
            m = cv2.dilate(m, np.ones((9, 9), np.uint8))
            img[Y0:Y1, X0:X1] = cv2.inpaint(np.ascontiguousarray(img[Y0:Y1, X0:X1]), m, 9, cv2.INPAINT_TELEA)
        return img.astype(np.float32)

    def items(self, design: Design, entry: FleetEntry, editor: bool):
        """Title (with outline) and registration as small RGBA canvas images: [(rgba float, x, y, kind)]."""
        out = []
        t = None if editor else title_left(design.title_text, design.title_font, **self.title_args(design))
        if t:
            mask, tx, ty, cap = t
            r = max(1, int(round(cap * design.title_outline))) if design.title_outline > 0 else 0
            m = np.asarray(mask).astype(np.float32) / 255
            if r:
                m = np.pad(m, r); tx -= r; ty -= r
                from ..common import stroke_mask
                ring = stroke_mask((m * 255).astype(np.uint8), r).astype(np.float32) / 255
            else:
                ring = np.zeros_like(m)
            col = design.rgb('title_colour'); oc = design.rgb('title_outline_colour')
            a = np.maximum(m, ring)
            rgb = (col[None, None] * m[..., None] + oc[None, None] * (ring * (1 - m))[..., None]) / np.maximum(a, 1e-4)[..., None]
            out.append((np.dstack([rgb, a]).astype(np.float32), tx, ty, 'title'))
        if not editor:
            rg = reg_left(entry.reg, design.reg_font, **self.reg_args(design))
            if rg:
                mask, x, y = rg
                m = np.asarray(mask).astype(np.float32) / 255
                rgb = np.broadcast_to(design.rgb('reg_colour'), m.shape + (3,))
                out.append((np.dstack([rgb, m]).astype(np.float32), x, y, 'reg'))
        return out

    def paint_fuselage(self, f, tex, var, design, entry, items, editor_acc):
        x, y, z, nx, ok = load_map(self.map_name(tex, var))
        f = self.clean(f, tex)
        red, gold, bright, mid, lum = classes(f)
        b = f[..., 2]; r = f[..., 0]
        belly = mid & (b > r + 2) & (lum > 140)
        belly = cv2.morphologyEx(belly.astype(np.uint8), cv2.MORPH_OPEN, np.ones((15, 15), np.uint8)) > 0
        s = flat_shade(f, lum, (red, gold, (bright | mid) & ~belly, belly))
        geo = ok & (np.abs(x) < 1.95)
        paint = (red | gold | bright | mid)
        paint = cv2.morphologyEx(paint.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
        paint &= geo
        t = fade_ramp(z, *self.FADE_Z, design).astype(np.float32)[..., None]
        C = design.rgb('body_colour') * (1 - t) + design.rgb('accent_colour') * t
        out = np.where(paint[..., None], np.clip(C * s[..., None], 0, 255), f)
        # titles and registration, projected from the side view
        side = geo & (np.abs(nx) > 0.3)
        right = nx > 0
        cvv = _cv(y)
        shade = np.clip(s, 0.6, 1.02)[..., None]
        for item, ix, iy, kind in items:
            zz = z + (NOSE['100'] - NOSE[var]) if kind == 'title' else z
            pr = project(item, ix, iy, _cu(zz), cvv, right, side)
            a = pr[..., 3:4]
            col = pr[..., :3] * (shade if kind == 'title' else 1.0)
            out = out * (1 - a) + col * a
        if editor_acc is not None:
            sel = ok & (nx < -0.3) & (np.abs(x) < 1.95)
            for key, shift in (('title', NOSE['100'] - NOSE[var]), ('reg', 0.0)):
                acc, cnt = editor_acc[key]
                splat(out, _cu(z + shift) * EDITOR_SCALE, cvv * EDITOR_SCALE, sel, acc, cnt)
        return np.clip(out + .5, 0, 255).astype(np.uint8)

    def paint_tail(self, f, design, entry, editor):
        x, y, z, nx, ok = load_map(self.map_name('E7', ''))
        red, gold, bright, mid, lum = classes(f)
        s = flat_shade(f, lum, (red, gold, bright, mid))
        fin = ok & (np.abs(x) < 0.6) & (y > 1.95)
        body = ok & (np.abs(x) < 1.95) & ~fin                  # fin root fairing and the top of the rear fuselage
        paint = (red | gold | bright | mid)
        paint = cv2.morphologyEx(paint.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
        paint &= fin | body
        if design.tailplane_colour == 'original':           # the T-tail's tailplane (and its fairing) stay as they are
            paint &= ~(fin & (y > 5.4) & (np.abs(nx) < 0.5))
        t = np.where(fin, 1.0, fade_ramp(z, *self.FADE_Z, design)).astype(np.float32)[..., None]
        C = design.rgb('body_colour') * (1 - t) + design.rgb('accent_colour') * t
        out = np.where(paint[..., None], np.clip(C * s[..., None], 0, 255), f)
        fu, fv = _fc(z, y)
        info = None
        if editor:
            sel = fin & (nx < -0.6)
            a_ = np.zeros((FIN_H, FIN_W, 3)); c_ = np.zeros((FIN_H, FIN_W))
            splat(out, fu, fv, sel, a_, c_)
            info = finish_splat(a_, c_)
        if design.tail_symbol != 'none':
            pos = (design.tail_pos or {}).get(self.id)
            ones = np.ones((FIN_H, FIN_W), np.float32)
            none = np.zeros((FIN_H, FIN_W), bool)
            nomap = lambda xs, ys: (np.full(len(ys), -1), np.full(len(xs), -1))
            sides = fin & (np.abs(nx) > 0.6)
            for flip, which in ((False, nx <= 0), (True, nx > 0)):
                rgb, al = fin_symbol((FIN_H, FIN_W), self.FIN_SPEC['quad'], ones, nomap, none, design,
                                     pos=pos, flip_ref=flip)
                item = np.dstack([rgb, al]).astype(np.float32)
                pr = project(item, 0, 0, fu, fv, which, sides & which, mirror=False)
                a = pr[..., 3:4]
                out = out * (1 - a) + pr[..., :3] * np.clip(s, 0.85, 1.05)[..., None] * a
        return np.clip(out + .5, 0, 255).astype(np.uint8), info

    def paint_engines(self, f, design):
        x, y, z, nx, ok = load_map(self.map_name('E6', ''))
        red, gold, bright, mid, lum = classes(f)
        s = flat_shade(f, lum, (red, gold, bright | mid))
        paint = cv2.morphologyEx((bright | mid).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
        paint &= ok & (np.abs(x) > 2.0) & ~red
        col = design.rgb('accent_colour' if design.engine_colour == 'accent' else 'body_colour')
        out = np.where(paint[..., None], np.clip(col * s[..., None], 0, 255), f)
        return np.clip(out + .5, 0, 255).astype(np.uint8)

    def plate(self, img: np.ndarray, reg: str):
        """Registration on the small plate (RegNum object)."""
        x0, y0, x1, y1 = self.PLATE
        bgc = np.median(img[y0 + 4:y1 - 4, x0 + 4:x0 + 20].reshape(-1, 3), 0)
        img[y0:y1, x0:x1] = bgc.astype(np.uint8)
        if not reg:
            return
        im = Image.fromarray(img[y0:y1, x0:x1])
        d = ImageDraw.Draw(im)
        sz = 60
        while sz > 8:
            F = ImageFont.truetype(font_path('Inter Bold'), sz)
            bb = F.getbbox(reg)
            if bb[2] - bb[0] <= (x1 - x0) - 20 and bb[3] - bb[1] <= (y1 - y0) * 0.6:
                break
            sz -= 2
        d.text(((x1 - x0 - (bb[2] - bb[0])) // 2 - bb[0], (y1 - y0 - (bb[3] - bb[1])) // 2 - bb[1]), reg,
               font=F, fill=(118, 118, 124))
        img[y0:y1, x0:x1] = np.asarray(im)

    def build(self, src, design, entry, out_root, progress=lambda m: None):
        folder = os.path.join(out_root, self.folder_name(design, entry))
        obj = os.path.join(folder, 'objects')
        os.makedirs(obj, exist_ok=True)
        var = self.length(entry)
        editor = bool(getattr(design, '_editor_blank', False))
        items = self.items(design, entry, editor)
        prev = {}
        ed = None
        if editor:
            Wc, Hc = int(CANVAS_W * EDITOR_SCALE), int(CANVAS_H * EDITOR_SCALE)
            ed = {k: (np.zeros((Hc, Wc, 3)), np.zeros((Hc, Wc))) for k in ('title', 'reg')}
        for tex in ('E1', 'E2', 'E3'):
            progress(f'{entry.reg}: fuselage {tex}')
            f = np.asarray(Image.open(self.find(src, f'{tex}.dds')).convert('RGB')).astype(np.float32)
            o = self.paint_fuselage(f, tex, var, design, entry, items, ed)
            del f
            if tex == 'E1':
                self.plate(o, '' if editor else entry.reg)
            write_dds(Image.fromarray(o), os.path.join(obj, f'{tex}.dds'), 'DXT1')
            prev[{'E1': 'Fuselage (front)', 'E2': 'Fuselage (middle)', 'E3': 'Fuselage (rear)'}[tex]] = preview(o)
            del o
        progress(f'{entry.reg}: tail')
        f = np.asarray(Image.open(self.find(src, 'E7.dds')).convert('RGB')).astype(np.float32)
        o, fin_img = self.paint_tail(f, design, entry, editor)
        write_dds(Image.fromarray(o), os.path.join(obj, 'E7.dds'), 'DXT1')
        prev['Tail'] = preview(o)
        progress(f'{entry.reg}: engines')
        f = np.asarray(Image.open(self.find(src, 'E6.dds')).convert('RGB')).astype(np.float32)
        o = self.paint_engines(f, design)
        write_dds(Image.fromarray(o), os.path.join(obj, 'E6.dds'), 'DXT1')
        prev['Engines'] = preview(o)
        del f, o
        for name in ('E4.dds', 'E5.dds'):
            p = self.find(src, name)
            if p:
                shutil.copyfile(p, os.path.join(obj, name))
        if editor:
            x0, x1 = self.TITLE['cols']
            for key, fn in (('title', self.TITLE['canvas']), ('reg', self.REG_SPEC['canvas'])):
                im = finish_splat(*ed[key])
                im.crop((int(x0 * EDITOR_SCALE), 0, int(x1 * EDITOR_SCALE), im.size[1])).save(os.path.join(folder, fn))
            if fin_img is not None:
                fin_img.save(os.path.join(folder, self.FIN_SPEC['file']))
        return prev

