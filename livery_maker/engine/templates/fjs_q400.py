"""FlyJSim Q4XP (Dash 8 Q400), painted over the FlyJSim house livery (DXT1 .dds textures)."""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from ..common import fade_map, Design, FleetEntry, apply_symbol, fade_ramp, fin_symbol, preview, text_mask, title_pair, write_dds
from .base import Template

S = 4096


class FJSQ400(Template):
    id = 'fjs_q400'
    name = 'FlyJSim Q4XP (Q400)'
    model = 'Q400'
    make = 'FJS'
    source_kind = 'livery'
    source_hint = 'Q4XP FlyJSim house livery folder (contains objects/exterior/Fuse_1.dds)'
    base_note = 'FlyJSim house livery (G-JEDI)'
    required = ['Fuse_1.dds', 'Fuse_2.dds', 'Fuse_3.dds', 'Fuse_4.dds', 'Tail.dds', 'Nacelle_Left.dds',
                'Nacelle_Right.dds', 'WingsInner.dds']
    fingerprints = {'Fuse_1.dds': 'c16e88b4a24c82ece684cd66023c1ece', 'Tail.dds': '1e4ba28e28c697a09624af7967d6f0f9'}
    variants = {'PW150A': {'eng': 'PW150A'}}
    TITLE = {'x_a': 2900, 'x_b': 5600, 'win_y': 880, 'y_sum': 4095, 'max_cap': 560, 'margin': 80,
             'files': ['objects/exterior/Fuse_1.dds', 'objects/exterior/Fuse_2.dds', 'objects/exterior/Fuse_3.dds',
                       'objects/exterior/Fuse_4.dds'], 'band': (250, 1550), 'cols': (0, 10500)}
    model3d = {
        'hint': 'FlyJSim Q4XP aircraft folder on your X-Plane PC (or a copy of its .obj files and Q4XP.acf)',
        'acf': 'Q4XP.acf',
        'objects': {'Fuse_1.obj': 'livery', 'Fuse_2.obj': 'livery', 'Fuse_3.obj': 'livery', 'Fuse_4.obj': 'livery',
                    'Tail.obj': 'livery', 'Nacelle_Left.obj': 'livery', 'Nacelle_Right.obj': 'livery',
                    'WingsInner.obj': 'livery', 'Wing_Left.obj': 'livery', 'Wing_Right.obj': 'livery',
                    'GlassBits.obj': 'glass', 'GEARS.obj': '#7C8086', '_Prop.obj': '#2E3135',
                    '_PropSpinner.obj': '#2E3135'},
        'datarefs': {'sim/flightmodel2/gear/deploy_ratio[0]': 1, 'sim/flightmodel2/gear/deploy_ratio[1]': 1,
                     'sim/flightmodel2/gear/deploy_ratio[2]': 1},
    }

    FADE = (8000, 11800)
    FIN_SPEC = dict(file='objects/exterior/Tail.dds', quad=((870, 2990), (2625, 3040), (1918, 1275), (3303, 1260)))
    REG_SPEC = dict(cap=140, cx=13700, base=880, y_sum=4095, anchor='centre',
                    files=['objects/exterior/Fuse_1.dds', 'objects/exterior/Fuse_2.dds', 'objects/exterior/Fuse_3.dds',
                           'objects/exterior/Fuse_4.dds'], band=(250, 1550), cols=(7000, 16384))
    FADE_Z = (-1.09, 6.20)           # the same fade in metres along the fuselage (from the 3D model)

    def fuselage(self, src, design: Design, entry: FleetEntry, progress):
        strip = np.concatenate([np.asarray(Image.open(self.find(src, f'Fuse_{i}.dds')).convert('RGB')) for i in (1, 2, 3, 4)], axis=1)
        H, WID = strip.shape[:2]
        m = np.zeros((H, WID), np.uint8)
        for x0, y0, x1, y1 in [(3104, 548, 4534, 738), (3901, 3354, 5331, 3538), (1334, 1011, 1773, 1121),
                               (1334, 2972, 1772, 3084), (10088, 600, 11117, 697), (10088, 3398, 11117, 3495)]:
            m[y0 - 14:y1 + 14, x0 - 14:x1 + 14] = 255
        img = strip.copy()
        for x0 in range(0, WID, 4096):
            sl = slice(x0, x0 + 4096)
            if m[:, sl].any():
                img[:, sl] = cv2.inpaint(np.ascontiguousarray(img[:, sl]), np.ascontiguousarray(m[:, sl]), 15, cv2.INPAINT_TELEA)
        del strip

        title = Image.new('L', (WID, H), 0)
        tcap = self.draw_titles(title, design)
        tout = self.title_outline(title, tcap, design)            # uint8 or None
        regm = Image.new('L', (WID, H), 0)
        self.draw_reg(regm, entry.reg, design)
        title = np.asarray(title); regm = np.asarray(regm)

        body, acc = design.rgb('body_colour'), design.rgb('accent_colour')
        # white fuselage skin (not the nose parts or the loose-parts panels) takes the body colour
        L = img.astype(np.float32).mean(2)
        Sat = img.max(2).astype(np.int16) - img.min(2).astype(np.int16)
        wl, _ = ndimage.label((L > 200) & (Sat < 22))
        del L, Sat
        seeds = [(2000, 1500), (2000, 2600), (2000, 2050), (6000, 1500), (6000, 2600), (3500, 880), (3500, 3215)]
        bodymask = np.isin(wl, sorted({int(wl[y, x]) for x, y in seeds} - {0}))
        del wl
        tcol, rcol = design.rgb('title_colour'), design.rgb('reg_colour')
        L8all = img.astype(np.float32).mean(2).clip(0, 255).astype(np.uint8)
        sheets = []
        for i in range(4):
            progress(f'{entry.reg}: fuselage sheet {i + 1}/4')
            f = img[:, i * 4096:(i + 1) * 4096].astype(np.float32)
            r, g, b = f[..., 0], f[..., 1], f[..., 2]
            lum = f.mean(2); sat = f.max(2) - f.min(2)
            design_px = (b > r + 10) | ((r > b + 15) & (g > b + 5) & (r - g < 40))
            paint = (sat >= 14) & design_px
            gx = np.arange(i * 4096, (i + 1) * 4096)
            other = (sat >= 40) & ~design_px
            panel = np.zeros((H, 4096), bool)
            for y0_, y1_ in ((0, 760), (3340, H)):
                cand = np.zeros((H, 4096), bool)
                cand[y0_:y1_] = (lum[y0_:y1_] >= 251) & (sat[y0_:y1_] < 5)
                cand[:, (gx < 6300) | (gx > 9800)] = False
                lb, n = ndimage.label(cand)
                if n:
                    sizes = ndimage.sum(cand, lb, range(1, n + 1)); k = int(np.argmax(sizes)) + 1
                    if sizes[k - 1] > 200000:
                        pm = ndimage.binary_fill_holes(ndimage.binary_closing(lb == k, iterations=6))
                        panel |= ndimage.binary_dilation(pm, iterations=4)
            fade = ((gx >= 4400) & (gx < 11000))[None, :] & ~panel
            paint |= fade & ~other
            paint[:, gx < 4400] = False
            ramp = np.clip((gx - 4400) / 1200.0, 0, 1).astype(np.float32)[None, :]
            if i == 3:
                paint[900:1900, 15300 - 12288:] = False
            a0, a1 = max(0, i * 4096 - 200), min(WID, (i + 1) * 4096 + 200)
            med = cv2.medianBlur(np.ascontiguousarray(L8all[:, a0:a1]), 151)[:, i * 4096 - a0:i * 4096 - a0 + 4096].astype(np.float32)
            s_ = np.clip(lum / np.maximum(med, 1), 0.35, 1.45)
            gm = cv2.morphologyEx(med.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
            s_[cv2.dilate((gm > 4).astype(np.uint8), np.ones((21, 21), np.uint8)) > 0] = 1.0
            t = fade_ramp(gx, *self.FADE, design)
            tz, ok = fade_map(self.id, f'Fuse_{i + 1}', 4096, self.FADE_Z, design)   # by position: no seams
            tt = np.where(ok, tz, t[None, :])[..., None]
            new = np.clip((body * (1 - tt) + acc * tt) * s_[..., None], 0, 255)
            wgt = paint.astype(np.float32) * np.where(gx < 5600, ramp, 1.0)
            bm = bodymask[:, i * 4096:(i + 1) * 4096] & ~panel
            base = np.where(bm[..., None], np.clip((body * (1 - tt) + acc * tt) * s_[..., None], 0, 255), f)
            o = base * (1 - wgt[..., None]) + new * wgt[..., None]
            # shade titles by the surface (panel lines, grime), not by the colour under them
            sh = np.where(bm | (wgt > 0), np.clip(s_, 0.6, 1.02), np.clip(o.mean(2) / 250.0, 0.6, 1.02))[..., None]
            if tout is not None:
                ol = tout[:, i * 4096:(i + 1) * 4096].astype(np.float32)[..., None] / 255
                o = o * (1 - ol) + design.rgb('title_outline_colour') * sh * ol
            tl = title[:, i * 4096:(i + 1) * 4096].astype(np.float32)[..., None] / 255
            o = o * (1 - tl) + (tcol * sh) * tl
            rl = regm[:, i * 4096:(i + 1) * 4096].astype(np.float32)[..., None] / 255
            o = o * (1 - rl) + rcol * rl
            sheets.append(np.clip(o + .5, 0, 255).astype(np.uint8))
        return sheets

    def tail(self, path, design: Design):
        f = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
        r, g, b = f[..., 0], f[..., 1], f[..., 2]
        lum = f.mean(2); sat = f.max(2) - f.min(2)
        acc = design.rgb('accent_colour')
        blue = (b > r + 40) & (b > 90)
        white = (lum > 200) & (sat < 30)
        lab, _ = ndimage.label(ndimage.binary_opening(ndimage.binary_fill_holes(blue | white), iterations=3))
        FA, FB, RUD = lab[1367, 1102], lab[2230, 2219], lab[1123, 3701]
        pieces = ndimage.binary_dilation(ndimage.binary_fill_holes(blue), iterations=2)
        whitish = (lum > 170) & (sat < 60)
        logo = np.isin(lab, [FA, FB, RUD]) & whitish
        logo = ndimage.binary_dilation(logo, iterations=5) & np.isin(lab, [FA, FB, RUD])
        ref = np.median(lum[blue])
        s = np.clip(lum / ref, 0.4, 1.35)
        s[logo] = 1.0
        paint = (pieces & (blue | logo | (sat > 25))) | logo
        paint &= ~((lum > 170) & (sat < 25) & ~logo)
        out = f.copy()
        out[paint] = acc * s[paint][:, None]
        quad = ((870, 2990), (2625, 3040), (1918, 1275), (3303, 1260))
        sym = fin_symbol((S, S), quad, lab == FB, lambda x, y: (3599 - y, 3322 - x), lab == FA, design,
                         mapping_is_mirror=True, pos=(design.tail_pos or {}).get(self.id))   # checked on the 3D model
        out = apply_symbol(out, sym, s)
        return np.clip(out + .5, 0, 255).astype(np.uint8)

    def recolour(self, path, design: Design, logo_box=None, sigma=120):
        img = np.asarray(Image.open(path).convert('RGB')).copy()
        if logo_box:
            x0, y0, x1, y1 = logo_box
            m = np.zeros(img.shape[:2], np.uint8); m[y0 - 25:y1 + 25, x0 - 25:x1 + 25] = 255
            img = cv2.inpaint(img, m, 15, cv2.INPAINT_TELEA)
        f = img.astype(np.float32); r, g, b = f[..., 0], f[..., 1], f[..., 2]
        lum = f.mean(2); sat = f.max(2) - f.min(2)
        dsg = (sat >= 14) & ((b > r + 10) | ((r > b + 15) & (g > b + 5) & (r - g < 40) & (lum > 200)))
        white = (lum > 215) & (sat < 14)
        t_raw = np.where(dsg, np.clip((b - r) / 165.0, 0, 1), 0).astype(np.float32)
        mk = (dsg | white).astype(np.float32)
        t = np.clip(cv2.GaussianBlur(t_raw * mk, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(mk, (0, 0), sigma), 1e-3), 0, 1)
        t = t * t * (3 - 2 * t)
        med = cv2.medianBlur(np.clip(lum, 0, 255).astype(np.uint8), 151).astype(np.float32)
        s = np.clip(lum / np.maximum(med, 1), 0.35, 1.45)
        gm = cv2.morphologyEx(med.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
        s[cv2.dilate((gm > 4).astype(np.uint8), np.ones((21, 21), np.uint8)) > 0] = 1.0
        apply = dsg | (white & (t > 0.01))
        far = design.rgb('accent_colour' if design.engine_colour == 'accent' else 'body_colour')
        T = design.rgb('body_colour') * (1 - t[..., None]) + far * t[..., None]
        out = np.where(apply[..., None], np.clip(T * s[..., None], 0, 255), f)
        return np.clip(out + .5, 0, 255).astype(np.uint8)

    def build(self, src, design, entry, out_root, progress=lambda m: None):
        folder = os.path.join(out_root, self.folder_name(design, entry))
        ext = os.path.join(folder, 'objects', 'exterior')
        os.makedirs(ext, exist_ok=True)
        prev = {}
        sheets = self.fuselage(src, design, entry, progress)
        for i, sh in enumerate(sheets):
            write_dds(Image.fromarray(sh), os.path.join(ext, f'Fuse_{i + 1}.dds'), 'DXT1')
        prev['Fuselage (rear)'] = preview(sheets[2])
        prev['Fuselage (front)'] = preview(sheets[0])
        del sheets
        progress(f'{entry.reg}: tail')
        tail = self.tail(self.find(src, 'Tail.dds'), design)
        write_dds(Image.fromarray(tail), os.path.join(ext, 'Tail.dds'), 'DXT1')
        prev['Tail'] = preview(tail)
        progress(f'{entry.reg}: nacelles and wing root')
        for name, box in (('Nacelle_Left', (401, 2296, 735, 2602)), ('Nacelle_Right', (400, 2296, 712, 2602)), ('WingsInner', None)):
            o = self.recolour(self.find(src, f'{name}.dds'), design, box)
            write_dds(Image.fromarray(o), os.path.join(ext, f'{name}.dds'), 'DXT1')
            if name == 'Nacelle_Left':
                prev['Nacelles'] = preview(o)
        sel = entry.selcal if '-' in entry.selcal or not entry.selcal else entry.selcal[:2] + '-' + entry.selcal[2:]
        with open(os.path.join(folder, 'LiveryStat.prf'), 'w') as fh:
            fh.write(f'AIRLINE|s|{design.airline}\nOPERATOR|s|{design.airline}\nREGISTRATION|s|{entry.reg}\n'
                     f'SELCAL|s|{sel}\nliveryCreator|s|Livery Maker\n')
        return prev
