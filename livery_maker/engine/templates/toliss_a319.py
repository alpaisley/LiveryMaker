"""ToLiss A319 (paintkit V2.1)."""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from ..common import (fade_map, Design, FleetEntry, PaintkitSheet, apply_symbol, fade_ramp, fin_symbol, paint_mask_labels,
                      preview, reg_pair, selcal_placard, title_pair)
from .base import Template

S = 4096


class ToLissA319(Template):
    id = 'toliss_a319'
    name = 'ToLiss A319'
    model = 'A319'
    make = 'Toliss'
    source_kind = 'paintkit'
    source_hint = 'ToLiss A319 paintkit folder (contains fuselage319.psd)'
    required = ['fuselage319.psd', 'engines.psd', 'wingL.psd', 'wingR.psd']
    variants = {'CFM': {'eng': 'CFM'}, 'IAE': {'eng': 'IAE'}}
    TITLE = {'x_a': 1065, 'x_b': 2184, 'win_y': 1092, 'y_sum': 1523, 'max_cap': 156, 'margin': 40,
             'files': ['objects/fuselage319.png'], 'band': (680, 1500), 'cols': (200, 3150)}
    model3d = {
        'hint': 'ToLiss A319 aircraft folder on your X-Plane PC (or a copy of its .obj files and a319.acf)',
        'acf': 'a319.acf',
        'objects': {'fuselage319.obj': 'livery', 'wingL.obj': 'livery', 'wingR.obj': 'livery', 'engines.obj': 'livery',
                    'windows.obj': 'glass', 'gear.obj': '#7C8086', 'blades.obj': '#3A3D42'},
        'datarefs': {'1-sim/fus': 1, 'anim/SHARK': 1},
        'variant_datarefs': {'CFM': {'anim/CFM': 1}, 'IAE': {'anim/IAE': 1}},
    }

    # texture knowledge (paintkit V2.1)
    SEED_AFT, SEED_FIN = (1083, 3022), (3686, 2938)
    FIN_QUAD = ((3082, 3253), (3848, 3253), (3787, 2435), (4057, 2435))
    FIN_SPEC = dict(file='objects/fuselage319.png', quad=FIN_QUAD)
    FIN_MIRROR = (7038, 838)                 # other side: x' = A - x, y' = y + B
    FADE = (250, 1500)
    FADE_Z = (13.32, 21.24)          # the same fade in metres along the fuselage (from the 3D model)
    WIN_L, WIN_SUM = 1092, 1092 + 431
    TITLE_X = (1065, 2184)
    REG = dict(cx=2165, base=3001, y_sum=5544, cap=46)
    REG_SPEC = dict(REG, files=['objects/fuselage319.png'], band=(2700, 3330), cols=(0, 3000))
    TITLE_MAX_CAP = 156

    def fuselage(self, kit, design: Design, entry: FleetEntry, progress):
        path = os.path.join(kit, 'fuselage319.psd')
        lab = paint_mask_labels(path)
        aft_id = lab[self.SEED_AFT[1], self.SEED_AFT[0]]
        fin_id = lab[self.SEED_FIN[1], self.SEED_FIN[0]]
        body, acc = design.rgb('body_colour'), design.rgb('accent_colour')

        yy, xx = np.mgrid[0:S, 0:S]
        t = fade_ramp(xx[0], *self.FADE, design)[None, :]
        a = np.zeros((S, S), np.float32)
        aft = ndimage.binary_dilation(lab == aft_id, iterations=16)
        a = np.where(aft, np.broadcast_to(t, (S, S)), 0).astype(np.float32)
        tz, ok = fade_map(self.id, 'fuselage319', S, self.FADE_Z, design)
        a = np.where(ok, tz, a)                                    # by position along the fuselage: no seams
        a[(yy >= 2400) & (xx >= 2850)] = 1.0                       # fin + tail cone parts
        del yy, xx
        base = body[None, None, :] * (1 - a[..., None]) + acc[None, None, :] * a[..., None]

        # tail symbol
        A, B = self.FIN_MIRROR
        clip_ref = ndimage.binary_dilation(lab == fin_id, iterations=16)
        clip_other = np.zeros_like(clip_ref)
        ys, xs = np.nonzero(clip_ref)
        ok = (ys + B < S) & (A - xs >= 0) & (A - xs < S)
        clip_other[ys[ok] + B, A - xs[ok]] = True
        sym = fin_symbol((S, S), self.FIN_QUAD, clip_ref, lambda x, y: (y + B, A - x), clip_other, design,
                         mapping_is_mirror=True, pos=(design.tail_pos or {}).get(self.id))
        base = apply_symbol(base, sym)

        # titles + registration
        layer = Image.new('L', (S, S), 0)
        cap = self.draw_titles(layer, design)
        ol = self.title_outline(layer, cap, design)
        if ol is not None:
            ol = ol.astype(np.float32)[..., None] / 255
            base = base * (1 - ol) + design.rgb('title_outline_colour') * ol
        tl = np.asarray(layer).astype(np.float32)[..., None] / 255
        base = base * (1 - tl) + design.rgb('title_colour') * tl
        layer = Image.new('L', (S, S), 0)
        r = self.REG
        self.draw_reg(layer, entry.reg, design)
        rl = np.asarray(layer).astype(np.float32)[..., None] / 255
        base = base * (1 - rl) + design.rgb('reg_colour') * rl

        progress('Compositing fuselage')
        return PaintkitSheet.get(path, 'PAINT HERE').render(np.clip(base, 0, 255) / 255, np.ones((S, S), np.float32), dirt=design.dirt_amount())

    def engines(self, kit, design: Design):
        col = design.rgb('accent_colour' if design.engine_colour == 'accent' else 'body_colour') / 255
        p = np.zeros((S, S, 3), np.float32); p[...] = col
        return PaintkitSheet.get(os.path.join(kit, 'engines.psd'), 'PAINT HERE').render(p, np.ones((S, S), np.float32), dirt=design.dirt_amount())

    def wing(self, psd_path, design: Design):
        LAYER, SEAM, CX, W, ARM = 'Winglets (Paint Here)', 519, 1395, 50, 12
        acc = design.rgb('accent_colour') / 255
        sheet = PaintkitSheet.get(psd_path, LAYER)
        al = sheet.layer_alpha
        rgb = np.empty((S, S, 3), np.float32); rgb[:] = np.clip(acc / 0.73, 0, 1)   # multiply layer over 0.73 grey
        out = sheet.render(rgb, dirt=design.dirt_amount())

        if design.tail_symbol == 'saltire' and design.winglet_symbol:
            lab, _ = ndimage.label(al > 0.5)
            blade = lab == lab[500, 1400]
            rows = np.arange(S)[:, None]
            upper, lower = blade & (rows < SEAM), blade & (rows >= SEAM)
            im = Image.new('L', (S, S), 0); d = ImageDraw.Draw(im)
            x0, x1 = CX - W, CX + W
            for face, top in ((upper, True), (lower, False)):
                col0 = np.nonzero(face[:, x0])[0]; col1 = np.nonzero(face[:, x1])[0]
                e0 = col0.min() if top else col0.max(); e1 = col1.min() if top else col1.max()
                s0 = SEAM - (1 if top else 0)
                for (ax, ay), (bx, by) in (((x0, e0), (x1, s0)), ((x1, e1), (x0, s0))):
                    vx, vy = bx - ax, by - ay
                    d.line([(ax - vx * .06, ay - vy * .06), (bx + vx * .06, by + vy * .06)], fill=255, width=ARM)
            sm = np.asarray(im) > 127
            sm &= (ndimage.binary_dilation(upper, iterations=3) & (rows < SEAM)) | (ndimage.binary_dilation(lower, iterations=3) & (rows >= SEAM))
            sm &= al > 0.5
            out[sm] = np.clip(design.rgb('symbol_colour') * 0.985, 0, 255).astype(np.uint8)

        if design.tailplane_colour == 'accent':
            lum = out.astype(np.float32).mean(2) / 255
            lab2, _ = ndimage.label(out.sum(2) > 60)
            seeds = [(725, 570), (1520, 450), (1670, 450), (1235, 900)]
            tail = np.isin(lab2, [lab2[y, x] for y, x in seeds])
            ref = np.median(lum[tail])
            shade = np.clip(lum / ref, 0, 1.15)[..., None]
            out[tail] = np.clip(shade[tail] * design.rgb('accent_colour'), 0, 255).astype(np.uint8)
        return out

    def build(self, src, design, entry, out_root, progress=lambda m: None):
        kit = os.path.dirname(self.find(src, 'fuselage319.psd'))
        folder = os.path.join(out_root, self.folder_name(design, entry))
        obj = os.path.join(folder, 'objects')
        os.makedirs(obj, exist_ok=True)
        prev = {}
        progress(f'{entry.reg}: fuselage')
        fus = self.fuselage(kit, design, entry, progress)
        Image.fromarray(fus).save(os.path.join(obj, 'fuselage319.png'), compress_level=3)
        prev['Fuselage'] = preview(fus)
        progress(f'{entry.reg}: engines')
        eng = self.engines(kit, design)
        Image.fromarray(eng).save(os.path.join(obj, 'engines.png'), compress_level=3)
        prev['Engines'] = preview(eng)
        progress(f'{entry.reg}: wings')
        for side in ('wingL', 'wingR'):
            w = self.wing(os.path.join(kit, f'{side}.psd'), design)
            Image.fromarray(w).save(os.path.join(obj, f'{side}.png'), compress_level=3)
        prev['Wings'] = preview(w)
        selcal_placard(entry.reg, entry.selcal, os.path.join(obj, 'selcal.png'))
        eng_type = self.variant_key(entry)['eng']
        with open(os.path.join(folder, 'livery.tlscfg'), 'w') as fh:
            fh.write(f'eng_type = {eng_type}\nsharklet = YES\nhas_satcom = NO\nexit_Configuration = 1_OWE\nuse_Imperial_Units = NO\n')
        return prev
