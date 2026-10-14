"""ToLiss A321 / A321neo (paintkit V2, classic four-door fuselage)."""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from ..common import (fade_map, Design, FleetEntry, PaintkitSheet, apply_symbol, fade_ramp, fin_symbol, paint_mask_labels,
                      preview, reg_pair, selcal_placard, title_pair)
from .base import Template

S = 4096


class ToLissA321(Template):
    id = 'toliss_a321'
    name = 'ToLiss A321 / A321neo'
    model = 'A321'
    make = 'Toliss'
    source_kind = 'paintkit'
    source_hint = 'ToLiss A321 paintkit folder (contains fuselage321.psd)'
    required = ['fuselage321.psd', 'engines.psd', 'LEAP1A.psd', 'NEO.psd', 'wing321L.psd']
    variants = {
        'ceo CFM': {'eng': 'CFM', 'sheet': 'engines', 'model': 'A321', 'gen': 'CEO'},
        'ceo IAE': {'eng': 'IAE', 'sheet': 'engines', 'model': 'A321', 'gen': 'CEO'},
        'neo LEAP': {'eng': 'LEA', 'sheet': 'LEAP1A', 'model': 'A21N', 'gen': 'NEO'},
        'neo PW': {'eng': 'PWG', 'sheet': 'NEO', 'model': 'A21N', 'gen': 'NEO'},
    }
    TITLE = {'x_a': 842, 'x_b': 2029, 'win_y': 1133, 'y_sum': 1573, 'max_cap': 175, 'margin': 40,
             'files': ['objects/fuselage321.png'], 'band': (700, 1560)}
    model3d = {
        'hint': 'ToLiss A321 aircraft folder on your X-Plane PC (or a copy of its .obj files and a321.acf)',
        'acf': 'a321.acf',
        'objects': {'fuselage321.obj': 'livery', 'wing321L.obj': 'livery', 'wing321R.obj': 'livery',
                    'engines.obj': 'livery', 'neo.obj': 'livery', 'LEAP1A.obj': 'livery',
                    'windows.obj': 'glass', 'gear.obj': '#7C8086', 'blades.obj': '#3A3D42'},
        # fus 2 = A321, neoType 0 = classic four-door fuselage (the one the livery paints)
        'datarefs': {'1-sim/fus': 2, '1-sim/neoType': 0, 'anim/SHARK': 1},
        'variant_datarefs': {'ceo CFM': {'anim/CFM': 1}, 'ceo IAE': {'anim/IAE': 1},
                             'neo LEAP': {'AirbusFBW/EngineTypeIndex': 3}, 'neo PW': {'anim/NEO': 1}},
    }

    SEED_AFT, SEED_FIN, SEED_FIN_R, SEED_APU = (1307, 3083), (3688, 2729), (3348, 3781), (3867, 3683)
    FIN_QUAD = ((3114, 3037), (3844, 3037), (3789, 2234), (4035, 2234))
    FIN_SPEC = dict(file='objects/fuselage321.png', quad=FIN_QUAD)
    FIN_MIRROR = (7036, 1053)
    FADE = (300, 1750)
    FADE_Z = (17.14, 25.94)          # the same fade in metres along the fuselage (from the 3D model)
    WIN_L, WIN_SUM = 1133, 1133 + 440
    TITLE_X = (842, 2029)
    TITLE_MAX_CAP = 175
    REG = dict(cx=2450, base=3067, y_sum=5665, cap=46)
    REG_SPEC = dict(REG, files=['objects/fuselage321.png'], band=(2780, 3380), cols=(0, 3300))

    def fuselage(self, kit, design, entry, progress):
        path = os.path.join(kit, 'fuselage321.psd')
        lab = paint_mask_labels(path)
        ids = {k: lab[p[1], p[0]] for k, p in (('aft', self.SEED_AFT), ('fin', self.SEED_FIN),
                                               ('fin_r', self.SEED_FIN_R), ('apu', self.SEED_APU))}
        body, acc = design.rgb('body_colour'), design.rgb('accent_colour')
        t = fade_ramp(np.arange(S), *self.FADE, design)[None, :]
        aft = ndimage.binary_dilation(lab == ids['aft'], iterations=16)
        a = np.where(aft, np.broadcast_to(t, (S, S)), 0).astype(np.float32)
        tz, ok = fade_map(self.id, 'fuselage321', S, self.FADE_Z, design)
        a = np.where(ok, tz, a)                                    # by position along the fuselage: no seams
        for k in ('fin', 'fin_r', 'apu'):
            a[ndimage.binary_dilation(lab == ids[k], iterations=16)] = 1.0
        base = body[None, None, :] * (1 - a[..., None]) + acc[None, None, :] * a[..., None]

        A, B = self.FIN_MIRROR
        clip_ref = ndimage.binary_dilation(lab == ids['fin'], iterations=16)
        clip_other = ndimage.binary_dilation(lab == ids['fin_r'], iterations=16)
        sym = fin_symbol((S, S), self.FIN_QUAD, clip_ref, lambda x, y: (y + B, A - x), clip_other, design,
                         mapping_is_mirror=True, pos=(design.tail_pos or {}).get(self.id))
        base = apply_symbol(base, sym)

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

    def engines(self, kit, design, sheet):
        col = design.rgb('accent_colour' if design.engine_colour == 'accent' else 'body_colour') / 255
        p = np.zeros((S, S, 3), np.float32); p[...] = col
        return PaintkitSheet.get(os.path.join(kit, f'{sheet}.psd'), 'PAINT HERE').render(p, np.ones((S, S), np.float32), dirt=design.dirt_amount())

    def wing(self, kit, design):
        LAYER, SHIFT = 'PAINT WINGLETS HERE copy', (182, -267)
        sheet = PaintkitSheet.get(os.path.join(kit, 'wing321L.psd'), LAYER)
        al = sheet.layer_alpha
        orig = sheet.render(np.ones((S, S, 3), np.float32)).astype(np.float32) / 255     # paintkit as supplied (white)
        lum0 = orig.mean(2)
        reg = al > 0.5
        reg[:3400] = False
        P = reg & (lum0 > 0.5)
        REF = float(np.median(lum0[P])) if P.any() else 0.82
        acc = design.rgb('accent_colour') / 255
        rgb = np.empty((S, S, 3), np.float32); rgb[:] = np.clip(acc / REF, 0, 1)
        out = sheet.render(rgb, dirt=design.dirt_amount())

        if design.tail_symbol == 'saltire' and design.winglet_symbol:
            lower = P.copy(); lower[:3790] = False
            im = Image.new('L', (S, S), 0); d = ImageDraw.Draw(im)
            lab, _ = ndimage.label(ndimage.binary_erosion(lower, iterations=4))
            faces = np.zeros((S, S), bool)
            for i, sl in enumerate(ndimage.find_objects(lab)):
                m = lab == i + 1
                if m.sum() < 20000:
                    continue
                m = ndimage.binary_dilation(m, iterations=4) & lower
                yy, xx = np.nonzero(m)
                pts = np.stack([xx, yy], 1).astype(float); mu = pts.mean(0)
                _, _, vt = np.linalg.svd(pts - mu, full_matrices=False); ax, pp = vt[0], vt[1]
                proj = (pts - mu) @ ax; perp = (pts - mu) @ pp
                lo, hi = proj.min(), proj.max(); Ln = hi - lo
                wl = np.ptp(perp[proj < lo + 0.1 * Ln]); wh = np.ptp(perp[proj > hi - 0.1 * Ln])
                st = lo + 0.30 * Ln if wl < wh else hi - 0.30 * Ln
                band = np.abs(proj - st) < 6
                p0, p1 = perp[band].min(), perp[band].max()
                c = mu + ax * st + pp * (p0 + p1) / 2
                hw = (p1 - p0) / 2; hl = hw * 1.05
                arm = max(10, int(0.13 * (p1 - p0)))
                for sgn in (1, -1):
                    a_ = c - ax * hl * 1.1 - pp * hw * 1.1 * sgn
                    b_ = c + ax * hl * 1.1 + pp * hw * 1.1 * sgn
                    d.line([tuple(a_), tuple(b_)], fill=255, width=arm)
                faces |= m
            sm = (np.asarray(im) > 127) & faces
            ys, xs = np.nonzero(sm)
            sm[ys + SHIFT[1], xs + SHIFT[0]] = True
            out[sm] = np.clip(design.rgb('symbol_colour') * 0.985, 0, 255).astype(np.uint8)

        if design.tailplane_colour == 'accent':
            lum = out.astype(np.float32).mean(2) / 255
            lab2, _ = ndimage.label(out.sum(2) > 60)
            seeds = [(850, 500), (1483, 500), (1617, 500), (1258, 750), (1233, 900)]
            tail = np.isin(lab2, sorted({int(lab2[y, x]) for y, x in seeds} - {0}))
            ref = np.median(lum[tail])
            shade = np.clip(lum / ref, 0, 1.15)[..., None]
            out[tail] = np.clip(shade[tail] * design.rgb('accent_colour'), 0, 255).astype(np.uint8)
        return out

    def build(self, src, design, entry, out_root, progress=lambda m: None):
        kit = os.path.dirname(self.find(src, 'fuselage321.psd'))
        v = self.variant_key(entry)
        folder = os.path.join(out_root, self.folder_name(design, entry))
        obj = os.path.join(folder, 'objects')
        os.makedirs(obj, exist_ok=True)
        prev = {}
        progress(f'{entry.reg}: fuselage')
        fus = self.fuselage(kit, design, entry, progress)
        Image.fromarray(fus).save(os.path.join(obj, 'fuselage321.png'), compress_level=3)
        prev['Fuselage'] = preview(fus)
        progress(f'{entry.reg}: engines')
        eng = self.engines(kit, design, v['sheet'])
        Image.fromarray(eng).save(os.path.join(obj, f"{v['sheet']}.png"), compress_level=3)
        prev['Engines'] = preview(eng)
        progress(f'{entry.reg}: wings')
        w = self.wing(kit, design)
        for side in ('wing321L', 'wing321R'):
            Image.fromarray(w).save(os.path.join(obj, f'{side}.png'), compress_level=3)
        prev['Wings'] = preview(w)
        selcal_placard(entry.reg, entry.selcal, os.path.join(obj, 'selcal.png'))
        with open(os.path.join(folder, 'livery.tlscfg'), 'w') as fh:
            fh.write(f"eng_type = {v['eng']}\nsharklet = YES\n321neo_config = NEO\nhas_satcom = NO\n"
                     f"exit_Configuration = CLASSIC\nnum_Extra_Fuel_Tanks = 0\nuse_Imperial_Units = NO\n")
        return prev
