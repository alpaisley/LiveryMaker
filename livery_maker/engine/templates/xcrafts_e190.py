"""X-Crafts E190, painted over an existing livery (KLM Cityhopper PH-EXD, DXT5 .dds textures)."""
from __future__ import annotations

import os
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from ..common import (fade_map, Design, FleetEntry, apply_symbol, fade_ramp, fin_symbol, font_path, mini_saltire, preview,
                      reg_pair, text_mask, title_pair, write_dds)
from .base import Template

S = 4096


class XCraftsE190(Template):
    id = 'xcrafts_e190'
    name = 'X-Crafts E190'
    model = 'E190'
    make = 'X-Crafts'
    source_kind = 'livery'
    source_hint = 'X-Crafts E-Jets KLM livery folder (contains objects/E190/FUSELAGE.dds)'
    base_note = 'KLM Cityhopper PH-EXD livery'
    required = ['FUSELAGE.dds', 'TAIL.dds', 'ENGINES_9.dds', 'WINGS_9.dds', 'REGISTRATION.dds', 'CABIN_2.dds', 'CABIN_2_LIT.dds']
    fingerprints = {'TAIL.dds': '6486dc1a8548873f86c66fad8676a051'}
    variants = {'CF34-10E': {'eng': 'CF34-10E'}}
    TITLE = {'x_a': 1160, 'x_b': 3140, 'win_y': 420, 'y_sum': 2000, 'max_cap': 260, 'margin': 80,
             'files': ['objects/E190/FUSELAGE.dds'], 'band': (0, 900)}
    model3d = {
        'hint': 'X-Crafts E190 aircraft folder on your X-Plane PC (or a copy of its .obj files and E190.acf)',
        'acf': 'E190.acf',
        'objects': {'FUSELAGE.obj': 'livery', 'TAIL.obj': 'livery', 'HOR_STAB_E195.obj': 'livery',
                    'WINGS_E195.obj': 'livery', 'ENGINES_E195.obj': 'livery', 'BELLY_E195.obj': '#C9CCD0',
                    'FANS_E195.obj': '#3A3D42', 'NOSE_GEAR.obj': '#7C8086', 'WINDSHIELD.obj': '#2E3135',
                    'WINDOWS_GLASS_EXT_E190.obj': 'glass', 'WINDSHIELD_GLASS_EXT.obj': 'glass'},
        'datarefs': {'sim/aircraft/parts/acf_gear_deploy[0]': 1, 'sim/aircraft/parts/acf_gear_deploy[1]': 1,
                     'sim/aircraft/parts/acf_gear_deploy[2]': 1},
    }

    FADE = (300, 2500)
    FIN_SPEC = dict(file='objects/TAIL.dds', quad=((1300, 4000), (3437, 4000), (3250, 1340), (4070, 1340)))
    REG_SPEC = dict(cap=46, cx=3450, base=2397, y_sum=6200, files=['objects/E190/FUSELAGE.dds'],
                    band=(2050, 2750), cols=(0, 4096))
    FADE_Z = (1.73, 11.71)           # the same fade in metres along the fuselage (from the 3D model)

    def e190_file(self, src, name):
        """Files inside objects/E190/ (the livery also carries E170/E175/E195 folders with the same names)."""
        for dp, _dn, fns in os.walk(src):
            if os.path.basename(dp) == 'E190' and name in fns:
                return os.path.join(dp, name)
        return self.find(src, name)

    def fuselage(self, path, design: Design, entry: FleetEntry):
        src = np.asarray(Image.open(path).convert('RGBA'))
        alpha = src[..., 3].copy()
        img = np.ascontiguousarray(src[..., :3])
        m = np.zeros((S, S), np.uint8)
        for x0, y0, x1, y1 in [(2125, 2357, 2353, 2434), (2124, 3758, 2353, 3835), (645, 915, 805, 965),
                               (645, 1035, 805, 1085), (815, 895, 836, 1095), (1100, 120, 2340, 360),
                               (1468, 1636, 2530, 1876), (1790, 2355, 2235, 2436), (1788, 3756, 2235, 3836)]:
            m[y0:y1, x0:x1] = 255
        img = cv2.inpaint(img, m, 9, cv2.INPAINT_TELEA)
        f = img.astype(np.float32)
        r, g, b = f[..., 0], f[..., 1], f[..., 2]
        lum = f.mean(2); sat = f.max(2) - f.min(2)
        gb = g / np.maximum(b, 1)
        LB = (b - r > 60) & (gb > 0.62)
        CH = (b - r > 30) & (gb <= 0.62) & (b > 60)
        WH = (sat < 30) & (lum > 150)
        s = np.clip(lum / np.where(LB, 167.0, np.where(CH, 70.7, 251.0)), 0.35, 1.06)
        body, acc = design.rgb('body_colour'), design.rgb('accent_colour')
        yy = np.arange(S)[:, None]; xx = np.arange(S)[None, :]
        out = f.copy()
        wl, _ = ndimage.label((LB | CH | WH) & (yy < 2000) & (xx < 3970))
        fseeds = [(2000, 200), (2000, 800), (2000, 1800), (1200, 1000)]
        fwd = np.isin(wl, sorted({int(wl[y, x]) for x, y in fseeds} - {0})) | ((yy < 2000) & (xx < 3970) & (LB | CH))
        tz, ok = fade_map(self.id, 'FUSELAGE', S, self.FADE_Z, design)     # by position along the fuselage
        tf = tz[fwd][:, None]
        out[fwd] = (body * (1 - tf) + acc * tf) * s[fwd][:, None]
        lab, _ = ndimage.label((LB | CH | WH) & (yy >= 2000))
        seeds = [(1000, 2300), (2000, 2450), (2000, 3000), (3000, 3600), (1000, 3850), (2600, 2150), (3600, 2350), (3600, 3850)]
        aft = np.isin(lab, sorted({int(lab[y, x]) for x, y in seeds} - {0})) | ((yy >= 2000) & (LB | CH))
        aft[2795:2953, 3415:3745] = False
        t = fade_ramp(np.arange(S), *self.FADE, design)
        T = body[None, :] * (1 - t)[:, None] + acc[None, :] * t[:, None]
        cols = np.nonzero(aft)
        ta = np.where(ok[cols], tz[cols], t[cols[1]])[:, None]
        out[cols] = (body * (1 - ta) + acc * ta) * s[cols][:, None]

        layer = Image.new('L', (S, S), 0)
        cap = self.draw_titles(layer, design)
        ol = self.title_outline(layer, cap, design)
        # shade titles by the surface (panel lines, grime), not by the colour under them
        sh = np.where((fwd | aft), np.clip(s, 0.6, 1.02), np.clip(out.mean(2) / 250.0, 0.6, 1.02))[..., None]
        if ol is not None:
            ol = ol.astype(np.float32)[..., None] / 255
            out = out * (1 - ol) + design.rgb('title_outline_colour') * sh * ol
        tl = np.asarray(layer).astype(np.float32)[..., None] / 255
        out = out * (1 - tl) + design.rgb('title_colour') * sh * tl
        layer = Image.new('L', (S, S), 0)
        self.draw_reg(layer, entry.reg, design)
        nm, _, _ = text_mask(entry.reg, design.reg_font, 36)
        nm = nm.crop(nm.getbbox()); nw, nh = nm.size
        layer.paste(nm, (725 - nw // 2, 940 - nh // 2), nm)
        r180 = nm.rotate(180)
        layer.paste(r180, (725 - nw // 2, 1060 - nh // 2), r180)
        rl = np.asarray(layer).astype(np.float32)[..., None] / 255
        out = out * (1 - rl) + design.rgb('reg_colour') * rl
        return np.dstack([np.clip(out + .5, 0, 255).astype(np.uint8), alpha])

    def tail(self, path, design: Design):
        src = np.asarray(Image.open(path).convert('RGBA'))
        alpha = src[..., 3]
        f = src[..., :3].astype(np.float32)
        r, g, b = f[..., 0], f[..., 1], f[..., 2]
        lum = f.mean(2); sat = f.max(2) - f.min(2)
        lab, _ = ndimage.label(ndimage.binary_erosion(alpha > 200, iterations=4))
        lab = ndimage.grey_dilation(lab, size=(9, 9)) * (alpha > 128)
        MAIN, RUD = lab[2076, 2062], lab[3561, 3837]
        main = np.isin(lab, [MAIN, RUD]) & (alpha > 128)
        acc = design.rgb('accent_colour')
        LB = (b - r > 60)
        skin = main & (((sat < 30) & (lum > 110)) | LB)
        ref = np.median(lum[main & (sat < 30) & (lum > 110)])
        s0 = np.clip(lum / ref, 0.5, 1.08)
        logo = ndimage.binary_dilation(LB & main, iterations=3)
        mk = (main & (sat < 30) & (lum > 110) & ~logo).astype(np.float32)
        fill = cv2.GaussianBlur(s0 * mk, (0, 0), 30) / np.maximum(cv2.GaussianBlur(mk, (0, 0), 30), 1e-3)
        s = s0.copy(); s[logo] = fill[logo]
        out = f.copy()
        out[skin] = acc * s[skin][:, None]
        other = LB & ~main & (alpha > 128)
        out[other] = acc * np.clip(lum[other] / 167.0, 0.3, 1.1)[:, None]
        quad = ((1300, 4000), (3437, 4000), (3250, 1340), (4070, 1340))
        qm = Image.new('L', (S, S), 0)
        rle, rte, tle, tte = (np.array(q, float) for q in quad)
        ImageDraw.Draw(qm).polygon([tuple(rle + (-30, 40)), tuple(rte + (40, 40)), tuple(tte + (40, -40)), tuple(tle + (-40, -40))], fill=255)
        clip = (lab == MAIN) & (alpha > 128)
        sym = fin_symbol((S, S), quad, (np.asarray(qm) > 127) & clip, lambda x, y: (5068 - y, 4164 - x), clip, design,
                         mapping_is_mirror=True, pos=(design.tail_pos or {}).get(self.id))   # checked on the 3D model
        out = apply_symbol(out, sym, s)
        return np.dstack([np.clip(out + .5, 0, 255).astype(np.uint8), alpha])

    def engines(self, path, design):
        src = np.asarray(Image.open(path).convert('RGBA'))
        alpha = src[..., 3]; f = src[..., :3].astype(np.float32)
        r, g, b = f[..., 0], f[..., 1], f[..., 2]
        lum = f.mean(2); sat = f.max(2) - f.min(2)
        yy, xx = np.mgrid[0:S, 0:S]
        box = (xx < 2490) & (yy > 90) & (yy < 2725)
        LB = (b - r > 60) & box
        skin = box & (((sat < 30) & (lum > 150)) | LB)
        ref = np.median(lum[box & (sat < 30) & (lum > 150)])
        s = np.clip(lum / ref, 0.45, 1.08); s[LB] = 1.0
        col = design.rgb('accent_colour' if design.engine_colour == 'accent' else 'body_colour')
        out = f.copy(); out[skin] = col * s[skin][:, None]
        return np.dstack([np.clip(out + .5, 0, 255).astype(np.uint8), alpha])

    def wings(self, path, design):
        src = np.asarray(Image.open(path).convert('RGBA'))
        alpha = src[..., 3]; f = src[..., :3].astype(np.float32)
        r, g, b = f[..., 0], f[..., 1], f[..., 2]
        lum = f.mean(2); sat = f.max(2) - f.min(2)
        yy, xx = np.mgrid[0:S, 0:S]
        x0, y0, x1, y1 = 3055, 3195, 2695, 3945
        left_ok = (xx - x0) * (y1 - y0) - (yy - y0) * (x1 - x0) >= 0
        right_ok = (xx - 3870) * (4096 - 3700) - (yy - 3700) * (3740 - 3870) <= 0
        zone = (yy >= 3180) & (yy < 4050) & (xx >= 2690) & left_ok & (right_ok | (yy < 3700))
        fb = ndimage.uniform_filter(f, size=(15, 15, 1))
        beige = (fb[..., 0] - fb[..., 2]) > 4
        zone &= ~beige
        LB = zone & (b - r > 60)
        grey = zone & (sat < 16) & (lum > 185)
        skin = grey | LB
        ref = np.median(lum[grey])
        s = np.clip(lum / ref, 0.6, 1.08); s[LB] = 1.0
        acc = design.rgb('accent_colour')
        out = f.copy(); out[skin] = acc * s[skin][:, None]
        boxes = [(2953, 3507, 3136, 3618), (3496, 3507, 3679, 3618), (3106, 3679, 3289, 3790), (3649, 3679, 3832, 3790)]
        for bx0, by0, bx1, by1 in boxes:
            sl = (slice(by0 - 12, by1 + 12), slice(bx0 - 12, bx1 + 12))
            blk = out[sl]; blk[~beige[sl]] = acc
        if design.tail_symbol == 'saltire' and design.winglet_symbol:
            im = Image.new('L', (S, S), 0); d = ImageDraw.Draw(im)
            for bx in boxes:
                mini_saltire(d, bx, 20)
            sm = (np.asarray(im) > 127)[..., None].astype(np.float32)
            out = out * (1 - sm) + design.rgb('symbol_colour') * 0.99 * sm
        return np.dstack([np.clip(out + .5, 0, 255).astype(np.uint8), alpha])

    def placard(self, path, reg):
        a = np.asarray(Image.open(path).convert('RGBA')).copy()
        x0, y0, x1, y1 = 505, 22, 1098, 160
        a[y0:y1, x0:x1, 3] = 0
        F = ImageFont.truetype(font_path('Inter Bold'), int(78 / 0.727))
        t = Image.new('L', (x1 - x0, y1 - y0), 0); d = ImageDraw.Draw(t)
        bb = d.textbbox((0, 0), reg, font=F)
        while bb[2] - bb[0] > (x1 - x0) - 40 and F.size > 20:
            F = ImageFont.truetype(font_path('Inter Bold'), F.size - 4); bb = d.textbbox((0, 0), reg, font=F)
        d.text(((x1 - x0 - (bb[2] - bb[0])) // 2 - bb[0], (y1 - y0 - (bb[3] - bb[1])) // 2 - bb[1]), reg, font=F, fill=255)
        mk = np.asarray(t)
        a[y0:y1, x0:x1, 3] = np.maximum(a[y0:y1, x0:x1, 3], mk)
        a[y0:y1, x0:x1, :3] = np.where(mk[..., None] > 0, 255, a[y0:y1, x0:x1, :3])
        return a

    def build(self, src, design, entry, out_root, progress=lambda m: None):
        folder = os.path.join(out_root, self.folder_name(design, entry))
        obj = os.path.join(folder, 'objects')
        os.makedirs(os.path.join(obj, 'E190'), exist_ok=True)
        prev = {}
        progress(f'{entry.reg}: fuselage')
        fus = self.fuselage(self.e190_file(src, 'FUSELAGE.dds'), design, entry)
        write_dds(Image.fromarray(fus, 'RGBA'), os.path.join(obj, 'E190', 'FUSELAGE.dds'), 'DXT5')
        prev['Fuselage'] = preview(fus)
        progress(f'{entry.reg}: tail')
        tail = self.tail(self.find(src, 'TAIL.dds'), design)
        write_dds(Image.fromarray(tail, 'RGBA'), os.path.join(obj, 'TAIL.dds'), 'DXT5')
        prev['Tail'] = preview(tail)
        progress(f'{entry.reg}: engines and wings')
        eng = self.engines(self.find(src, 'ENGINES_9.dds'), design)
        write_dds(Image.fromarray(eng, 'RGBA'), os.path.join(obj, 'ENGINES_9.dds'), 'DXT5')
        prev['Engines'] = preview(eng)
        wng = self.wings(self.find(src, 'WINGS_9.dds'), design)
        write_dds(Image.fromarray(wng, 'RGBA'), os.path.join(obj, 'WINGS_9.dds'), 'DXT5')
        prev['Wings'] = preview(wng)
        plac = self.placard(self.e190_file(src, 'REGISTRATION.dds'), entry.reg)
        write_dds(Image.fromarray(plac, 'RGBA'), os.path.join(obj, 'E190', 'REGISTRATION.dds'), 'DXT5')
        for n in ('CABIN_2.dds', 'CABIN_2_LIT.dds'):
            shutil.copy(self.e190_file(src, n), os.path.join(obj, 'E190', n))
        with open(os.path.join(obj, 'E190', 'tail.txt'), 'w') as fh:
            fh.write(entry.reg)
        return prev
