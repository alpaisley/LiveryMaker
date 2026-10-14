"""Embraer ERJ145, painted over an existing livery (Loganair G-SAJC tartan) - no paintkit available."""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

from ..common import (fade_map, Design, FleetEntry, apply_symbol, fade_ramp, fin_symbol, preview, reg_pair, text_mask,
                      title_pair)
from .base import Template

S = 4096


class ERJ145(Template):
    id = 'erj145'
    name = 'Embraer ERJ145'
    model = 'E145'
    make = 'X-Crafts'
    source_kind = 'livery'
    source_hint = 'ERJ145 Loganair livery folder (contains objects/E145/ERJ_FUSELAGE.png)'
    base_note = 'Loganair G-SAJC tartan livery'
    required = ['ERJ_FUSELAGE.png', 'ERJ_TAIL.png', 'ERJ_BELLY.png', 'ERJ_ENGINES.png', 'ERJ_WINGS.png']
    fingerprints = {'ERJ_FUSELAGE.png': '2ecaf943884233353d193644f20f49fb', 'ERJ_TAIL.png': 'ecd7f36fbca07683d2ca91706d9bd89c'}
    variants = {'AE3007': {'eng': 'AE3007'}}
    TITLE = {'x_a': 1790, 'x_b': 3580, 'win_y': 380, 'y_sum': 1862, 'max_cap': 260, 'margin': 0,
             'files': ['objects/E145/ERJ_FUSELAGE.png'], 'band': (0, 820)}
    model3d = {
        'hint': 'X-Crafts ERJ145 aircraft folder on your X-Plane PC (or a copy of its .obj files and E145.acf)',
        'acf': 'E145.acf',
        'objects': {'E145_FUSELAGE.obj': 'livery', 'ERJ_MAIN_DOOR.obj': 'livery', 'ERJ_BELLY.obj': 'livery',
                    'ERJ_WINGS.obj': 'livery', 'ERJ_WINGTIPS.obj': 'livery', 'ERJ_TAIL.obj': 'livery',
                    'ERJ_ENGINES.obj': 'livery', 'ERJ_NLG.obj': '#7C8086', 'ERJ_MLG.obj': '#7C8086',
                    'WHEELS_FAST.obj': '#2E3135', 'WINDOWS_GLASS_EXT.obj': 'glass'},
        # ACF_Type 3 = standard ERJ145 (it picks the matching window set; 4 and 5 are the XR, with winglets)
        'datarefs': {'XCrafts/ERJ/ACF_Type': 3, 'sim/aircraft/view/acf_peX': -0.46,
                     'sim/aircraft/parts/acf_gear_deploy[0]': 1, 'sim/aircraft/parts/acf_gear_deploy[1]': 1,
                     'sim/aircraft/parts/acf_gear_deploy[2]': 1},
    }

    FADE = (400, 1900)
    FIN_SPEC = dict(file='objects/ERJ_TAIL.png', quad=((2382, 179), (700, 40), (1200, 1560), (40, 1545)))
    REG_SPEC = dict(cap=56, cx=1639, base=2756, y_sum=6318, files=['objects/E145/ERJ_FUSELAGE.png'],
                    band=(2228, 2905), cols=(0, 4096))
    FADE_Z = (14.95, 20.69)          # the same fade in metres along the fuselage (from the 3D model)

    # ------------------------------------------------------------ fuselage
    def fuselage(self, path, design: Design, entry: FleetEntry):
        src = Image.open(path)
        alpha = np.asarray(src)[..., 3].copy()
        img = np.asarray(src.convert('RGB')).copy()
        acc = design.rgb('accent_colour')

        mask = np.zeros((S, S), np.uint8)
        for x0, y0, x1, y1 in [(1565, 459, 2627, 648), (1828, 1215, 2886, 1404),
                               (552, 2466, 1180, 2541), (515, 3786, 1142, 3858)]:
            mask[y0:y1, x0:x1] = 255
        for x0, y0, x1, y1 in [(1440, 2678, 1836, 2768), (1442, 3550, 1838, 3640), (890, 210, 950, 250), (895, 1628, 948, 1672)]:
            sub = img[y0:y1, x0:x1].astype(int)
            m = (sub.min(2) < 235) | ((sub.max(2) - sub.min(2)) > 25)
            mask[y0:y1, x0:x1][m] = 255
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        img = cv2.inpaint(img, mask, 9, cv2.INPAINT_TELEA)
        for x0, y0, x1, y1 in [(20, 876, 106, 931), (18, 951, 103, 1006)]:
            img[y0:y1, x0:x1] = img[y0 - 6:y0 - 3, x0:x1].mean((0, 1)).astype(np.uint8)

        f = img.astype(np.float32)
        lum = f.mean(2); sat = f.max(2) - f.min(2)
        skin = (lum > 100) & (sat < 40)
        shade = np.clip(lum / 250.0, 0, 1.03)

        t = fade_ramp(np.arange(S), *self.FADE, design)
        ba = np.zeros((S, S), np.float32)
        ba[2228:2905] = t[None, :]
        seg = img[3300:3530].astype(np.float32).mean(2)
        top = np.full(S, 3386)
        for x in range(1700):
            r = np.nonzero(seg[:, x] < 236)[0]
            if len(r):
                top[x] = max(3386, 3300 + r.max() + 3)
        top = ndimage.median_filter(top, size=9)
        for x in range(S):
            ba[top[x]:4096, x] = t[x]
        ba[2228:2995, 3250:] = 1.0
        ba[2960:3340, 2950:] = 1.0
        apply_skin = skin & (ba > 0)

        stripe_zone = np.zeros((S, S), bool)
        stripe_zone[2228:2995, 3250:] = True
        stripe_zone[3560:4096, 3250:] = True
        stripe_zone[2960:3340, 3600:] = True
        for x0, y0, x1, y1 in [(3835, 2300, 4096, 2480), (3600, 3570, 3760, 3690), (3880, 3720, 4000, 3840)]:
            stripe_zone[y0:y1, x0:x1] = False
        stripe = stripe_zone & (((f[..., 0] > 150) & (f[..., 0] - f[..., 1] > 80)) | (f.max(2) < 70))

        # fuselage skin (not cockpit parts or cargo bays) takes the body colour
        body = design.rgb('body_colour')
        wl, _ = ndimage.label((lum > 150) & (sat < 40))
        seeds = [(2000, 300), (2000, 1500), (1000, 2500), (1000, 3800), (3000, 600), (3000, 1300), (1660, 1400), (1280, 400), (310, 2630), (310, 3680)]
        bodymask = np.isin(wl, sorted({int(wl[y, x]) for x, y in seeds} - {0}))
        rows = np.arange(S)[:, None]
        bands = (rows < 1880) | ((rows >= 2228) & (rows < 2905)) | (rows >= top[None, :])
        bodymask &= bands & (np.arange(S)[None, :] < 3950)
        bodymask[950:1300, 3800:] = False                    # equipment piece next to the right-hand band
        # fade by position along the fuselage (no seams), on the skin only, not on loose parts
        tz, ok = fade_map(self.id, 'ERJ_FUSELAGE', S, self.FADE_Z, design)
        forced = ba >= 1.0
        ba = np.where(ok & ((ba > 0) | bodymask) & ~forced, tz, ba)
        apply_skin = skin & (ba > 0)
        base_col = np.where(bodymask[..., None], body[None, None, :] * shade[..., None], f)
        a3 = ba[..., None]
        out = np.where(apply_skin[..., None], base_col * (1 - a3) + acc[None, None, :] * shade[..., None] * a3, base_col)
        out[stripe] = acc * 0.97

        layer = Image.new('L', (S, S), 0)
        cap = self.draw_titles(layer, design)
        ol = self.title_outline(layer, cap, design)
        if ol is not None:
            ol = ol.astype(np.float32)[..., None] / 255
            out = out * (1 - ol) + design.rgb('title_outline_colour')[None, None, :] * shade[..., None] * ol
        tl = np.asarray(layer).astype(np.float32)[..., None] / 255
        out = out * (1 - tl) + design.rgb('title_colour')[None, None, :] * shade[..., None] * tl

        layer = Image.new('L', (S, S), 0)
        self.draw_reg(layer, entry.reg, design)
        tail2 = entry.reg.replace('-', '')[-2:]
        for (x0, y0, x1, y1), rot in [((893, 213, 947, 247), False), ((898, 1633, 945, 1668), True)]:
            lm, _, _ = text_mask(tail2, design.reg_font, y1 - y0)
            lm = lm.crop(lm.getbbox())
            if rot:
                lm = lm.rotate(180)
            layer.paste(lm, ((x0 + x1) // 2 - lm.size[0] // 2, (y0 + y1) // 2 - lm.size[1] // 2), lm)
        rl = np.asarray(layer).astype(np.float32)[..., None] / 255
        out = out * (1 - rl) + design.rgb('reg_colour') * rl
        layer = Image.new('L', (S, S), 0)
        for (x0, y0, x1, y1), rot in [((26, 880, 101, 927), False), ((23, 955, 98, 1002), True)]:
            lm, _, _ = text_mask(tail2, design.reg_font, y1 - y0)
            lm = lm.crop(lm.getbbox())
            if rot:
                lm = lm.rotate(180)
            layer.paste(lm, ((x0 + x1) // 2 - lm.size[0] // 2, (y0 + y1) // 2 - lm.size[1] // 2), lm)
        wl = np.asarray(layer).astype(np.float32)[..., None] / 255
        out = out * (1 - wl) + 240 * wl
        return np.dstack([np.clip(out + .5, 0, 255).astype(np.uint8), alpha])

    # ------------------------------------------------------------ tail
    def tail(self, path, design: Design):
        src = Image.open(path)
        alpha = np.asarray(src)[..., 3]
        f = np.asarray(src.convert('RGB')).astype(np.float32)
        lum = f.mean(2); sat = f.max(2) - f.min(2)
        acc = design.rgb('accent_colour')
        m = f.max(2) > 6
        m = ndimage.binary_fill_holes(ndimage.binary_closing(m, structure=np.ones((3, 3)), iterations=12))
        lab, _ = ndimage.label(m)
        FIN1, FIN2, STAB = lab[655, 1360], lab[1890, 2729], lab[3261, 984]
        out = f.copy()

        fins = np.isin(lab, [FIN1, FIN2])
        keep = (sat > 60) & (f[..., 2] > 150) & (f[..., 0] > 150)
        fins &= ~keep
        out[fins] = acc
        near = ndimage.binary_dilation(fins, iterations=25) & ~fins & ~keep
        redp = (f[..., 0] > 140) & (f[..., 0] - f[..., 1] > 70) & (f[..., 2] < 110)
        out[near & (redp | ((lum > 200) & (sat < 25)) | (lum < 45))] = acc

        quad = ((2382, 179), (700, 40), (1200, 1560), (40, 1545))       # rle, rte, tle, tte (fin side 1)
        body = Image.new('L', (S, S), 0)
        rle, rte, tle, tte = (np.array(q, float) for q in quad)
        ImageDraw.Draw(body).polygon([tuple(rte + (0, -40)), tuple(rle + (60, -40)), tuple(tle + (60, 60)), tuple(tte + (-60, 60))], fill=255)
        clip_ref = (np.asarray(body) > 127) & (lab == FIN1)
        clip_other = lab == FIN2
        sym = fin_symbol((S, S), quad, clip_ref, lambda x, y: (2544 - y, 4084 - x), clip_other, design,
                         mapping_is_mirror=True, flip_ref=True,
                         pos=(design.tail_pos or {}).get(self.id))   # orientation checked on the 3D model
        out = apply_symbol(out, sym)

        if design.tailplane_colour == 'accent':
            yy, xx = np.mgrid[0:S, 0:S]
            fb = ndimage.uniform_filter(f, size=(9, 9, 1))
            warm = (fb[..., 0] - fb[..., 2]) > 7
            stab = (lab == STAB) & (sat < 18) & (lum > 60) & ~warm & (xx >= (yy - 2240) * 0.37 - 20)
            del yy, xx
            ref = np.median(lum[stab])
            out = np.where(stab[..., None], acc * np.clip(lum / ref, 0, 1.15)[..., None], out)

        red = (f[..., 0] > 140) & (f[..., 0] - f[..., 1] > 70) & (f[..., 2] < 110)

        def zone(x0, y0, x1, y1):
            z = np.zeros((S, S), bool); z[y0:y1, x0:x1] = True; return z
        metal = np.zeros((S, S), bool)
        for x0, y0, x1, y1 in [(2957, 3418, 3195, 3592), (3778, 3418, 4024, 3598), (2356, 3792, 2614, 4004), (3190, 3803, 3438, 4004)]:
            metal[y0:y1, x0:x1] = True
        out[zone(2330, 3236, 4096, 4096) & ~metal & ~zone(2860, 3180, 3130, 3405) & ~zone(3550, 3180, 4096, 3338)] = acc
        for z in (zone(2180, 0, 4096, 280), zone(560, 4030, 1400, 4096)):
            piece = ndimage.binary_fill_holes(ndimage.binary_closing(z & (lum > 25), iterations=30)) & z
            out[piece & (red | (lum < 70) | ((lum > 190) & (sat < 25))) & ~fins] = acc
        out[red & ~fins] = acc
        return np.dstack([np.clip(out + .5, 0, 255).astype(np.uint8), alpha])

    # ------------------------------------------------------------ belly + engines
    def belly(self, path):
        src = Image.open(path); a = np.asarray(src)[..., 3]
        f = np.asarray(src.convert('RGB')).astype(np.float32)
        lum = f.mean(2); sat = f.max(2) - f.min(2)
        zone = np.zeros((S, S), bool); zone[2380:4096, :] = True
        outline = ndimage.binary_fill_holes(ndimage.binary_closing(zone & (lum > 0.5), iterations=6)) & zone
        dark = outline & (lum >= 1) & (lum < 60) & (sat < 10)
        sh = np.clip(lum / 3.0, 0.3, 1.6)[..., None]
        LIGHT = np.array((222, 225, 229), np.float32)
        out = np.where(dark[..., None], np.clip(LIGHT * (0.9 + 0.1 * sh), 0, 255), f)
        return np.dstack([np.clip(out + .5, 0, 255).astype(np.uint8), a])

    def engines(self, path, design):
        src = Image.open(path); a = np.asarray(src)[..., 3]
        f = np.asarray(src.convert('RGB')).astype(np.float32)
        r, g, b = f[..., 0], f[..., 1], f[..., 2]
        red = (r > 30) & (r > 1.7 * g) & (r > 1.7 * b)
        ref = np.median(r[red & (r > 120)])
        col = design.rgb('accent_colour' if design.engine_colour == 'accent' else 'body_colour')
        out = np.where(red[..., None], np.clip(col * np.clip(r / ref, 0, 1.2)[..., None], 0, 255), f)
        return np.dstack([np.clip(out + .5, 0, 255).astype(np.uint8), a])

    def build(self, src, design, entry, out_root, progress=lambda m: None):
        folder = os.path.join(out_root, self.folder_name(design, entry))
        obj = os.path.join(folder, 'objects')
        os.makedirs(os.path.join(obj, 'E145'), exist_ok=True)
        prev = {}
        progress(f'{entry.reg}: fuselage')
        fus = self.fuselage(self.find(src, 'ERJ_FUSELAGE.png'), design, entry)
        Image.fromarray(fus, 'RGBA').save(os.path.join(obj, 'E145', 'ERJ_FUSELAGE.png'), compress_level=3)
        prev['Fuselage'] = preview(fus)
        progress(f'{entry.reg}: tail')
        tail = self.tail(self.find(src, 'ERJ_TAIL.png'), design)
        Image.fromarray(tail, 'RGBA').save(os.path.join(obj, 'ERJ_TAIL.png'), compress_level=3)
        prev['Tail'] = preview(tail)
        progress(f'{entry.reg}: belly and engines')
        Image.fromarray(self.belly(self.find(src, 'ERJ_BELLY.png')), 'RGBA').save(os.path.join(obj, 'ERJ_BELLY.png'), compress_level=3)
        eng = self.engines(self.find(src, 'ERJ_ENGINES.png'), design)
        Image.fromarray(eng, 'RGBA').save(os.path.join(obj, 'ERJ_ENGINES.png'), compress_level=3)
        prev['Engines'] = preview(eng)
        import shutil
        shutil.copy(self.find(src, 'ERJ_WINGS.png'), os.path.join(obj, 'ERJ_WINGS.png'))
        cam = self.find(src, 'X-Camera_E145.csv')
        if cam:
            shutil.copy(cam, folder)
        return prev
