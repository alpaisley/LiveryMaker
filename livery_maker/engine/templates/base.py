"""Template interface: one class per supported aircraft add-on."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from ..common import Design, FleetEntry, Progress, noop, safe_name


class Template:
    id: str = ''
    name: str = ''                    # shown in the app, e.g. 'ToLiss A319'
    model: str = ''                   # ICAO-ish model for the fleet sheet (A319, A21N, ...)
    make: str = ''
    sim: str = 'Xplane'
    source_kind: str = 'paintkit'     # 'paintkit' | 'livery'
    source_hint: str = ''             # what folder to choose
    required: List[str] = []          # file names that must exist somewhere under the source folder
    variants: Dict[str, dict] = {}    # variant label -> template-specific settings
    fingerprints: Dict[str, str] = {} # file name -> md5 of the base livery the template was measured on
    base_note: str = ''               # e.g. which existing livery to use as the base
    # 3D preview (optional): {'acf': file, 'objects': {obj file: 'livery' | 'glass' | '#RRGGBB'},
    #                         'datarefs': {...}, 'variant_datarefs': {variant: {...}}, 'hint': text}
    model3d: Optional[dict] = None

    # ---- helpers
    @staticmethod
    def find(root: str, name: str) -> Optional[str]:
        for dp, _dn, fns in os.walk(root):
            if '__MACOSX' in dp:
                continue
            for fn in fns:
                if fn.lower() == name.lower():
                    return os.path.join(dp, fn)
        return None

    def check_source(self, root: str) -> List[str]:
        if not root or not os.path.isdir(root):
            return ['Source folder not set']
        return [f'Missing {r}' for r in self.required if not self.find(root, r)]

    def source_warnings(self, root: str) -> List[str]:
        from ..common import file_digest
        odd = [name for name, md5 in self.fingerprints.items()
               if root and (p := self.find(root, name)) and file_digest(p) != md5]
        if not odd:
            return []
        return [f'This isn’t the {self.base_note}, which this aircraft was measured on '
                f'({", ".join(odd)} differ), so markings may land in the wrong places.']

    def model_for(self, entry: FleetEntry) -> str:
        return self.variant_key(entry).get('model', self.model)

    def folder_name(self, design: Design, entry: FleetEntry) -> str:
        return safe_name(f'{design.airline} {self.model_for(entry)} {entry.reg} {entry.variant}')

    def variant_key(self, entry: FleetEntry) -> dict:
        return self.variants.get(entry.variant) or next(iter(self.variants.values()))

    # Fuselage titles: where they sit on the texture, and the strip of texture the title editor shows.
    # {'x_a', 'x_b', 'win_y', 'y_sum', 'max_cap', 'margin', 'files': [texture paths in the livery folder, joined
    #  side by side], 'band': (y0, y1)}
    TITLE: dict = {}

    def title_args(self, design: Design) -> dict:
        T, p = self.TITLE, (design.title_pos or {}).get(self.id, {})
        return dict(x_a=T['x_a'], x_b=T['x_b'], win_y=T['win_y'], max_cap=T['max_cap'], scale=design.title_scale,
                    margin=T['margin'], dx=int(p.get('dx', 0)), dy=int(p.get('dy', 0)), size=float(p.get('size', 1.0)))

    def draw_titles(self, layer, design: Design) -> int:
        from ..common import title_pair
        a = self.title_args(design)
        return title_pair(layer, design.title_text, design.title_font, a['x_a'], a['x_b'], a['win_y'],
                          self.TITLE['y_sum'], a['max_cap'], a['scale'], a['margin'], a['dx'], a['dy'], a['size'])

    # Fuselage registration: {'cap', 'cx', 'base', 'y_sum', 'anchor' ('base' or 'centre'), 'files', 'band', 'cols'}
    REG_SPEC: dict = {}
    # Fin for the tail logo: {'file': texture with the reference (left) fin side, 'quad': (root LE, root TE, tip LE, tip TE)}
    FIN_SPEC: dict = {}

    def reg_args(self, design: Design) -> dict:
        R, p = self.REG_SPEC, (design.reg_pos or {}).get(self.id, {})
        return dict(cap=R['cap'], cx=R['cx'], base_y=R['base'], anchor=R.get('anchor', 'base'),
                    dx=int(p.get('dx', 0)), dy=int(p.get('dy', 0)), size=float(p.get('size', 1.0)))

    def draw_reg(self, layer, reg: str, design: Design) -> None:
        from ..common import reg_pair
        if getattr(design, '_editor_blank', False):      # side view for the editor: leave it off
            return
        a = self.reg_args(design)
        reg_pair(layer, reg, design.reg_font, a['cap'], a['cx'], a['base_y'], self.REG_SPEC['y_sum'],
                 a['anchor'], a['dx'], a['dy'], a['size'])

    def reg_preview(self, design: Design, reg: str):
        """The left-side registration as the build would place it: (mask, x, y), or None."""
        from ..common import reg_left
        return reg_left(reg, design.reg_font, **self.reg_args(design))

    def title_outline(self, layer, cap: int, design: Design):
        """The outside stroke for titles drawn on `layer` (uint8 array), or None when the design has no outline."""
        from ..common import stroke_mask
        if design.title_outline <= 0 or cap <= 0:
            return None
        return stroke_mask(np.asarray(layer), max(1, int(round(cap * design.title_outline))))

    def title_preview(self, design: Design):
        """The left-side title as the build would place it: (mask, x, y, cap) in texture pixels, or None."""
        from ..common import title_left
        return title_left(design.title_text, design.title_font, **self.title_args(design))

    def datarefs_for(self, entry: FleetEntry) -> dict:
        d = dict((self.model3d or {}).get('datarefs', {}))
        d.update((self.model3d or {}).get('variant_datarefs', {}).get(entry.variant, {}))
        return d

    # ---- to implement
    def build(self, src: str, design: Design, entry: FleetEntry, out_root: str,
              progress: Progress = noop) -> Dict[str, Image.Image]:
        """Write the livery folder under out_root; return {sheet name: preview image}."""
        raise NotImplementedError
