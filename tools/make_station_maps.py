"""Rebuild the fuselage position maps in livery_maker/engine/stations from the aircraft's 3D models.

Only needed when adding an aircraft or re-measuring one. Run from the LiveryMaker folder, for example:
    python tools/make_station_maps.py toliss_a319 "/path/to/ToLiss A319" fuselage319.obj=fuselage319
Arguments: template id, folder with that aircraft's .obj files and .acf, then one or more  object=texture  pairs
(the texture name without extension). Several objects can share a texture; they are averaged where they overlap.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from livery_maker.engine import obj8                      # noqa: E402
from livery_maker.engine.common import FleetEntry          # noqa: E402
from livery_maker.engine.templates import TEMPLATES        # noqa: E402


def station_map(objs, datarefs, texname, R=1024):
    """Distance along the fuselage (z, metres) for each texel, interpolated across every triangle; NaN elsewhere."""
    zsum = np.zeros((R, R)); cnt = np.zeros((R, R))
    for path in objs:
        m = obj8.load(path, datarefs)
        if m.texture.lower().rsplit('.', 1)[0] != texname.lower():
            continue
        P = m.positions.reshape(-1, 3, 3)[:, :, 2]; U = m.uvs.reshape(-1, 3, 2)
        X = U[..., 0] * R; Y = (1 - U[..., 1]) * R
        for x, y, z in zip(X, Y, P):
            if np.ptp(x) > R * 0.25 or np.ptp(y) > R * 0.25:
                continue
            x0, x1 = max(int(np.floor(x.min())), 0), min(int(np.ceil(x.max())) + 1, R)
            y0, y1 = max(int(np.floor(y.min())), 0), min(int(np.ceil(y.max())) + 1, R)
            if x1 <= x0 or y1 <= y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
            d = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
            if abs(d) < 1e-9:
                continue
            l0 = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / d
            l1 = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / d
            l2 = 1 - l0 - l1
            ins = (l0 >= -0.02) & (l1 >= -0.02) & (l2 >= -0.02)
            if ins.any():
                zz = l0 * z[0] + l1 * z[1] + l2 * z[2]
                zsum[y0:y1, x0:x1][ins] += zz[ins]; cnt[y0:y1, x0:x1][ins] += 1
    return np.where(cnt > 0, zsum / np.maximum(cnt, 1), np.nan)


def main(argv):
    if len(argv) < 3:
        print(__doc__); return 1
    tid, folder, pairs = argv[0], argv[1], argv[2:]
    t = TEMPLATES[tid]
    dr = t.datarefs_for(FleetEntry(tid, 'X', next(iter(t.variants)), ''))
    by_tex = {}
    for p in pairs:
        o, tex = p.split('=')
        by_tex.setdefault(tex, []).append(os.path.join(folder, o))
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'livery_maker', 'engine', 'stations')
    for tex, objs in by_tex.items():
        z = station_map(objs, dr, tex)
        np.savez_compressed(os.path.join(out, f'{tid}__{tex}.npz'), z=z.astype(np.float16))
        print(f'{tid} {tex}: {np.isfinite(z).mean():.0%} of the texture is fuselage')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
