"""Reader for X-Plane OBJ8 models, posed for a still 3D preview.

Only what a livery preview needs: vertices, texture coordinates, triangles and the animation commands, which are
evaluated once at fixed dataref values (gear down, doors shut, the right engine type shown). Lights, LODs other than
the first, smoke and manipulators are ignored.
"""
from __future__ import annotations

import io
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Mesh:
    name: str
    texture: str                      # texture file name from the OBJ (e.g. 'fuselage319.png'), '' if none
    positions: np.ndarray             # (n, 3) float32, metres, X-Plane axes (x right, y up, z aft)
    normals: np.ndarray               # (n, 3) float32
    uvs: np.ndarray                   # (n, 2) float32, v measured from the bottom of the image
    glass: bool = False               # BLEND_GLASS objects, drawn see-through


def _rot(axis, deg) -> np.ndarray:
    a = np.asarray(axis, np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(4)
    x, y, z = a / n
    t = math.radians(deg)
    c, s, C = math.cos(t), math.sin(t), 1 - math.cos(t)
    m = np.eye(4)
    m[:3, :3] = [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                 [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                 [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]
    return m


def _trans(v) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = v
    return m


def _interp(keys: List[Tuple[float, object]], val: float):
    """Piecewise-linear lookup in (dataref value, output) keys; clamped at the ends."""
    if len(keys) == 1:
        return keys[0][1]
    ks = sorted(keys, key=lambda k: k[0]) if keys[0][0] > keys[-1][0] else keys
    if val <= ks[0][0]:
        return ks[0][1]
    if val >= ks[-1][0]:
        return ks[-1][1]
    for (a, va), (b, vb) in zip(ks, ks[1:]):
        if a <= val <= b:
            t = 0.0 if b == a else (val - a) / (b - a)
            return np.asarray(va, np.float64) * (1 - t) + np.asarray(vb, np.float64) * t
    return ks[-1][1]


_VT = re.compile(rb'^[ \t]*VT[ \t]+(.*)$', re.M)
_IDX10 = re.compile(rb'^[ \t]*IDX10[ \t]+(.*)$', re.M)
_IDX = re.compile(rb'^[ \t]*IDX[ \t]+(\S+)[ \t]*$', re.M)


def load(path: str, datarefs: Optional[Dict[str, float]] = None, offset=(0.0, 0.0, 0.0)) -> Mesh:
    """Read an OBJ8 file and return its visible triangles posed at the given dataref values (missing ones are 0)."""
    dr = datarefs or {}
    raw = open(path, 'rb').read()
    head = raw[:4000].decode('latin1')
    tex = ''
    m = re.search(r'^\s*TEXTURE[ \t]+(\S+)', head, re.M)
    if m:
        tex = os.path.basename(m.group(1).replace('\\', '/'))
    glass = bool(re.search(r'^\s*BLEND_GLASS', head, re.M))

    vt = _VT.findall(raw)
    if not vt:
        return Mesh(os.path.basename(path), tex, *(np.zeros((0, k), np.float32) for k in (3, 3, 2)), glass=glass)
    V = np.loadtxt(io.BytesIO(b'\n'.join(vt)), dtype=np.float64, ndmin=2)[:, :8]
    idx = b' '.join(_IDX10.findall(raw)) + b' ' + b' '.join(_IDX.findall(raw))
    I = np.array(idx.split(), dtype=np.int64)

    # walk the command section
    stack: List[dict] = [{'M': np.eye(4), 'vis': True}]
    cur = stack[-1]
    tris: List[Tuple[np.ndarray, np.ndarray]] = []        # (index slice, matrix)
    lod_seen = 0
    in_lod = True
    pending = None                                         # keyframe table being collected

    def val(ref):
        return float(dr.get(ref, 0.0))

    for line in raw.decode('latin1').splitlines():
        s = line.split()
        if not s:
            continue
        c = s[0]
        if c in ('VT', 'IDX', 'IDX10') or c.startswith('#'):
            continue
        if pending is not None:
            if c in ('ANIM_trans_key', 'ANIM_rotate_key'):
                pending['keys'].append((float(s[1]), [float(x) for x in s[2:]]))
                continue
            if c == 'ANIM_keyframe_loop':
                continue
            if c in ('ANIM_trans_end', 'ANIM_rotate_end'):
                v = val(pending['ref'])
                if pending['kind'] == 'trans':
                    cur['M'] = cur['M'] @ _trans(_interp([(k, np.array(o)) for k, o in pending['keys']], v))
                else:
                    ang = float(np.ravel(_interp([(k, np.array(o[:1])) for k, o in pending['keys']], v))[0])
                    cur['M'] = cur['M'] @ _rot(pending['axis'], ang)
                pending = None
                continue
        if c == 'ATTR_LOD':
            lod_seen += 1
            in_lod = lod_seen == 1
        elif c == 'ANIM_begin':
            stack.append({'M': cur['M'].copy(), 'vis': cur['vis']})
            cur = stack[-1]
        elif c == 'ANIM_end':
            if len(stack) > 1:
                stack.pop()
                cur = stack[-1]
        elif c == 'ANIM_trans' and len(s) >= 7:
            a, b = np.array(s[1:4], float), np.array(s[4:7], float)
            v1, v2 = (float(s[7]), float(s[8])) if len(s) >= 9 else (0.0, 0.0)
            ref = s[9] if len(s) > 9 else ''
            t = 0.0 if (not ref or ref == 'none' or v1 == v2) else min(1.0, max(0.0, (val(ref) - v1) / (v2 - v1)))
            cur['M'] = cur['M'] @ _trans(a * (1 - t) + b * t)
        elif c == 'ANIM_rotate' and len(s) >= 6:
            axis = [float(x) for x in s[1:4]]
            r1, r2 = float(s[4]), float(s[5])
            v1, v2 = (float(s[6]), float(s[7])) if len(s) >= 8 else (0.0, 0.0)
            ref = s[8] if len(s) > 8 else ''
            t = 0.0 if (not ref or ref == 'none' or v1 == v2) else min(1.0, max(0.0, (val(ref) - v1) / (v2 - v1)))
            cur['M'] = cur['M'] @ _rot(axis, r1 * (1 - t) + r2 * t)
        elif c == 'ANIM_trans_begin':
            pending = {'kind': 'trans', 'ref': s[1] if len(s) > 1 else '', 'keys': []}
        elif c == 'ANIM_rotate_begin':
            pending = {'kind': 'rot', 'axis': [float(x) for x in s[1:4]], 'ref': s[4] if len(s) > 4 else '', 'keys': []}
        elif c in ('ANIM_hide', 'ANIM_show') and len(s) >= 4:
            v1, v2, v = float(s[1]), float(s[2]), val(s[3])
            if min(v1, v2) <= v <= max(v1, v2):
                cur['vis'] = c == 'ANIM_show' and all(st['vis'] for st in stack[:-1])
        elif c == 'TRIS' and len(s) >= 3:
            if cur['vis'] and in_lod:
                off, cnt = int(s[1]), int(s[2])
                tris.append((I[off:off + cnt], cur['M'].copy()))

    # bake: every triangle gets its own copies of its vertices, transformed by the matrix in force
    P, N, T = [], [], []
    for ix, M in tris:
        ix = ix[(ix >= 0) & (ix < len(V))]
        ix = ix[: len(ix) // 3 * 3]
        if not len(ix):
            continue
        v = V[ix]
        p = v[:, :3] @ M[:3, :3].T + M[:3, 3]
        n = v[:, 3:6] @ M[:3, :3].T
        P.append(p); N.append(n); T.append(v[:, 6:8])
    if not P:
        z = np.zeros((0, 3), np.float32)
        return Mesh(os.path.basename(path), tex, z, z, np.zeros((0, 2), np.float32), glass)
    P = np.concatenate(P) + np.asarray(offset, np.float64)
    N = np.concatenate(N)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    return Mesh(os.path.basename(path), tex, P.astype(np.float32), N.astype(np.float32),
                np.concatenate(T).astype(np.float32), glass)


def acf_offsets(acf_path: str) -> Dict[str, Tuple[float, float, float]]:
    """Object file name -> (x, y, z) placement in metres, from the .acf's attached-object list."""
    out: Dict[int, dict] = {}
    pat = re.compile(r'^P _obja/(\d+)/_v10_att_(file_stl|x_acf_prt_ref|y_acf_prt_ref|z_acf_prt_ref) (.*)$')
    with open(acf_path, 'rb') as fh:
        for line in fh:
            if not line.startswith(b'P _obja/'):
                continue
            m = pat.match(line.decode('latin1').rstrip())
            if m:
                out.setdefault(int(m.group(1)), {})[m.group(2)] = m.group(3).strip()
    ft = 0.3048
    res = {}
    for d in out.values():
        f = d.get('file_stl')
        if f:
            res[os.path.basename(f.replace('\\', '/')).lower()] = tuple(
                float(d.get(k, 0)) * ft for k in ('x_acf_prt_ref', 'y_acf_prt_ref', 'z_acf_prt_ref'))
    return res
