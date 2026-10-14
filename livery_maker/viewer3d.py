"""3D preview: the aircraft's own X-Plane models wearing the livery that was just built."""
from __future__ import annotations

import ctypes
import math
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QPointF, Qt, QThread, Signal
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .engine import obj8

MAX_TEX = 4096


def configure_default_format():
    """Ask for a core OpenGL 4.1 context (what macOS offers). Call before QApplication is created."""
    f = QSurfaceFormat()
    f.setVersion(4, 1)
    f.setProfile(QSurfaceFormat.CoreProfile)
    f.setDepthBufferSize(24)
    f.setSamples(4)
    QSurfaceFormat.setDefaultFormat(f)


@dataclass
class Part:
    mesh: obj8.Mesh
    image: Optional[Image.Image]          # livery texture, or None for a flat colour
    colour: tuple = (0.6, 0.6, 0.62, 1.0)


def find_texture(root: str, name: str, skip_liveries=False) -> Optional[str]:
    """A texture by base name with any image extension (.png or .dds) somewhere under root."""
    if not root or not name or not os.path.isdir(root):
        return None
    stem = os.path.splitext(os.path.basename(name.replace('\\', '/')))[0].lower()
    for dp, dn, fns in os.walk(root):
        if skip_liveries:
            dn[:] = [d for d in dn if d.lower() != 'liveries']
        for fn in fns:
            s, e = os.path.splitext(fn)
            if s.lower() == stem and e.lower() in ('.png', '.dds'):
                return os.path.join(dp, fn)
    return None


def hex_rgba(h: str, alpha=1.0):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (alpha,)


def load_scene(spec: dict, model_dir: str, livery_dir: str, datarefs: dict, progress=lambda m: None,
               source_dir: str = '') -> List[Part]:
    """Load the template's exterior objects, posed, and dressed (runs off the UI thread).

    Each object's texture comes from the new livery if it has one, otherwise from the aircraft folder (its default
    textures) or the base livery. With none of those, it gets the flat colour from the template: 'livery' parts are
    light grey, '#RRGGBB' parts that colour, and 'glass' parts are always see-through dark glass.
    """
    acf = _find(model_dir, spec['acf'])
    offs = obj8.acf_offsets(acf) if acf else {}
    parts, cache = [], {}
    for name, how in spec['objects'].items():
        path = _find(model_dir, name)
        if not path or (offs and name.lower() not in offs):      # leftover files the aircraft doesn't attach
            continue
        progress(f'3D: loading {name}')
        mesh = obj8.load(path, datarefs, offs.get(name.lower(), (0.0, 0.0, 0.0)))
        if not len(mesh.positions):
            continue
        if how == 'glass':
            parts.append(Part(mesh, None, (0.08, 0.10, 0.13, 0.85)))
            continue
        col = hex_rgba(how) if how.startswith('#') else (0.85, 0.86, 0.87, 1.0)
        tp = (find_texture(livery_dir, mesh.texture) or find_texture(model_dir, mesh.texture, skip_liveries=True)
              or find_texture(source_dir, mesh.texture))
        img = None
        if tp:
            if tp not in cache:
                try:
                    im = Image.open(tp)
                    im = im.convert('RGBA' if im.mode in ('RGBA', 'LA', 'P') else 'RGB')
                    if max(im.size) > MAX_TEX:
                        im.thumbnail((MAX_TEX, MAX_TEX), Image.LANCZOS)
                    cache[tp] = im
                except Exception:
                    cache[tp] = None
            img = cache[tp]
        parts.append(Part(mesh, img, col))
    return parts


def _find(root: str, name: str) -> Optional[str]:
    if not root or not os.path.isdir(root):
        return None
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            if fn.lower() == name.lower():
                return os.path.join(dp, fn)
    return None


def model_folder_problems(spec: dict, model_dir: str) -> List[str]:
    if not model_dir or not os.path.isdir(model_dir):
        return ['Folder not set']
    need = [spec['acf']] + [n for n, how in spec['objects'].items() if how == 'livery']
    return [f'Missing {n}' for n in need if not _find(model_dir, n)]


# ------------------------------------------------------------------ maths
def perspective(fovy, aspect, near, far):
    f = 1 / math.tan(math.radians(fovy) / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = (np.asarray(v, np.float32) for v in (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


VERT = """#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in vec2 aUV;
uniform mat4 uMVP;
out vec3 vNrm; out vec2 vUV; out vec3 vPos;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); vNrm = aNrm; vUV = vec2(aUV.x, 1.0 - aUV.y); vPos = aPos; }
"""

FRAG = """#version 330 core
in vec3 vNrm; in vec2 vUV; in vec3 vPos;
uniform sampler2D uTex; uniform int uUseTex; uniform vec4 uColour; uniform vec3 uEye; uniform int uGround;
out vec4 frag;
void main() {
    if (uGround == 1) { frag = uColour; return; }
    vec4 c = uUseTex == 1 ? texture(uTex, vUV) : uColour;
    vec3 n = normalize(vNrm);
    vec3 v = normalize(uEye - vPos);
    if (dot(n, v) < 0.0) n = -n;
    vec3 L1 = normalize(vec3(-0.35, 0.85, -0.40));
    vec3 L2 = normalize(vec3(0.6, 0.3, 0.7));
    float dif = max(dot(n, L1), 0.0) * 0.62 + max(dot(n, L2), 0.0) * 0.18 + 0.32;
    float spec = pow(max(dot(reflect(-L1, n), v), 0.0), 48.0) * 0.18;
    frag = vec4(c.rgb * dif + spec, uUseTex == 1 ? 1.0 : uColour.a);
}
"""


class ModelView(QOpenGLWidget):
    """Orbit view: drag to turn, scroll or pinch to zoom, right-drag or shift-drag to pan, double-click to reset."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parts: List[Part] = []
        self.gpu = []                      # (vao, buffers, count, tex id or 0, colour, glass)
        self.dirty = False
        self.prog = None
        self.ground = None
        self.reset_view(update=False)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)

    # ---------------- scene
    def set_parts(self, parts: List[Part]):
        self.parts = parts
        if parts:
            P = np.concatenate([p.mesh.positions for p in parts])
            lo, hi = P.min(0), P.max(0)
            self.centre = (lo + hi) / 2
            self.radius = float(np.linalg.norm(hi - lo) / 2)
            self.floor = float(lo[1])
        self.dirty = True
        self.reset_view()

    def reset_view(self, update=True):
        self.yaw, self.pitch = -128.0, 14.0           # front-left three-quarter, slightly above
        self.zoom = 1.0
        self.pan = np.zeros(3, np.float32)
        if not hasattr(self, 'centre'):
            self.centre, self.radius, self.floor = np.zeros(3, np.float32), 20.0, -4.0
        if update:
            self.update()

    def set_side(self, which):
        self.yaw, self.pitch = {'left': (-90, 3), 'right': (90, 3), 'front': (180, 5), 'top': (-90, 89)}[which]
        self.pan = np.zeros(3, np.float32)
        self.update()

    # ---------------- GL
    def initializeGL(self):
        from OpenGL import GL
        self.GL = GL
        self.prog = self._program(VERT, FRAG)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.86, 0.89, 0.93, 1.0)

    def _program(self, vs, fs):
        GL = self.GL
        def sh(kind, src):
            s = GL.glCreateShader(kind); GL.glShaderSource(s, src); GL.glCompileShader(s)
            if not GL.glGetShaderiv(s, GL.GL_COMPILE_STATUS):
                raise RuntimeError(GL.glGetShaderInfoLog(s).decode())
            return s
        p = GL.glCreateProgram()
        GL.glAttachShader(p, sh(GL.GL_VERTEX_SHADER, vs)); GL.glAttachShader(p, sh(GL.GL_FRAGMENT_SHADER, fs))
        GL.glLinkProgram(p)
        if not GL.glGetProgramiv(p, GL.GL_LINK_STATUS):
            raise RuntimeError(GL.glGetProgramInfoLog(p).decode())
        return p

    def _vao(self, pos, nrm, uv):
        GL = self.GL
        vao = GL.glGenVertexArrays(1); GL.glBindVertexArray(vao)
        bufs = []
        for loc, arr, k in ((0, pos, 3), (1, nrm, 3), (2, uv, 2)):
            b = GL.glGenBuffers(1); bufs.append(b)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, b)
            a = np.ascontiguousarray(arr, np.float32)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, a.nbytes, a, GL.GL_STATIC_DRAW)
            GL.glEnableVertexAttribArray(loc)
            GL.glVertexAttribPointer(loc, k, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
        GL.glBindVertexArray(0)
        return vao, bufs

    def _free(self):
        GL = self.GL
        for vao, bufs, _n, tex, _c, _g in self.gpu:
            GL.glDeleteVertexArrays(1, [vao]); GL.glDeleteBuffers(len(bufs), bufs)
            if tex:
                GL.glDeleteTextures([tex])
        self.gpu = []

    def _upload(self):
        GL = self.GL
        self._free()
        texcache = {}
        for p in self.parts:
            m = p.mesh
            vao, bufs = self._vao(m.positions, m.normals, m.uvs)
            tex = 0
            if p.image is not None:
                key = id(p.image)
                if key not in texcache:
                    im = p.image
                    tex = GL.glGenTextures(1)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
                    fmt = GL.GL_RGBA if im.mode == 'RGBA' else GL.GL_RGB
                    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
                    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, fmt, im.size[0], im.size[1], 0, fmt, GL.GL_UNSIGNED_BYTE, im.tobytes())
                    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                    texcache[key] = tex
                tex = texcache[key]
            self.gpu.append((vao, bufs, len(m.positions), tex, p.colour, p.colour[3] < 1 or m.glass))
        # ground: a soft round floor under the wheels
        r, y, c = self.radius * 1.6, self.floor - 0.02, self.centre
        a = np.linspace(0, 2 * math.pi, 73)
        ring = np.stack([c[0] + r * np.cos(a), np.full_like(a, y), c[2] + r * np.sin(a)], 1)
        mid = np.array([c[0], y, c[2]])
        q = np.concatenate([[mid, ring[i + 1], ring[i]] for i in range(72)]).astype(np.float32)
        vao, bufs = self._vao(q, np.tile([0, 1, 0], (len(q), 1)), np.zeros((len(q), 2)))
        self.gpu.append((vao, bufs, len(q), 0, None, False))
        self.dirty = False

    def paintGL(self):
        GL = self.GL
        if self.dirty:
            self._upload()
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if not self.gpu:
            return
        w, h = max(1, self.width()), max(1, self.height())
        dist = self.radius * 2.0 / self.zoom
        yaw, pitch = math.radians(self.yaw), math.radians(self.pitch)
        target = self.centre + self.pan
        eye = target + dist * np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)], np.float32)
        up = (0, 0, -1) if abs(self.pitch) > 85 else (0, 1, 0)
        mvp = perspective(35, w / h, max(0.1, dist / 200), dist + self.radius * 8) @ look_at(eye, target, up)
        GL.glUseProgram(self.prog)
        loc = lambda n: GL.glGetUniformLocation(self.prog, n)
        GL.glUniformMatrix4fv(loc('uMVP'), 1, GL.GL_TRUE, mvp)
        GL.glUniform3f(loc('uEye'), *map(float, eye))
        GL.glUniform1i(loc('uTex'), 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        opaque = [g for g in self.gpu if not g[5]]
        clear = [g for g in self.gpu if g[5]]
        for vao, _b, n, tex, col, _glass in opaque:
            ground = col is None
            GL.glUniform1i(loc('uGround'), 1 if ground else 0)
            GL.glUniform1i(loc('uUseTex'), 1 if tex else 0)
            GL.glUniform4f(loc('uColour'), *((0.79, 0.82, 0.86, 1.0) if ground else col))
            if tex:
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glBindVertexArray(vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, n)
        if clear:
            GL.glEnable(GL.GL_BLEND); GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glDepthMask(GL.GL_FALSE)
            for vao, _b, n, tex, col, _glass in clear:
                GL.glUniform1i(loc('uGround'), 0); GL.glUniform1i(loc('uUseTex'), 0)
                GL.glUniform4f(loc('uColour'), *col)
                GL.glBindVertexArray(vao)
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, n)
            GL.glDepthMask(GL.GL_TRUE); GL.glDisable(GL.GL_BLEND)
        GL.glBindVertexArray(0)

    # ---------------- mouse
    def mousePressEvent(self, e):
        self._last = e.position()

    def mouseMoveEvent(self, e):
        d = e.position() - getattr(self, '_last', e.position())
        self._last = e.position()
        if e.buttons() & Qt.RightButton or e.modifiers() & Qt.ShiftModifier:
            s = self.radius * 2.0 / self.zoom / max(1, self.height()) * 0.8
            yaw = math.radians(self.yaw)
            right = np.array([math.cos(yaw), 0, -math.sin(yaw)], np.float32)
            self.pan += (-d.x() * right + d.y() * np.array([0, 1, 0], np.float32)) * s
        else:
            self.yaw -= d.x() * 0.4
            self.pitch = max(-30.0, min(89.0, self.pitch + d.y() * 0.3))
        self.update()

    def wheelEvent(self, e):
        steps = e.angleDelta().y() / 120 if e.angleDelta().y() else e.pixelDelta().y() / 40
        self.zoom = max(0.4, min(12.0, self.zoom * (1.12 ** steps)))
        self.update()

    def mouseDoubleClickEvent(self, e):
        self.reset_view()


class _Loader(QObject):
    done = Signal(object, str)
    progress = Signal(str)

    def __init__(self, spec, model_dir, livery_dir, datarefs, source_dir=''):
        super().__init__()
        self.args = (spec, model_dir, livery_dir, datarefs)
        self.source_dir = source_dir

    def run(self):
        try:
            self.done.emit(load_scene(*self.args, progress=self.progress.emit, source_dir=self.source_dir), '')
        except Exception as ex:                                    # shown in the panel, never crashes the app
            import traceback
            self.done.emit([], traceback.format_exc())


class ModelPanel(QWidget):
    """The 3D tab: the view plus a row of view buttons and a status line."""
    message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = ModelView()
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        for text, fn in (('Reset', self.view.reset_view), ('Left', lambda: self.view.set_side('left')),
                         ('Right', lambda: self.view.set_side('right')), ('Front', lambda: self.view.set_side('front')),
                         ('Top', lambda: self.view.set_side('top'))):
            b = QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
        row.addStretch(1)
        self.status = QLabel('Drag to turn · scroll to zoom · right-drag to move · double-click to reset')
        self.status.setStyleSheet('color:#666;')
        lay.addWidget(self.view, 1); lay.addLayout(row); lay.addWidget(self.status)
        self.thread = None

    def busy(self):
        return self.thread is not None and self.thread.isRunning()

    def load(self, spec, model_dir, livery_dir, datarefs, source_dir=''):
        self.status.setText('Loading 3D model…')
        self.thread = QThread(self)
        self.loader = _Loader(spec, model_dir, livery_dir, datarefs, source_dir)
        self.loader.moveToThread(self.thread)
        self.thread.started.connect(self.loader.run)
        self.loader.progress.connect(self.status.setText)
        self.loader.done.connect(self._loaded)
        self.loader.done.connect(self.thread.quit)
        self.thread.start()

    def _loaded(self, parts, err):
        if err:
            self.status.setText('Could not load the 3D model. Details are in the Progress box.')
            self.message.emit('3D preview error:\n' + err)
            return
        if not parts:
            self.status.setText('No exterior models found in the 3D model folder.')
            return
        self.view.set_parts(parts)
        kv = getattr(self, 'keep_view', None)
        if kv:                                                     # live refresh: keep the same view
            self.view.yaw, self.view.pitch, self.view.zoom, self.view.pan = kv
            self.view.update()
        self.status.setText('Drag to turn · scroll to zoom · right-drag to move · double-click to reset')
