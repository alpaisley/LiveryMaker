"""Title position editor: drag the fuselage titles or the registration on a flat side view; the other side of
the aircraft follows in the build."""
from __future__ import annotations

import copy
import os
import tempfile
import traceback

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QVBoxLayout,
                               QWidget)

from .engine.common import FleetEntry, Project
from .engine.export import build_one
from .engine.templates import TEMPLATES

Image.MAX_IMAGE_PIXELS = None
BG_MAX_W = 2600            # side view background is kept at most this wide


def side_strip(t, livery_dir: str, spec: dict = None):
    """Join the template's fuselage textures side by side and cut out the band given by spec (default: titles)."""
    T = spec or t.TITLE
    y0, y1 = T['band']
    parts = []
    for rel in T['files']:
        p = os.path.join(livery_dir, rel)
        if not os.path.exists(p):
            return None
        im = Image.open(p)
        im = im.convert('RGB').crop((0, y0, im.size[0], y1))
        parts.append(im)
    full_w = sum(p.size[0] for p in parts)
    joined = Image.new('RGB', (full_w, y1 - y0))
    x = 0
    for p in parts:
        joined.paste(p, (x, 0)); x += p.size[0]
    x0, x1 = T.get('cols', (0, full_w))
    joined = joined.crop((x0, 0, x1, y1 - y0))
    k = min(1.0, BG_MAX_W / (x1 - x0))
    return joined.resize((int((x1 - x0) * k), int((y1 - y0) * k)), Image.LANCZOS), (x0, x1)


def fin_view(t, livery_dir: str, max_side: int = 1000):
    """The reference fin side straightened upright (leading edge left, tip up): (image, info) or None."""
    from PIL import ImageDraw
    from .engine.common import fin_frame
    F = t.FIN_SPEC
    p = os.path.join(livery_dir, F['file'])
    if not os.path.exists(p):
        return None
    ctr, right, up, H, es = fin_frame(F['quad'])
    pts = [np.array(q, float) - ctr for q in F['quad']]
    us = [q @ right for q in pts]; vs = [q @ up for q in pts]
    pad = 0.18 * H
    umin, umax, vmin, vmax = min(us) - pad, max(us) + pad, min(vs) - pad, max(vs) + pad
    k = min(1.0, max_side / max(umax - umin, vmax - vmin))
    W, Hs = int((umax - umin) * k), int((vmax - vmin) * k)
    a, b = right / k, -up / k
    c0 = ctr + right * umin + up * vmax
    tex = Image.open(p).convert('RGB')
    img = tex.transform((W, Hs), Image.AFFINE, (a[0], b[0], c0[0], a[1], b[1], c0[1]), resample=Image.BICUBIC)
    poly = [((u - umin) * k, (vmax - v) * k) for u, v in zip(us, vs)]
    poly = [poly[0], poly[1], poly[3], poly[2]]                          # root LE, root TE, tip TE, tip LE
    clip = Image.new('L', (W, Hs), 0)
    ImageDraw.Draw(clip).polygon(poly, fill=255)
    info = dict(H=H, k=k, umin=umin, vmax=vmax, W=W, Hs=Hs, es=(float(es @ right), float(es @ up)), clip=clip)
    return img, info


def fin_overlay(design, info, pos):
    """The tail symbol in the straightened fin view, clipped to the fin, as an RGBA image."""
    from PIL import ImageDraw, ImageFilter
    H, k, W, Hs = info['H'], info['k'], info['W'], info['Hs']
    cx = (-info['umin'] + float(pos.get('dx', 0)) * H) * k
    cy = (info['vmax'] - float(pos.get('dy', 0)) * H) * k
    size = float(pos.get('size', 1.0))
    ang = float(pos.get('angle', 0.0))
    ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    out = Image.new('RGBA', (W, Hs), (0, 0, 0, 0))
    if design.tail_symbol == 'saltire':
        m = Image.new('L', (W, Hs), 0); d = ImageDraw.Draw(m)
        e = np.array(info['es'])
        arm = max(2, int(design.saltire_width * H * size * k))
        for v in (np.array([1.0, 0.0]) + e, np.array([-1.0, 0.0]) + e):
            v = v / np.linalg.norm(v); L = 3 * H * k
            v = np.array([v[0] * ca - v[1] * sa, v[0] * sa + v[1] * ca])          # x right, y up
            d.line([(cx - v[0] * L, cy + v[1] * L), (cx + v[0] * L, cy - v[1] * L)], fill=255, width=arm)
        m = m.filter(ImageFilter.GaussianBlur(0.6))
        col = Image.new('RGBA', (W, Hs), tuple(int(c) for c in design.rgb('symbol_colour')) + (0,))
        col.putalpha(m)
        out = col
    elif design.tail_symbol == 'image' and design.tail_image and os.path.exists(design.tail_image):
        logo = Image.open(design.tail_image).convert('RGBA')
        h = max(2, int(design.tail_image_size * size * H * k))
        w = max(2, int(logo.size[0] * h / logo.size[1]))
        logo = logo.resize((w, h), Image.LANCZOS)
        if ang:
            logo = logo.rotate(ang, resample=Image.BICUBIC, expand=True)        # + = anticlockwise
            w, h = logo.size
        out.paste(logo, (int(cx - w / 2), int(cy - h / 2)), logo)
    a = np.asarray(out)[..., 3].astype(np.float32) * (np.asarray(info['clip']).astype(np.float32) / 255)
    out.putalpha(Image.fromarray(a.astype(np.uint8)))
    return out


def pil_qimage(im: Image.Image) -> QImage:
    im = im.convert('RGBA')
    return QImage(im.tobytes('raw', 'RGBA'), im.size[0], im.size[1], QImage.Format_RGBA8888).copy()


class _BgWorker(QObject):
    done = Signal(object, str)          # (template id, livery folder) or error
    progress = Signal(str)

    def __init__(self, project: Project, entry: FleetEntry):
        super().__init__()
        self.project, self.entry = project, entry

    def run(self):
        try:
            build_one(self.project, self.entry, self.progress.emit)
            t = TEMPLATES[self.entry.template]
            self.done.emit(os.path.join(self.project.output_dir, t.folder_name(self.project.design, self.entry)), '')
        except Exception:
            self.done.emit('', traceback.format_exc())


class TitleCanvas(QWidget):
    moved = Signal(int, int)            # dx, dy while dragging
    released = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.bg = None                  # QPixmap of the strip
        self.full_w, self.band, self.xoff = 1, (0, 1), 0
        self.win_y = 0
        self.title = None               # (QImage colour title, tx, ty, w, h) in texture pixels
        self.dx = self.dy = 0
        self.snap = 0
        self.box = True                 # dashed box round the thing being moved
        self.drag = None
        self.message = ''

    # geometry: texture px -> widget px
    def frame(self):
        H = self.band[1] - self.band[0]
        k = min((self.width() - 20) / self.full_w, (self.height() - 20) / H)
        ox = (self.width() - self.full_w * k) / 2
        oy = (self.height() - H * k) / 2
        return k, ox, oy

    def to_widget(self, x, y):
        k, ox, oy = self.frame()
        return ox + (x - self.xoff) * k, oy + (y - self.band[0]) * k

    def title_rect(self):
        if not self.title:
            return QRectF()
        _img, tx, ty, w, h = self.title
        k, _, _ = self.frame()
        x, y = self.to_widget(tx, ty)
        return QRectF(x, y, w * k, h * k)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor('#2B2D31'))
        if self.bg is None:
            p.setPen(QColor('#BBBBBB'))
            p.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self.message or 'Press “Load side view”.')
            return
        k, ox, oy = self.frame()
        H = self.band[1] - self.band[0]
        p.drawPixmap(QRectF(ox, oy, self.full_w * k, H * k), self.bg, QRectF(self.bg.rect()))
        # guide line (window line / usual registration height)
        if self.win_y is not None:
            _, wy = self.to_widget(0, self.win_y)
            pen = QPen(QColor(255, 140, 0, 150)); pen.setStyle(Qt.DashLine); pen.setWidth(1)
            p.setPen(pen); p.drawLine(QPointF(ox, wy), QPointF(ox + self.full_w * k, wy))
        if self.title:
            r = self.title_rect()
            p.drawImage(r, self.title[0])
            if self.box:
                pen = QPen(QColor(0, 160, 255, 220) if self.drag else QColor(0, 160, 255, 120)); pen.setStyle(Qt.DashLine)
                p.setPen(pen); p.drawRect(r)

    def mousePressEvent(self, e):
        if self.title and self.title_rect().contains(e.position()):
            self.drag = (e.position(), self.dx, self.dy)
            self.setCursor(Qt.ClosedHandCursor)
            self.update()

    def mouseMoveEvent(self, e):
        if not self.drag:
            over = self.title and self.title_rect().contains(e.position())
            self.setCursor(Qt.OpenHandCursor if over else Qt.ArrowCursor)
            return
        start, dx0, dy0 = self.drag
        k, _, _ = self.frame()
        dx = int(round(dx0 + (e.position().x() - start.x()) / k))
        dy = int(round(dy0 + (e.position().y() - start.y()) / k))
        if abs(dy) < self.snap:                 # stay on the window line unless pulled clear
            dy = 0
        self.moved.emit(dx, dy)

    def mouseReleaseEvent(self, _e):
        if self.drag:
            self.drag = None
            self.setCursor(Qt.OpenHandCursor)
            self.update()
            self.released.emit()

    def keyPressEvent(self, e):
        step = 20 if e.modifiers() & Qt.ShiftModifier else 4
        d = {Qt.Key_Left: (-step, 0), Qt.Key_Right: (step, 0), Qt.Key_Up: (0, -step), Qt.Key_Down: (0, step)}.get(e.key())
        if d and self.title:
            self.moved.emit(self.dx + d[0], self.dy + d[1])
            self.released.emit()
        else:
            super().keyPressEvent(e)


class TitleEditor(QWidget):
    """The 'Title position' tab. get_project() returns the current project; positions are reported through changed."""
    changed = Signal()
    log = Signal(str)

    def __init__(self, get_project):
        super().__init__()
        self.get_project = get_project
        self.positions = {}             # titles: template id -> {'dx', 'dy', 'size'}
        self.reg_positions = {}         # registration: the same
        self.tail_positions = {}        # tail logo: {'dx', 'dy' as fractions of fin height, 'size'}
        self.fin_info = None
        self.livery_dirs = {}           # template id -> livery folder built without titles
        self.thread = None

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.aircraft = QComboBox()
        for tid, t in TEMPLATES.items():
            if t.TITLE:
                self.aircraft.addItem(t.name, tid)
        self.aircraft.currentIndexChanged.connect(self.show_current)
        self.load_btn = QPushButton('Load side view'); self.load_btn.clicked.connect(self.load_background)
        self.reset_btn = QPushButton('Reset'); self.reset_btn.clicked.connect(self.reset)
        self.item = QComboBox(); self.item.addItems(['Title', 'Registration', 'Tail logo'])
        self.item.currentIndexChanged.connect(self.show_current)
        top.addWidget(QLabel('Aircraft')); top.addWidget(self.aircraft, 1); top.addWidget(QLabel('Move'))
        top.addWidget(self.item); top.addWidget(self.load_btn); top.addWidget(self.reset_btn)
        lay.addLayout(top)

        self.canvas = TitleCanvas()
        self.canvas.moved.connect(self.on_moved)
        self.canvas.released.connect(self.changed)
        lay.addWidget(self.canvas, 1)

        row = QHBoxLayout()
        self.size = QSlider(Qt.Horizontal); self.size.setRange(40, 180); self.size.setValue(100)
        self.size.valueChanged.connect(self.on_size)
        self.size.sliderReleased.connect(self.changed)
        self.size_lab = QLabel('100%'); self.size_lab.setMinimumWidth(44)
        row.addWidget(QLabel('Size')); row.addWidget(self.size, 1); row.addWidget(self.size_lab)
        self.angle = QSlider(Qt.Horizontal); self.angle.setRange(-90, 90); self.angle.setValue(0)
        self.angle.valueChanged.connect(self.on_angle)
        self.angle.sliderReleased.connect(self.changed)
        self.angle_lab = QLabel('0°'); self.angle_lab.setMinimumWidth(44)
        self.angle_name = QLabel('Angle')
        for wdg in (self.angle_name, self.angle, self.angle_lab):
            wdg.setVisible(False)
        row.addSpacing(12); row.addWidget(self.angle_name); row.addWidget(self.angle, 1); row.addWidget(self.angle_lab)
        lay.addLayout(row)
        self.info = QLabel('Drag the title to move it, or use the arrow keys (Shift for bigger steps). It stays on the '
                           'window line unless you pull it clear. The other side of the aircraft follows automatically.')
        self.info.setWordWrap(True); self.info.setStyleSheet('color:#666;')
        lay.addWidget(self.info)

    # ---------------- state
    def tid(self):
        return self.aircraft.currentData()

    def is_reg(self):
        return self.item.currentIndex() == 1

    def is_tail(self):
        return self.item.currentIndex() == 2

    def store(self) -> dict:
        return self.tail_positions if self.is_tail() else self.reg_positions if self.is_reg() else self.positions

    def spec(self, tid=None) -> dict:
        t = TEMPLATES[tid or self.tid()]
        return t.REG_SPEC if self.is_reg() else t.TITLE

    def set_positions(self, pos: dict, reg_pos: dict = None, tail_pos: dict = None):
        self.positions = copy.deepcopy(pos or {})
        self.reg_positions = copy.deepcopy(reg_pos or {})
        self.tail_positions = copy.deepcopy(tail_pos or {})
        self.livery_dirs = {}
        self.show_current()

    def pos(self):
        return self.store().setdefault(self.tid(), {'dx': 0, 'dy': 0, 'size': 1.0})

    def reg_text(self):
        proj = self.get_project()
        e = next((e for e in proj.fleet if e.template == self.tid() and e.reg), None)
        return e.reg if e else 'G-TEST'

    def select(self, tid):
        i = self.aircraft.findData(tid)
        if i >= 0:
            self.aircraft.setCurrentIndex(i)

    # ---------------- drawing
    def show_current(self, *_):
        tid = self.tid()
        p = self.store().get(tid, {})
        for wdg in (self.angle_name, self.angle, self.angle_lab):
            wdg.setVisible(self.is_tail())
        if self.is_tail():
            return self.show_fin(tid, p)
        self.canvas.box = True
        self.info.setText(
            'Drag the registration to move it, or use the arrow keys (Shift for bigger steps). It stays at its usual '
            'height unless you pull it clear. The other side of the aircraft follows automatically.' if self.is_reg() else
            'Drag the title to move it, or use the arrow keys (Shift for bigger steps). It stays on the window line '
            'unless you pull it clear. The other side of the aircraft follows automatically.')
        self.size.blockSignals(True); self.size.setValue(int(round(p.get('size', 1.0) * 100))); self.size.blockSignals(False)
        self.size_lab.setText(f'{self.size.value()}%')
        c = self.canvas
        d = self.livery_dirs.get(tid)
        spec = self.spec(tid)
        strip = side_strip(TEMPLATES[tid], d, spec) if d else None
        if strip:
            img, (x0, x1) = strip
            c.xoff, c.full_w = x0, x1 - x0
            c.bg = QPixmap.fromImage(pil_qimage(img))
        else:
            c.bg = None
            c.message = 'Press “Load side view” to see this aircraft’s fuselage. It builds the livery once without ' \
                        'titles or registration (about a minute for the first aircraft of a type).'
        c.band = spec['band']
        c.win_y = spec['base'] if self.is_reg() else spec['win_y']
        self.refresh_title()

    def show_fin(self, tid, p):
        self.info.setText('Drag on the fin to move the tail logo, or use the arrow keys (Shift for bigger steps). It '
                          'snaps to the middle height unless you pull it clear. Angle tilts it: plus leans the top '
                          'towards the nose. The other side of the fin follows automatically. Choose the saltire or '
                          'your own PNG on the Design tab.')
        self.size.blockSignals(True); self.size.setValue(int(round(p.get('size', 1.0) * 100))); self.size.blockSignals(False)
        self.size_lab.setText(f'{self.size.value()}%')
        self.angle.blockSignals(True); self.angle.setValue(int(round(p.get('angle', 0.0)))); self.angle.blockSignals(False)
        self.angle_lab.setText(f'{self.angle.value()}°')
        c = self.canvas
        d = self.livery_dirs.get(tid)
        fv = fin_view(TEMPLATES[tid], d) if d else None
        self.fin_info = fv[1] if fv else None
        if fv:
            img, info = fv
            c.xoff, c.full_w, c.band = 0, info['W'], (0, info['Hs'])
            c.bg = QPixmap.fromImage(pil_qimage(img))
        else:
            c.bg = None
            c.message = 'Press “Load side view” to see this aircraft’s fin. It builds the livery once without ' \
                        'titles, registration or tail logo (about a minute for the first aircraft of a type).'
        c.win_y = None
        c.box = False
        self.refresh_title()

    def refresh_title(self):
        """Re-render the title from the current design (text, font, size, colour) and position."""
        tid = self.tid()
        c = self.canvas
        if self.is_tail():
            info = self.fin_info
            if not info:
                c.title = None; c.update(); return
            p = self.tail_positions.get(tid, {})
            ov = fin_overlay(self.get_project().design, info, p)
            c.title = (pil_qimage(ov), 0, 0, info['W'], info['Hs'])        # drag anywhere on the fin
            hk = info['H'] * info['k']
            c.dx, c.dy = int(round(float(p.get('dx', 0)) * hk)), int(round(-float(p.get('dy', 0)) * hk))
            c.snap = max(4, int(hk * 0.04))
            c.update()
            return
        proj = self.get_project()
        d = proj.design
        d.title_pos = self.positions
        d.reg_pos = self.reg_positions
        if self.is_reg():
            t = TEMPLATES[tid].reg_preview(d, self.reg_text())
            if t is None:
                c.title = None
            else:
                mask, x, y = t
                col = Image.new('RGBA', mask.size, tuple(int(v) for v in d.rgb('reg_colour')) + (0,))
                col.putalpha(mask)
                c.title = (pil_qimage(col), x, y, mask.size[0], mask.size[1])
                c.snap = max(4, int(mask.size[1] * 0.25))
            p = self.store().get(tid, {})
            c.dx, c.dy = int(p.get('dx', 0)), int(p.get('dy', 0))
            c.update()
            return
        t = TEMPLATES[tid].title_preview(d)
        if t is None:
            c.title = None
        else:
            mask, tx, ty, cap = t
            r = max(1, int(round(cap * d.title_outline))) if d.title_outline > 0 else 0
            if r:                                           # room for the outline round the letters
                big = Image.new('L', (mask.size[0] + 2 * r, mask.size[1] + 2 * r), 0)
                big.paste(mask, (r, r)); mask = big; tx -= r; ty -= r
            col = Image.new('RGBA', mask.size, tuple(int(v) for v in d.rgb('title_colour')) + (0,))
            col.putalpha(mask)
            if r:
                from .engine.common import stroke_mask
                ring = Image.fromarray(stroke_mask(np.asarray(mask), r))
                outl = Image.new('RGBA', mask.size, tuple(int(v) for v in d.rgb('title_outline_colour')) + (0,))
                outl.putalpha(ring)
                col = Image.alpha_composite(outl, col)
            c.title = (pil_qimage(col), tx, ty, mask.size[0], mask.size[1])
            c.snap = max(6, int(cap * 0.12))
        p = self.positions.get(tid, {})
        c.dx, c.dy = int(p.get('dx', 0)), int(p.get('dy', 0))
        c.update()

    # ---------------- interaction
    def on_moved(self, dx, dy):
        p = self.pos()
        if self.is_tail():
            if not self.fin_info:
                return
            hk = self.fin_info['H'] * self.fin_info['k']
            p['dx'], p['dy'] = round(float(dx / hk), 4), round(float(-dy / hk), 4)
        else:
            p['dx'], p['dy'] = dx, dy
        self.refresh_title()

    def on_size(self, v):
        self.pos()['size'] = v / 100
        self.size_lab.setText(f'{v}%')
        self.refresh_title()
        if not self.size.isSliderDown():
            self.changed.emit()

    def on_angle(self, v):
        self.pos()['angle'] = float(v)
        self.angle_lab.setText(f'{v}°')
        self.refresh_title()
        if not self.angle.isSliderDown():
            self.changed.emit()

    def reset(self):
        self.store().pop(self.tid(), None)
        self.show_current()
        self.changed.emit()

    # ---------------- background
    def load_background(self):
        if self.thread and self.thread.isRunning():
            return
        tid = self.tid()
        proj = copy.deepcopy(self.get_project())
        t = TEMPLATES[tid]
        probs = t.check_source(proj.sources.get(tid, ''))
        if probs:
            self.canvas.bg = None
            self.canvas.message = f'{t.name}: ' + '; '.join(probs) + '. Set it on the Aircraft files tab.'
            self.canvas.update()
            return
        entry = next((e for e in proj.fleet if e.template == tid), None)
        entry = copy.deepcopy(entry) if entry else FleetEntry(tid, 'G-TEST', next(iter(t.variants)), '')
        entry.reg = entry.reg or 'G-TEST'
        proj.design.title_text = ''                          # the background has no titles…
        proj.design._editor_blank = True                     # …no registration…
        proj.design.tail_symbol = 'none'                     # …and no tail logo
        proj.output_dir = os.path.join(tempfile.gettempdir(), 'LiveryMakerTitles')
        self.canvas.bg = None
        self.canvas.message = f'Building the {t.name} side view…'
        self.canvas.update()
        self.load_btn.setEnabled(False)
        self.thread = QThread(self)
        self.worker = _BgWorker(proj, entry)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.log)
        self.worker.done.connect(lambda folder, err, tid=tid: self._loaded(tid, folder, err))
        self.worker.done.connect(self.thread.quit)
        self.thread.start()

    def _loaded(self, tid, folder, err):
        self.load_btn.setEnabled(True)
        if err:
            self.log.emit('Side view error:\n' + err)
            self.canvas.message = 'Could not build the side view. The details are in the Progress box.'
            self.canvas.update()
            return
        self.livery_dirs[tid] = folder
        if tid == self.tid():
            self.show_current()

    def busy(self):
        return bool(self.thread and self.thread.isRunning())
