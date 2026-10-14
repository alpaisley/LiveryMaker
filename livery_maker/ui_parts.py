"""Small UI pieces for the main window: collapsible sections, the livery colour strip and the getting-started list."""
from __future__ import annotations

import os

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QFontDatabase, QLinearGradient, QPainter, QPainterPath, QPen,
                           QPolygonF)
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from .engine.common import font_path

_FAMILIES: dict = {}


def qt_family(font_name: str) -> str:
    """Register a built-in or chosen font file with Qt once and return its family name."""
    p = font_path(font_name)
    if p not in _FAMILIES:
        fam = ''
        if os.path.exists(p):
            fid = QFontDatabase.addApplicationFont(p)
            fams = QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
            fam = fams[0] if fams else ''
        _FAMILIES[p] = fam
    return _FAMILIES[p]


class Section(QWidget):
    """A heading you click to fold its contents away. Remembers whether it was open."""

    def __init__(self, title: str, content: QWidget, settings=None, start_open=True):
        super().__init__()
        self.settings, self.key = settings, f'section/{title}'
        is_open = start_open
        if settings is not None:
            v = settings.value(self.key, None)
            if v is not None:
                is_open = str(v).lower() in ('true', '1')
        self.head = QToolButton()
        self.head.setText(title)
        self.head.setCheckable(True)
        self.head.setChecked(is_open)
        self.head.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.head.setStyleSheet('QToolButton { border:none; font-weight:600; font-size:13px; padding:4px 0; }')
        self.head.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.body = content
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 2, 0, 2); lay.setSpacing(2)
        lay.addWidget(self.head); lay.addWidget(self.body)
        self.head.toggled.connect(self._toggle)
        self._toggle(is_open)

    def _toggle(self, on):
        self.head.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.body.setVisible(on)
        if self.settings is not None:
            self.settings.setValue(self.key, on)


class LiveryStrip(QWidget):
    """A small picture of the design: fuselage colour fading into the tail colour, the titles (with their outline),
    the registration and a fin with the tail symbol. Redrawn whenever the design changes."""

    def __init__(self):
        super().__init__()
        self.design = None
        self.setMinimumHeight(74)
        self.setMaximumHeight(90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_design(self, d):
        self.design = d
        self.update()

    def paintEvent(self, _e):
        d = self.design
        if d is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        body, acc = QColor(d.body_colour), QColor(d.accent_colour)
        fus = QRectF(6, H * 0.42, W - 12, H * 0.42)
        # fade: start/finish follow the design's fade sliders, roughly
        mid = 0.66 + 0.18 * d.fade_shift
        half = 0.12 * max(0.2, d.fade_length)
        g = QLinearGradient(fus.left(), 0, fus.right(), 0)
        g.setColorAt(0, body); g.setColorAt(max(0.0, mid - half), body)
        g.setColorAt(min(1.0, mid + half), acc); g.setColorAt(1, acc)
        path = QPainterPath(); path.addRoundedRect(fus, fus.height() / 2, fus.height() / 2)
        p.setPen(QPen(QColor(0, 0, 0, 60), 1)); p.setBrush(QBrush(g)); p.drawPath(path)
        # fin: swept, rising from the top of the rear fuselage
        fh = fus.height()
        root_le, root_te = fus.right() - fh * 2.9, fus.right() - fh * 0.35
        top = 3.0
        fin = QPolygonF([QPointF(root_le, fus.top() + 1), QPointF(root_le + fh * 1.55, top),
                         QPointF(root_le + fh * 2.25, top), QPointF(root_te, fus.top() + 1)])
        fx = root_le
        p.setBrush(acc); p.setPen(QPen(QColor(0, 0, 0, 60), 1)); p.drawPolygon(fin)
        if d.tail_symbol == 'saltire':
            p.save()
            clip = QPainterPath(); clip.addPolygon(fin); p.setClipPath(clip)
            pen = QPen(QColor(d.symbol_colour), max(2.0, fus.height() * 0.24)); p.setPen(pen)
            b = fin.boundingRect()
            p.drawLine(b.topLeft(), b.bottomRight()); p.drawLine(b.bottomLeft(), b.topRight())
            p.restore()
        elif d.tail_symbol == 'image':
            p.setPen(Qt.NoPen); p.setBrush(QColor(255, 255, 255, 170))
            c = fin.boundingRect().center(); r = fus.height() * 0.32
            p.drawEllipse(c, r, r)
        # titles
        text = (d.title_text or d.airline or '').strip()
        if text:
            f = QFont(qt_family(d.title_font) or self.font().family())
            f.setPixelSize(int(fus.height() * 0.62)); f.setBold(True)
            tp = QPainterPath()
            tp.addText(0, 0, f, text)
            br = tp.boundingRect()
            max_w = (fx - fus.left()) * 0.78
            if br.width() > max_w:
                f.setPixelSize(max(6, int(f.pixelSize() * max_w / br.width())))
                tp = QPainterPath(); tp.addText(0, 0, f, text); br = tp.boundingRect()
            dx = fus.left() + (fx - fus.left()) * 0.18 - br.left()
            dy = fus.center().y() - br.center().y()
            tp.translate(dx, dy)
            if d.title_outline > 0:
                w = max(1.5, br.height() * d.title_outline * 2.2)
                p.setPen(QPen(QColor(d.title_outline_colour), w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                p.setBrush(Qt.NoBrush); p.drawPath(tp)
            p.setPen(Qt.NoPen); p.setBrush(QColor(d.title_colour)); p.drawPath(tp)
        # registration
        f = QFont(qt_family(d.reg_font) or self.font().family()); f.setPixelSize(max(7, int(fus.height() * 0.2)))
        p.setFont(f); p.setPen(QColor(d.reg_colour))
        p.drawText(QRectF(fx - fus.height() * 2.2, fus.bottom() - fus.height() * 0.42, fus.height() * 2, fus.height() * 0.4),
                   Qt.AlignRight | Qt.AlignVCenter, 'G-XXXX')
        p.end()


class Checklist(QWidget):
    """Getting started: three steps with ticks and a button each, shown until the first preview."""
    go = Signal(str)        # 'files', 'design', 'fleet', 'preview'

    STEPS = (('files', 'Set your aircraft files',
              'Point at least one aircraft at its paintkit or base livery on the Aircraft files tab.'),
             ('design', 'Set your design',
              'Airline name, titles, colours and tail symbol on the Design tab.'),
             ('fleet', 'Add your fleet',
              'One row per aircraft on the Fleet tab, with its registration and engines.'),
             ('preview', 'Preview an aircraft',
              'Select a row on the Fleet tab and press Preview to see it, in 3D if you have set its model.'))

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(24, 24, 24, 24); lay.setSpacing(14)
        t = QLabel('Getting started'); t.setStyleSheet('font-size:18px; font-weight:600;')
        lay.addWidget(t)
        self.rows = {}
        for key, title, help_ in self.STEPS:
            row = QWidget(); h = QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(12)
            tick = QLabel('○'); tick.setFixedWidth(22); tick.setStyleSheet('font-size:18px; color:#8E8E93;')
            txt = QLabel(f'<b>{title}</b><br><span style="color:gray">{help_}</span>'); txt.setWordWrap(True)
            btn = QPushButton('Go'); btn.setFixedWidth(70); btn.clicked.connect(lambda _=False, k=key: self.go.emit(k))
            h.addWidget(tick); h.addWidget(txt, 1); h.addWidget(btn)
            lay.addWidget(row)
            self.rows[key] = tick
        lay.addStretch(1)

    def set_done(self, key, done):
        tick = self.rows[key]
        tick.setText('✓' if done else '○')
        tick.setStyleSheet(f'font-size:18px; color:{"#30B94D" if done else "#8E8E93"};')
