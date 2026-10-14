"""Livery Maker - desktop app (PySide6)."""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from dataclasses import asdict

import hashlib
import json

from PySide6.QtCore import QObject, QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QIcon, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSlider, QSplitter, QTabWidget,
                               QTableWidget, QToolButton, QVBoxLayout, QWidget)

from .engine.common import BUILTIN_FONTS, Design, FleetEntry, Project
from .engine.export import build_all, build_one, validate, warnings
from .engine.templates import TEMPLATES
from . import viewer3d
from .title_editor import TitleEditor
from .ui_parts import Checklist, LiveryStrip, Section

APP = 'Livery Maker'
EXT = 'Livery Maker project (*.lmproj)'


def pil_to_pixmap(im) -> QPixmap:
    im = im.convert('RGBA')
    qi = QImage(im.tobytes('raw', 'RGBA'), im.size[0], im.size[1], QImage.Format_RGBA8888)
    return QPixmap.fromImage(qi.copy())


# ---------------------------------------------------------------- background work
class Worker(QObject):
    progress = Signal(str)
    aircraft_done = Signal(object, object)      # FleetEntry, {sheet: PIL image}
    finished = Signal(str)                      # '' on success, otherwise error text

    def __init__(self, project: Project, entry=None, make_zips=False):
        super().__init__()
        self.project, self.entry, self.make_zips = project, entry, make_zips
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            if self.entry is not None:                          # preview one aircraft into a temp folder
                prev = build_one(self.project, self.entry, self.progress.emit)
                self.aircraft_done.emit(self.entry, prev)
            else:
                build_all(self.project, self.progress.emit, lambda e, p: self.aircraft_done.emit(e, p),
                          cancelled=lambda: self._cancel, make_zips=self.make_zips)
            self.finished.emit('Cancelled' if self._cancel else '')
        except Exception:
            self.finished.emit(traceback.format_exc())


# ---------------------------------------------------------------- small widgets
class ColourButton(QPushButton):
    changed = Signal()

    def __init__(self, colour='#FFFFFF'):
        super().__init__()
        self.setFixedWidth(110)
        self.clicked.connect(self.pick)
        self.set(colour)

    def set(self, colour):
        self.colour = colour.upper()
        q = QColor(self.colour)
        fg = '#000000' if q.lightness() > 140 else '#FFFFFF'
        self.setText(self.colour)
        self.setStyleSheet(f'background:{self.colour}; color:{fg}; border:1px solid #888; border-radius:4px; padding:4px;')

    def pick(self):
        c = QColorDialog.getColor(QColor(self.colour), self, 'Choose colour')
        if c.isValid():
            self.set(c.name())
            self.changed.emit()


class PathEdit(QWidget):
    changed = Signal()

    def __init__(self, folder=True, filt=''):
        super().__init__()
        self.folder, self.filt = folder, filt
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(); self.edit.editingFinished.connect(self.changed)
        b = QPushButton('Choose…'); b.clicked.connect(self.browse)
        lay.addWidget(self.edit, 1); lay.addWidget(b)

    def browse(self):
        start = self.edit.text() or os.path.expanduser('~')
        p = (QFileDialog.getExistingDirectory(self, 'Choose folder', start) if self.folder
             else QFileDialog.getOpenFileName(self, 'Choose file', start, self.filt)[0])
        if p:
            self.edit.setText(p)
            self.changed.emit()

    def text(self):
        return self.edit.text().strip()

    def setText(self, t):
        self.edit.setText(t or '')


class FolderField(QWidget):
    """Compact folder picker: status dot, path (home shown as ~, hint as placeholder), a … button, drag-and-drop."""
    changed = Signal()
    COLOURS = {'ok': '#30B94D', 'warn': '#F0A020', 'bad': '#E5484D', 'off': '#8E8E93'}

    def __init__(self, placeholder='', tooltip=''):
        super().__init__()
        self.setAcceptDrops(True)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        self.dot = QLabel('●'); self.dot.setFixedWidth(14)
        self.edit = QLineEdit(); self.edit.setPlaceholderText(placeholder)
        self.edit.editingFinished.connect(self.changed)
        self.edit.setToolTip(tooltip)
        b = QPushButton('…'); b.setFixedWidth(34); b.setToolTip('Choose folder'); b.clicked.connect(self.browse)
        lay.addWidget(self.dot); lay.addWidget(self.edit, 1); lay.addWidget(b)
        self.tip = tooltip
        self.set_state('off', '')

    def set_state(self, state, detail):
        self.dot.setStyleSheet(f'color:{self.COLOURS[state]}; font-size:13px;')
        self.dot.setToolTip(detail)
        self.edit.setToolTip(self.tip + (f'\n\n{detail}' if detail else ''))

    def browse(self):
        start = self.text() or os.path.expanduser('~')
        p = QFileDialog.getExistingDirectory(self, 'Choose folder', start)
        if p:
            self.setText(p); self.changed.emit()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if p and os.path.isdir(p):
                self.setText(p); self.changed.emit(); break

    def text(self):
        return os.path.expanduser(self.edit.text().strip())

    def setText(self, t):
        t = t or ''
        home = os.path.expanduser('~')
        self.edit.setText('~' + t[len(home):] if home and t.startswith(home + os.sep) else t)
        self.edit.setCursorPosition(0)


def slider(lo, hi, val):
    s = QSlider(Qt.Horizontal); s.setRange(lo, hi); s.setValue(val)
    return s


# ---------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP)
        self.resize(1400, 880)
        self.settings = QSettings('LiveryMaker', 'LiveryMaker')
        self.path = None
        self.thread = None
        self.worker = None
        self.loading = False

        self.built_sigs = {}            # registration -> design signature of its last build (this session)
        self.last_preview_row = None    # fleet row of the last preview, for live refresh
        self.pending_preview = False
        self.previewed_once = False
        self.build_total = self.build_count = 0

        self.tabs = QTabWidget()
        self.tabs.addTab(self.build_design_tab(), 'Design')
        self.tabs.addTab(self.build_fleet_tab(), 'Fleet')
        self.tabs.addTab(self.build_sources_tab(), 'Aircraft files')

        # output: one compact row
        out = QWidget(); ol = QHBoxLayout(out); ol.setContentsMargins(0, 0, 0, 0)
        self.out_dir = FolderField('Folder for the finished liveries', 'Where Build all saves the livery folders.')
        self.out_dir.changed.connect(self.touch)
        self.zip_cb = QCheckBox('Also make .zips')
        self.zip_cb.setToolTip('Also make a .zip of each livery, handy for copying to the X-Plane PC')
        ol.addWidget(QLabel('Save liveries in')); ol.addWidget(self.out_dir, 1); ol.addWidget(self.zip_cb)

        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(8, 4, 4, 8)
        ll.addWidget(self.tabs, 1); ll.addWidget(out)

        # preview page: getting-started list until the first preview, then the preview tabs
        from PySide6.QtWidgets import QProgressBar, QStackedWidget
        self.preview_title = QLabel('Preview'); self.preview_title.setStyleSheet('font-weight:600; font-size:14px;')
        self.preview_tabs = QTabWidget()
        self.checklist = Checklist(); self.checklist.go.connect(self.checklist_go)
        self.preview_stack = QStackedWidget()
        prev_inner = QWidget(); pi = QVBoxLayout(prev_inner); pi.setContentsMargins(0, 0, 0, 0)
        pi.addWidget(self.preview_title); pi.addWidget(self.preview_tabs, 1)
        self.preview_stack.addWidget(self.checklist); self.preview_stack.addWidget(prev_inner)
        self.title_editor = TitleEditor(self.collect)
        self.title_editor.changed.connect(self.touch)
        self.title_editor.log.connect(self.logmsg)
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self.preview_stack, 'Preview')
        self.right_tabs.addTab(self.title_editor, 'Positions')
        self.right_tabs.currentChanged.connect(self.right_tab_changed)

        # progress: a slim bar and a one-line status; the full log folds away
        self.progress = QProgressBar(); self.progress.setMaximumHeight(14); self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        self.progress_lab = QLabel('Ready'); self.progress_lab.setStyleSheet('color:gray;')
        self.details_btn = QToolButton(); self.details_btn.setText('Details'); self.details_btn.setCheckable(True)
        self.details_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon); self.details_btn.setArrowType(Qt.RightArrow)
        self.details_btn.setStyleSheet('QToolButton { border:none; }')
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(160); self.log.setVisible(False)
        self.details_btn.toggled.connect(lambda on: (self.log.setVisible(on),
                                                     self.details_btn.setArrowType(Qt.DownArrow if on else Qt.RightArrow)))
        prow = QHBoxLayout(); prow.addWidget(self.progress_lab, 1); prow.addWidget(self.progress, 1); prow.addWidget(self.details_btn)
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(4, 4, 8, 8)
        rl.addWidget(self.right_tabs, 1); rl.addLayout(prow); rl.addWidget(self.log)

        split = QSplitter(); split.addWidget(left); split.addWidget(right); split.setSizes([620, 780])
        self.setCentralWidget(split)
        self.build_toolbar()
        self.build_menu()
        self.live_timer = QTimer(self); self.live_timer.setSingleShot(True); self.live_timer.setInterval(2500)
        self.live_timer.timeout.connect(self.live_refresh)
        self.status_timer = QTimer(self); self.status_timer.setSingleShot(True); self.status_timer.setInterval(300)
        self.status_timer.timeout.connect(self.refresh_status)
        last = self.settings.value('last_project', '')
        if last and os.path.exists(last):
            self.open_project(last)
        else:
            self.load_project(Project(design=Design()))
        self.statusBar().hide()

    # ---------------- UI construction
    def build_toolbar(self):
        from PySide6.QtWidgets import QToolBar
        tb = QToolBar('Actions'); tb.setMovable(False); tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(Qt.TopToolBarArea, tb)
        self.preview_act = QAction('▶  Preview', self); self.preview_act.setShortcut(QKeySequence('Ctrl+R'))
        self.preview_act.setToolTip('Preview the selected aircraft (⌘R)'); self.preview_act.triggered.connect(self.preview_selected)
        self.build_act = QAction('Build all', self); self.build_act.setShortcut(QKeySequence('Ctrl+B'))
        self.build_act.setToolTip('Build every livery in the fleet (⌘B)'); self.build_act.triggered.connect(self.build_everything)
        self.stop_act = QAction('Stop', self); self.stop_act.setShortcut(QKeySequence('Ctrl+.'))
        self.stop_act.setToolTip('Stop after the current aircraft (⌘.)'); self.stop_act.triggered.connect(self.cancel)
        self.stop_act.setEnabled(False)
        self.open_act = QAction('Open output folder', self); self.open_act.triggered.connect(self.open_output)
        for a in (self.preview_act, self.build_act, self.stop_act):
            tb.addAction(a)
        tb.addSeparator(); tb.addAction(self.open_act); tb.addSeparator()
        self.auto_cb = QCheckBox('Live preview')
        self.auto_cb.setToolTip('After a change, refresh the last preview by itself a few seconds later')
        self.auto_cb.setChecked(str(self.settings.value('live_preview', 'true')).lower() in ('true', '1'))
        self.auto_cb.toggled.connect(lambda on: self.settings.setValue('live_preview', on))
        tb.addWidget(self.auto_cb)

    def build_menu(self):
        m = self.menuBar().addMenu('File')
        for text, key, fn in (('New project', QKeySequence.New, self.new_project), ('Open project…', QKeySequence.Open, self.open_dialog),
                              ('Save project', QKeySequence.Save, self.save), ('Save project as…', QKeySequence.SaveAs, self.save_as)):
            a = QAction(text, self); a.setShortcut(key); a.triggered.connect(fn); m.addAction(a)
        b = self.menuBar().addMenu('Build')
        for a in (self.preview_act, self.build_act, self.stop_act, self.open_act):
            b.addAction(a)
        h = self.menuBar().addMenu('Help')
        a = QAction('About Livery Maker', self); a.triggered.connect(self.about); h.addAction(a)

    def build_design_tab(self):
        from PySide6.QtWidgets import QGridLayout
        w = QWidget(); outer = QVBoxLayout(w); outer.setContentsMargins(4, 4, 4, 4)
        self.strip = LiveryStrip()
        outer.addWidget(self.strip)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget(); scroll.setWidget(inner)
        lay = QVBoxLayout(inner); lay.setContentsMargins(4, 0, 4, 0)

        def form_box():
            box = QWidget(); f = QFormLayout(box); f.setContentsMargins(18, 0, 0, 6)
            f.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
            return box, f

        # airline & text
        box, f = form_box()
        self.airline = QLineEdit(); self.title_text = QLineEdit(); self.home_base = QLineEdit(); self.home_base.setMaxLength(4)
        self.title_text.setPlaceholderText('The words painted on the fuselage')
        self.title_font = QComboBox(); self.title_font.addItems(list(BUILTIN_FONTS) + ['Other font file…'])
        self.title_font_file = PathEdit(False, 'Fonts (*.ttf *.otf)'); self.title_font_file.setVisible(False)
        self.title_font.currentTextChanged.connect(lambda t: self.title_font_file.setVisible(t == 'Other font file…'))
        self.title_scale = slider(50, 100, 100)
        self.reg_font = QComboBox(); self.reg_font.addItems(list(BUILTIN_FONTS))
        f.addRow('Airline name', self.airline)
        f.addRow('Fuselage titles', self.title_text)
        f.addRow('Title font', self.title_font); f.addRow('', self.title_font_file)
        f.addRow('Title size', self.title_scale)
        f.addRow('Registration font', self.reg_font)
        f.addRow('Home base (ICAO)', self.home_base)
        lay.addWidget(Section('Airline and text', box, self.settings))

        # colours: a compact grid
        box = QWidget(); g = QGridLayout(box); g.setContentsMargins(18, 0, 0, 6); g.setHorizontalSpacing(10); g.setVerticalSpacing(6)
        self.c_body, self.c_accent, self.c_title, self.c_symbol, self.c_reg, self.c_outline = (ColourButton() for _ in range(6))
        for i, (label, btn) in enumerate((('Fuselage', self.c_body), ('Tail', self.c_accent), ('Titles', self.c_title),
                                          ('Registration', self.c_reg), ('Tail symbol', self.c_symbol))):
            g.addWidget(QLabel(label), i // 2, (i % 2) * 2); g.addWidget(btn, i // 2, (i % 2) * 2 + 1)
        self.outline_w = slider(0, 12, 0); self.outline_lab = QLabel(); self.outline_lab.setMinimumWidth(130)
        self.outline_w.valueChanged.connect(self.update_outline_label)
        g.addWidget(QLabel('Title outline'), 3, 0); g.addWidget(self.c_outline, 3, 1)
        orow = QHBoxLayout(); orow.addWidget(self.outline_w, 1); orow.addWidget(self.outline_lab)
        g.addLayout(orow, 3, 2, 1, 2)
        self.engine_col = QComboBox(); self.engine_col.addItems(['Tail colour', 'Fuselage colour'])
        self.tailplane_col = QComboBox(); self.tailplane_col.addItems(['Tail colour', 'Leave as original'])
        g.addWidget(QLabel('Engines'), 4, 0); g.addWidget(self.engine_col, 4, 1)
        g.addWidget(QLabel('Tailplane'), 4, 2); g.addWidget(self.tailplane_col, 4, 3)
        g.setColumnStretch(4, 1)
        lay.addWidget(Section('Colours', box, self.settings))

        # fade & weathering
        box, f = form_box()
        self.fade_shift = slider(-50, 50, 0); self.fade_len = slider(40, 200, 100)
        f.addRow('Fade: earlier ◀ ▶ later', self.fade_shift)
        f.addRow('Fade: sharp ◀ ▶ long', self.fade_len)
        self.weathering = slider(0, 100, 50)
        self.weathering_lab = QLabel(); self.weathering_lab.setStyleSheet('color:gray;')
        self.weathering.valueChanged.connect(self.update_weathering_label)
        self.weathering.setToolTip('ToLiss A319 and A321: uses the paintkit’s own dirt layers; the middle is the paintkit '
                                   'as supplied. The ERJ145, Q4XP and E190 keep the weathering of the livery they are built on.')
        f.addRow('Weathering: clean ◀ ▶ dirty', self.weathering); f.addRow('', self.weathering_lab)
        lay.addWidget(Section('Fade and weathering', box, self.settings))

        # tail
        box, f = form_box()
        self.tail_symbol = QComboBox(); self.tail_symbol.addItems(['Saltire', 'Image (PNG)', 'None'])
        self.tail_image = PathEdit(False, 'Images (*.png)')
        self.tail_image_size = slider(20, 90, 55)
        self.saltire_width = slider(6, 25, 14)
        self.winglet_cb = QCheckBox('Small saltire on winglets / sharklets')
        f.addRow('Symbol', self.tail_symbol); f.addRow('Image file', self.tail_image)
        f.addRow('Image size', self.tail_image_size); f.addRow('Saltire arm width', self.saltire_width)
        f.addRow('', self.winglet_cb)
        hint = QLabel('Move, resize and tilt it per aircraft on the Positions tab.'); hint.setStyleSheet('color:gray;')
        f.addRow('', hint)
        self.tail_symbol.currentIndexChanged.connect(self.update_tail_rows)
        lay.addWidget(Section('Tail', box, self.settings))
        lay.addStretch(1)
        outer.addWidget(scroll, 1)

        for wdg in (self.airline, self.title_text, self.home_base):
            wdg.textChanged.connect(self.touch)
        for wdg in (self.title_font, self.reg_font, self.engine_col, self.tailplane_col, self.tail_symbol):
            wdg.currentIndexChanged.connect(self.touch)
        for wdg in (self.title_scale, self.fade_shift, self.fade_len, self.tail_image_size, self.saltire_width, self.weathering):
            wdg.valueChanged.connect(self.touch)
        self.outline_w.valueChanged.connect(self.touch)
        for wdg in (self.c_body, self.c_accent, self.c_title, self.c_symbol, self.c_reg, self.c_outline, self.tail_image, self.title_font_file):
            wdg.changed.connect(self.touch)
        self.winglet_cb.toggled.connect(self.touch)
        return w

    def update_outline_label(self, *_):
        v = self.outline_w.value()
        self.outline_lab.setText('Off' if v == 0 else f'{v}% of letter height')
        self.c_outline.setEnabled(v > 0)

    def update_weathering_label(self, *_):
        v = self.weathering.value()
        txt = 'Clean' if v == 0 else 'As the paintkit comes' if v == 50 else f'{v}%  (lighter than supplied)' if v < 50 else f'{v}%  (dirtier than supplied)'
        self.weathering_lab.setText(txt)

    def update_tail_rows(self):
        mode = self.tail_symbol.currentIndex()
        self.tail_image.setEnabled(mode == 1); self.tail_image_size.setEnabled(mode == 1)
        self.saltire_width.setEnabled(mode == 0); self.winglet_cb.setEnabled(mode == 0)

    def build_fleet_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Aircraft', 'Registration', 'Engines / version', 'SELCAL', 'Status'])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in (1, 2, 3, 4):
            hh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.doubleClicked.connect(lambda idx: self.preview_selected() if idx.column() == 4 else None)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemChanged.connect(lambda it: self.touch() if it.column() != 4 else None)
        row = QHBoxLayout()
        for text, fn in (('Add aircraft', self.add_row), ('Duplicate', self.dup_row), ('Remove', self.remove_row)):
            b = QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
        row.addStretch(1)
        lay.addWidget(QLabel('One row per livery. Each registration becomes its own livery folder.'))
        lay.addWidget(self.table, 1); lay.addLayout(row)
        return w

    def build_sources_tab(self):
        from PySide6.QtWidgets import QGridLayout
        w = QWidget(); outer = QVBoxLayout(w)
        intro = QLabel('Point each aircraft at its paintkit or base livery. The 3D model is optional and turns on the '
                       '3D preview. Drag a folder onto a box or click … to choose one. Files are only read, never changed.')
        intro.setWordWrap(True); intro.setStyleSheet('color:palette(mid);')
        outer.addWidget(intro)
        grid = QGridLayout(); grid.setHorizontalSpacing(10); grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 3); grid.setColumnStretch(2, 2)
        for c, text in enumerate(('', 'Paintkit or base livery', '3D model (optional)')):
            h = QLabel(text); h.setStyleSheet('font-weight:600;'); grid.addWidget(h, 0, c)
        self.src_edits, self.src_status = {}, {}
        self.model_edits, self.model_status = {}, {}
        row = 1
        for tid, t in TEMPLATES.items():
            name = QLabel(t.name); name.setStyleSheet('font-weight:600;')
            name.setToolTip(t.source_hint + (f'. Base livery: {t.base_note}.' if t.base_note else '.'))
            if t.source_kind == 'paintkit':
                ph = f'{t.name} paintkit folder'
            else:
                ph = f'{t.base_note} folder' if t.base_note else 'Base livery folder'
            pe = FolderField(ph, t.source_hint + (f'. Base livery: {t.base_note}.' if t.base_note else '.'))
            pe.changed.connect(self.touch); pe.changed.connect(self.refresh_sources)
            grid.addWidget(name, row, 0); grid.addWidget(pe, row, 1)
            if t.model3d:
                me = FolderField('Aircraft folder', '3D preview: ' + t.model3d['hint'] + '.')
                me.changed.connect(self.touch); me.changed.connect(self.refresh_sources)
                grid.addWidget(me, row, 2)
                self.model_edits[tid] = me
            msg = QLabel(); msg.setWordWrap(True); msg.setVisible(False)
            grid.addWidget(msg, row + 1, 1, 1, 2)
            self.src_edits[tid], self.src_status[tid] = pe, msg
            row += 2
        outer.addLayout(grid)
        outer.addStretch(1)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(w)
        return scroll

    def refresh_sources(self):
        """Status dots, and a line under an aircraft only when something needs attention."""
        for tid, t in TEMPLATES.items():
            notes = []
            pe, p = self.src_edits[tid], self.src_edits[tid].text()
            if not p:
                pe.set_state('off', 'Not set')
            else:
                probs = t.check_source(p)
                warn = [] if probs else t.source_warnings(p)
                if probs:
                    pe.set_state('bad', '; '.join(probs)); notes.append(('bad', '; '.join(probs)))
                elif warn:
                    pe.set_state('warn', ' '.join(warn)); notes.append(('warn', ' '.join(warn)))
                else:
                    pe.set_state('ok', 'Ready')
            me = self.model_edits.get(tid)
            if me is not None:
                mp = me.text()
                if not mp:
                    me.set_state('off', 'Not set: no 3D tab for this aircraft')
                else:
                    mprobs = viewer3d.model_folder_problems(t.model3d, mp)
                    if mprobs:
                        me.set_state('bad', '; '.join(mprobs)); notes.append(('bad', '3D model: ' + '; '.join(mprobs)))
                    else:
                        me.set_state('ok', 'Ready for 3D preview')
            msg = self.src_status[tid]
            if notes:
                col = FolderField.COLOURS['bad' if any(k == 'bad' for k, _ in notes) else 'warn']
                msg.setText('  '.join(n for _, n in notes)); msg.setStyleSheet(f'color:{col}; font-size:11px;')
            msg.setVisible(bool(notes))

    # ---------------- fleet rows
    def add_row(self, entry: FleetEntry = None):
        if entry is None or isinstance(entry, bool):
            entry = FleetEntry('toliss_a319', '', next(iter(TEMPLATES['toliss_a319'].variants)), '')
        r = self.table.rowCount()
        self.table.insertRow(r)
        ac = QComboBox()
        for tid, t in TEMPLATES.items():
            ac.addItem(t.name, tid)
        ac.setCurrentIndex(list(TEMPLATES).index(entry.template))
        var = QComboBox()
        self._fill_variants(var, entry.template, entry.variant)
        ac.currentIndexChanged.connect(lambda _i, a=ac, v=var: (self._fill_variants(v, a.currentData(), ''), self.touch()))
        var.currentIndexChanged.connect(self.touch)
        from PySide6.QtWidgets import QTableWidgetItem
        self.table.setCellWidget(r, 0, ac)
        self.table.setItem(r, 1, QTableWidgetItem(entry.reg))
        self.table.setCellWidget(r, 2, var)
        self.table.setItem(r, 3, QTableWidgetItem(entry.selcal))
        st = QTableWidgetItem(''); st.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self.table.setItem(r, 4, st)
        self.touch()

    def _fill_variants(self, combo, tid, current):
        combo.blockSignals(True)
        combo.clear(); combo.addItems(list(TEMPLATES[tid].variants))
        if current in TEMPLATES[tid].variants:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    def dup_row(self):
        r = self.table.currentRow()
        if r >= 0:
            e = self.row_entry(r); e.reg = ''; self.add_row(e)

    def remove_row(self):
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r); self.touch()

    def row_entry(self, r) -> FleetEntry:
        reg = (self.table.item(r, 1).text() if self.table.item(r, 1) else '').strip().upper()
        sel = (self.table.item(r, 3).text() if self.table.item(r, 3) else '').strip().upper()
        return FleetEntry(self.table.cellWidget(r, 0).currentData(), reg, self.table.cellWidget(r, 2).currentText(), sel)

    # ---------------- project <-> widgets
    def collect(self) -> Project:
        font = self.title_font.currentText()
        if font == 'Other font file…':
            font = self.title_font_file.text() or 'Inter Display ExtraBold Italic'
        d = Design(
            airline=self.airline.text().strip() or 'My Airline', title_text=self.title_text.text(),
            title_font=font, title_scale=self.title_scale.value() / 100, reg_font=self.reg_font.currentText(),
            body_colour=self.c_body.colour, accent_colour=self.c_accent.colour, title_colour=self.c_title.colour,
            symbol_colour=self.c_symbol.colour, reg_colour=self.c_reg.colour,
            fade_shift=self.fade_shift.value() / 100, fade_length=self.fade_len.value() / 100,
            tail_symbol=['saltire', 'image', 'none'][self.tail_symbol.currentIndex()], tail_image=self.tail_image.text(),
            tail_image_size=self.tail_image_size.value() / 100, saltire_width=self.saltire_width.value() / 100,
            winglet_symbol=self.winglet_cb.isChecked(),
            engine_colour='accent' if self.engine_col.currentIndex() == 0 else 'body',
            tailplane_colour='accent' if self.tailplane_col.currentIndex() == 0 else 'original',
            home_base=self.home_base.text().strip().upper(), weathering=self.weathering.value() / 100,
            title_outline=self.outline_w.value() / 100, title_outline_colour=self.c_outline.colour,
            title_pos={k: dict(v) for k, v in self.title_editor.positions.items()} if hasattr(self, 'title_editor') else {},
            reg_pos={k: dict(v) for k, v in self.title_editor.reg_positions.items()} if hasattr(self, 'title_editor') else {},
            tail_pos={k: dict(v) for k, v in self.title_editor.tail_positions.items()} if hasattr(self, 'title_editor') else {})
        fleet = [self.row_entry(r) for r in range(self.table.rowCount())]
        sources = {tid: e.text() for tid, e in self.src_edits.items() if e.text()}
        models = {tid: e.text() for tid, e in self.model_edits.items() if e.text()}
        return Project(design=d, fleet=fleet, sources=sources, output_dir=self.out_dir.text(), models=models)

    def load_project(self, p: Project):
        self.loading = True
        d = p.design
        self.airline.setText(d.airline); self.title_text.setText(d.title_text); self.home_base.setText(d.home_base)
        if d.title_font in BUILTIN_FONTS:
            self.title_font.setCurrentText(d.title_font)
        else:
            self.title_font.setCurrentText('Other font file…'); self.title_font_file.setText(d.title_font)
        self.reg_font.setCurrentText(d.reg_font if d.reg_font in BUILTIN_FONTS else 'Inter Display Bold')
        self.title_scale.setValue(int(d.title_scale * 100))
        for b, c in ((self.c_body, d.body_colour), (self.c_accent, d.accent_colour), (self.c_title, d.title_colour),
                     (self.c_symbol, d.symbol_colour), (self.c_reg, d.reg_colour)):
            b.set(c)
        self.engine_col.setCurrentIndex(0 if d.engine_colour == 'accent' else 1)
        self.tailplane_col.setCurrentIndex(0 if d.tailplane_colour == 'accent' else 1)
        self.fade_shift.setValue(int(d.fade_shift * 100)); self.fade_len.setValue(int(d.fade_length * 100))
        self.weathering.setValue(int(round(d.weathering * 100))); self.update_weathering_label()
        self.c_outline.set(d.title_outline_colour); self.outline_w.setValue(int(round(d.title_outline * 100)))
        self.update_outline_label()
        self.tail_symbol.setCurrentIndex({'saltire': 0, 'image': 1, 'none': 2}.get(d.tail_symbol, 0))
        self.tail_image.setText(d.tail_image); self.tail_image_size.setValue(int(d.tail_image_size * 100))
        self.saltire_width.setValue(int(d.saltire_width * 100)); self.winglet_cb.setChecked(d.winglet_symbol)
        self.table.setRowCount(0)
        for e in p.fleet:
            self.add_row(e)
        for tid, e in self.src_edits.items():
            e.setText(p.sources.get(tid, ''))
        for tid, e in self.model_edits.items():
            e.setText(p.models.get(tid, ''))
        self.out_dir.setText(p.output_dir)
        self.update_tail_rows(); self.refresh_sources()
        self.title_editor.set_positions(d.title_pos, d.reg_pos, d.tail_pos)
        self.loading = False
        self.dirty = False
        self.last_preview_row = None
        self.update_title()
        self.refresh_status()

    def touch(self, *_):
        if not self.loading:
            self.dirty = True
            self.update_title()
            if hasattr(self, 'status_timer'):
                self.status_timer.start()
                if self.auto_cb.isChecked() and self.last_preview_row is not None:
                    self.live_timer.start()
            rt = getattr(self, 'right_tabs', None)
            if rt is not None and rt.currentWidget() is self.title_editor:
                self.title_editor.refresh_title()           # text, font, size or colour may have changed

    # ---------------- live bits: colour strip, fleet status, checklist, live preview
    def design_sig(self, proj: Project, entry: FleetEntry) -> str:
        d = asdict(proj.design)
        blob = json.dumps([d, proj.sources.get(entry.template, ''), asdict(entry)], sort_keys=True, default=str)
        return hashlib.md5(blob.encode()).hexdigest()

    def refresh_status(self):
        if self.loading:
            return
        proj = self.collect()
        self.strip.set_design(proj.design)
        ok_types = {tid for tid, t in TEMPLATES.items() if not t.check_source(proj.sources.get(tid, ''))}
        colours = {'ok': '#30B94D', 'warn': '#F0A020', 'bad': '#E5484D', 'off': '#8E8E93'}
        from PySide6.QtWidgets import QTableWidgetItem
        self.table.blockSignals(True)
        for r in range(self.table.rowCount()):
            e = self.row_entry(r)
            if e.template not in ok_types:
                state, text, tip = 'bad', 'Files missing', 'Set this aircraft’s files on the Aircraft files tab.'
            elif not e.reg:
                state, text, tip = 'warn', 'No registration', 'Add a registration.'
            else:
                sig = self.built_sigs.get(e.reg)
                folder = os.path.join(proj.output_dir, TEMPLATES[e.template].folder_name(proj.design, e)) if proj.output_dir else ''
                if sig == self.design_sig(proj, e):
                    state, text, tip = 'ok', 'Built', 'Built with the current design.'
                elif sig:
                    state, text, tip = 'warn', 'Changed', 'The design has changed since this was built. Build again.'
                elif folder and os.path.isdir(folder):
                    state, text, tip = 'off', 'Built earlier', 'A livery folder exists from an earlier session.'
                else:
                    state, text, tip = 'off', 'Ready', 'Ready to build.'
            it = self.table.item(r, 4)
            if it is None:
                it = QTableWidgetItem(); it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable); self.table.setItem(r, 4, it)
            it.setText('●  ' + text); it.setForeground(QColor(colours[state])); it.setToolTip(tip)
        self.table.blockSignals(False)
        od = self.out_dir.text()
        self.out_dir.set_state('ok' if od and os.path.isdir(od) else 'warn' if od else 'off',
                               'Ready' if od and os.path.isdir(od) else 'Will be created when you build' if od else 'Not set')
        # checklist
        self.checklist.set_done('files', bool(ok_types))
        self.checklist.set_done('design', bool(self.airline.text().strip()))
        self.checklist.set_done('fleet', self.table.rowCount() > 0)
        self.checklist.set_done('preview', self.previewed_once)
        self.preview_stack.setCurrentIndex(1 if self.previewed_once else 0)

    def checklist_go(self, key):
        if key == 'preview':
            self.preview_selected()
        else:
            self.tabs.setCurrentIndex({'design': 0, 'fleet': 1, 'files': 2}[key])

    def live_refresh(self):
        r = self.last_preview_row
        if r is None or r >= self.table.rowCount() or not self.auto_cb.isChecked():
            return
        if self.thread and self.thread.isRunning():
            self.pending_preview = True
            return
        self.table.selectRow(r)
        self.preview_selected(live=True)

    def right_tab_changed(self, _i):
        if self.right_tabs.currentWidget() is self.title_editor:
            r = self.table.currentRow()
            if r >= 0:
                self.title_editor.select(self.row_entry(r).template)
            self.title_editor.refresh_title()

    def update_title(self):
        name = os.path.basename(self.path) if self.path else 'Untitled'
        self.setWindowTitle(f"{name}{' •' if getattr(self, 'dirty', False) else ''} — {APP}")

    # ---------------- file menu
    def maybe_save(self) -> bool:
        if not getattr(self, 'dirty', False):
            return True
        r = QMessageBox.question(self, APP, 'Save changes to this project?', QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.save()
        return r == QMessageBox.Discard

    def new_project(self):
        if self.maybe_save():
            self.path = None
            self.load_project(Project(design=Design()))

    def open_dialog(self):
        if self.maybe_save():
            p, _ = QFileDialog.getOpenFileName(self, 'Open project', os.path.expanduser('~'), EXT)
            if p:
                self.open_project(p)

    def open_project(self, path):
        try:
            with open(path) as fh:
                proj = Project.from_json(fh.read())
        except Exception as e:
            QMessageBox.warning(self, APP, f'Could not open {path}:\n{e}')
            return
        self.path = path
        self.load_project(proj)
        self.settings.setValue('last_project', path)

    def save(self) -> bool:
        if not self.path:
            return self.save_as()
        with open(self.path, 'w') as fh:
            fh.write(self.collect().to_json())
        self.dirty = False; self.update_title()
        self.settings.setValue('last_project', self.path)
        self.progress_lab.setText(f'Saved {os.path.basename(self.path)}')
        return True

    def save_as(self) -> bool:
        p, _ = QFileDialog.getSaveFileName(self, 'Save project', os.path.join(os.path.expanduser('~'), f'{self.airline.text() or "Airline"}.lmproj'), EXT)
        if not p:
            return False
        if not p.endswith('.lmproj'):
            p += '.lmproj'
        self.path = p
        return self.save()

    def about(self):
        QMessageBox.about(self, APP, f'<b>{APP}</b><p>Makes X-Plane liveries from your own paintkits and base liveries.</p>'
                          '<p>Supported aircraft: ' + ', '.join(t.name for t in TEMPLATES.values()) + '.</p>'
                          '<p>Fonts: Inter, © The Inter Project Authors, SIL Open Font License 1.1.</p>')

    def closeEvent(self, ev):
        self.live_timer.stop(); self.pending_preview = False
        if (self.thread and self.thread.isRunning()) or self.title_editor.busy():
            QMessageBox.information(self, APP, 'Please wait for the current build to finish or press Stop.')
            ev.ignore(); return
        if self.maybe_save():
            ev.accept()
        else:
            ev.ignore()

    # ---------------- building
    def logmsg(self, m):
        self.log.appendPlainText(m)
        first = m.splitlines()[0] if m else ''
        self.progress_lab.setText(first if len(first) < 90 else first[:87] + '…')

    def start(self, worker: Worker):
        self.preview_tabs.clear()
        self.right_tabs.setCurrentIndex(0)
        self.thread = QThread(self)
        self.worker = worker
        worker.moveToThread(self.thread)
        self.thread.started.connect(worker.run)
        worker.progress.connect(self.logmsg)
        worker.aircraft_done.connect(self.show_preview)
        worker.finished.connect(self.done)
        worker.finished.connect(self.thread.quit)
        self.preview_act.setEnabled(False); self.build_act.setEnabled(False); self.stop_act.setEnabled(True)
        self.progress.setVisible(True)
        if worker.entry is not None:
            self.progress.setRange(0, 0)                         # busy: one aircraft
        else:
            self.build_total, self.build_count = len(worker.project.fleet), 0
            self.progress.setRange(0, max(1, self.build_total)); self.progress.setValue(0)
        self.thread.start()

    def done(self, err):
        self.preview_act.setEnabled(True); self.build_act.setEnabled(True); self.stop_act.setEnabled(False)
        self.progress.setVisible(False)
        if err == 'Cancelled':
            self.logmsg('Stopped.')
        elif err:
            self.logmsg('Error:\n' + err)
            self.details_btn.setChecked(True)
            QMessageBox.critical(self, APP, 'Something went wrong. The details are under Details, below the preview.')
        else:
            self.logmsg('Finished.' if self.worker is None or self.worker.entry is not None
                        else f'Finished: {self.build_count} liveries built.')
        self.refresh_status()
        if self.pending_preview:
            self.pending_preview = False
            self.live_timer.start()

    def show_preview(self, entry, prev):
        if self.worker is not None and self.worker.entry is None:            # Build all: count and remember
            self.build_count += 1
            self.progress.setValue(self.build_count)
            self.progress_lab.setText(f'Aircraft {self.build_count} of {self.build_total}: {entry.reg} done')
            self.built_sigs[entry.reg] = self.design_sig(self.built_proj, entry)
        keep = self.preview_tabs.tabText(self.preview_tabs.currentIndex()) if self.preview_tabs.count() else ''
        self.previewed_once = True
        self.preview_stack.setCurrentIndex(1)
        self.preview_tabs.clear()
        self.preview_title.setText(f'Preview — {entry.reg} ({TEMPLATES[entry.template].name}, {entry.variant})')
        if self.worker is not None and self.worker.entry is not None:      # previews only, not during Build all
            self.add_3d_tab(entry)
        side = max(300, min(self.preview_tabs.width() - 30, self.preview_tabs.height() - 60))
        for name, im in prev.items():
            lab = QLabel(); lab.setAlignment(Qt.AlignCenter)
            lab.setPixmap(pil_to_pixmap(im).scaled(side, side, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            sc = QScrollArea(); sc.setWidget(lab); sc.setWidgetResizable(True)
            self.preview_tabs.addTab(sc, name)
        for i in range(self.preview_tabs.count()):                         # stay on the tab you were looking at
            if self.preview_tabs.tabText(i) == keep:
                self.preview_tabs.setCurrentIndex(i); break

    def add_3d_tab(self, entry):
        proj = getattr(self, 'built_proj', None)
        t = TEMPLATES[entry.template]
        mdir = proj.models.get(entry.template, '') if proj else ''
        if not (t.model3d and mdir) or viewer3d.model_folder_problems(t.model3d, mdir):
            return
        livery = os.path.join(proj.output_dir, t.folder_name(proj.design, entry))
        panel = viewer3d.ModelPanel()
        panel.message.connect(self.logmsg)
        prev = getattr(self, 'panels_3d', [])
        if prev and prev[-1].view.parts and getattr(self, 'last_3d_type', None) == entry.template:
            v = prev[-1].view
            panel.keep_view = (v.yaw, v.pitch, v.zoom, v.pan.copy())
        self.last_3d_type = entry.template
        self.panels_3d = [p for p in getattr(self, 'panels_3d', []) if p.busy()] + [panel]   # keep loaders alive
        self.preview_tabs.addTab(panel, '3D')
        panel.load(t.model3d, mdir, livery, t.datarefs_for(entry), proj.sources.get(entry.template, ''))

    def check(self, proj: Project, need_output=True):
        probs = [p for p in validate(proj) if need_output or not p.startswith('Choose an output')]
        if not proj.fleet:
            probs.insert(0, 'Add at least one aircraft on the Fleet tab.')
        if probs:
            QMessageBox.warning(self, APP, 'Please fix these first:\n\n• ' + '\n• '.join(probs))
            return False
        w = warnings(proj)
        if w and QMessageBox.question(self, APP, 'Heads up:\n\n• ' + '\n• '.join(w) + '\n\nContinue anyway?') != QMessageBox.Yes:
            return False
        return True

    def preview_selected(self, live=False):
        if self.thread and self.thread.isRunning():
            return
        proj = self.collect()
        r = self.table.currentRow()
        if r < 0:
            if not proj.fleet:
                QMessageBox.information(self, APP, 'Add an aircraft on the Fleet tab first.'); return
            r = 0
        entry = self.row_entry(r)
        if not entry.reg:
            entry.reg = 'G-TEST'
        tid = entry.template
        probs = TEMPLATES[tid].check_source(proj.sources.get(tid, ''))
        if probs:
            QMessageBox.warning(self, APP, f'{TEMPLATES[tid].name}: ' + '; '.join(probs) + '\n\nSet it on the Aircraft files tab.')
            return
        proj.output_dir = os.path.join(tempfile.gettempdir(), 'LiveryMakerPreview')
        self.built_proj = proj
        self.last_preview_row = r
        self.logmsg(f'Updating the preview of {entry.reg}…' if live else
                    f'Previewing {entry.reg}… (the first aircraft of each type takes longer while its files load)')
        self.start(Worker(proj, entry))

    def build_everything(self):
        proj = self.collect()
        if not self.check(proj):
            return
        self.built_proj = proj
        self.logmsg(f'Building {len(proj.fleet)} liveries into {proj.output_dir}')
        self.start(Worker(proj, None, self.zip_cb.isChecked()))

    def cancel(self):
        if self.worker:
            self.worker.cancel()
            self.logmsg('Stopping after the current aircraft…')

    def open_output(self):
        p = self.out_dir.text()
        if p and os.path.isdir(p):
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))


def main():
    viewer3d.configure_default_format()
    app = QApplication(sys.argv)
    app.setApplicationName(APP)
    app.setOrganizationName('LiveryMaker')
    icon = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icon.png')
    if os.path.exists(icon):
        app.setWindowIcon(QIcon(icon))
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
