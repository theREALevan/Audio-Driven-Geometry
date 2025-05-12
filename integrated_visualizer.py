import sys, os
import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPainter, QColor, QPen, QIntValidator
from PyQt5.QtCore import QUrl, QRectF
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QMediaPlaylist
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import math

###############################################################################
# Constants and Helper Functions
###############################################################################

# Constants
CANVAS_WIDTH = 400
CANVAS_HEIGHT = 400

def clean_points(points, eps=2):
    if not points:
        return points
    clean = [points[0]]
    for pt in points[1:]:
        if np.linalg.norm(np.array(pt) - np.array(clean[-1])) > eps:
            clean.append(pt)
    return clean

def transform_points(points, width, height, margin=10):
    points = np.array(points)
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()
    shape_width = max_x - min_x
    shape_height = max_y - min_y
    if shape_width == 0 or shape_height == 0:
        scale = 1
    else:
        scale = min((width - 2 * margin) / shape_width, (height - 2 * margin) / shape_height)
    offset_x = (width - scale * shape_width) / 2.0
    offset_y = (height - scale * shape_height) / 2.0
    transformed = np.column_stack(((points[:, 0] - min_x) * scale + offset_x,
                                height - ((points[:, 1] - min_y) * scale + offset_y)))
    return transformed

###############################################################################
# Drawing Canvas Class
###############################################################################

class DrawingCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(CANVAS_WIDTH, CANVAS_HEIGHT)
        self.strokes = []
        self.current_stroke = []
        self.bg_image = None
        self.bg_photo = None
        self.setMouseTracking(True)
        self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        if self.bg_photo:
            painter.drawImage(0, 0, self.bg_photo)
        else:
            painter.fillRect(self.rect(), QtCore.Qt.white)
        
        # Draw completed strokes
        for stroke in self.strokes:
            if len(stroke) > 1:
                points = [QtCore.QPointF(x, y) for x, y in stroke]
                painter.setPen(QPen(QtCore.Qt.black, 2))
                painter.drawPolyline(points)
        
        # Draw current stroke
        if self.current_stroke and len(self.current_stroke) > 1:
            points = [QtCore.QPointF(x, y) for x, y in self.current_stroke]
            painter.setPen(QPen(QtCore.Qt.red, 2))
            painter.drawPolyline(points)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = True
            if self.strokes:
                self.current_stroke = [self.strokes[-1][-1]]
            else:
                self.current_stroke = []
            self.current_stroke.append((event.x(), event.y()))
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.current_stroke.append((event.x(), event.y()))
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.drawing:
            self.drawing = False
            if len(self.current_stroke) > 1:
                self.strokes.append(self.current_stroke)
            self.current_stroke = []
            self.update()

    def load_background_image(self, filepath):
        try:
            image = Image.open(filepath)
            image = image.resize((CANVAS_WIDTH, CANVAS_HEIGHT), Image.LANCZOS)
            self.bg_image = image
            self.bg_photo = QImage(image.tobytes(), image.width, image.height, 
                                 image.width * 3, QImage.Format_RGB888)
            self.update()
        except Exception as e:
            print("Error loading image:", e)

    def delete_last_stroke(self):
        if self.strokes:
            self.strokes.pop()
            self.update()

    def get_total_points(self):
        total = []
        for stroke in self.strokes:
            if not total:
                total.extend(stroke)
            else:
                if stroke[0] != total[-1]:
                    total.append(stroke[0])
                total.extend(stroke)
        if self.current_stroke:
            if total and self.current_stroke[0] != total[-1]:
                total.append(self.current_stroke[0])
            total.extend(self.current_stroke)
        return clean_points(total)

    def clear(self):
        self.strokes = []
        self.current_stroke = []
        self.bg_image = None
        self.bg_photo = None
        self.update()

###############################################################################
# Fourier Selection Widget Class
###############################################################################

class FourierSelectionWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(CANVAS_WIDTH, CANVAS_HEIGHT)
        self.points = None
        self.F = None
        self.centroid = None
        self.num_coeffs = 20
        self.main_window = None
        self.rotation_angle = 0.0
        
        # Create UI elements
        layout = QtWidgets.QVBoxLayout(self)
        
        # Canvas for showing original and reconstructed shapes
        self.canvas = QtWidgets.QWidget()
        self.canvas.setMinimumSize(CANVAS_WIDTH, CANVAS_HEIGHT)
        layout.addWidget(self.canvas)
        
        # Slider for number of coefficients
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(2, 60)
        self.slider.setValue(20)
        self.slider.valueChanged.connect(self.update_reconstruction)
        layout.addWidget(self.slider)
        
        # Rotation slider
        rot_layout = QtWidgets.QHBoxLayout()
        self.rot_label = QtWidgets.QLabel("Rotation: 0°")
        self.rot_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.rot_slider.setRange(-180, 180)
        self.rot_slider.setValue(0)
        self.rot_slider.valueChanged.connect(self.on_rotation_changed)
        rot_layout.addWidget(self.rot_label)
        rot_layout.addWidget(self.rot_slider)
        layout.addLayout(rot_layout)

        # Add to Scene and Delete Selected buttons
        scene_buttons_layout = QtWidgets.QHBoxLayout()
        self.add_to_scene_button = QtWidgets.QPushButton("Add to Scene")
        self.add_to_scene_button.clicked.connect(self.add_to_scene)
        self.delete_selected_button = QtWidgets.QPushButton("Delete Selected")
        self.delete_selected_button.clicked.connect(self.delete_selected)
        scene_buttons_layout.addWidget(self.add_to_scene_button)
        scene_buttons_layout.addWidget(self.delete_selected_button)
        layout.addLayout(scene_buttons_layout)

    def on_rotation_changed(self, val):
        self.rotation_angle = float(val)
        self.rot_label.setText(f"Rotation: {val}°")
        self.update_reconstruction()

    def set_main_window(self, window):
        """Set reference to main window"""
        self.main_window = window

    def set_points(self, points):
        self.points = np.array(points)
        self.centroid = np.mean(self.points, axis=0)
        self.centered = self.points - self.centroid
        z = self.centered[:, 0] + 1j * self.centered[:, 1]
        self.F = np.fft.fft(z)
        self.rot_slider.setValue(0)
        self.update_reconstruction()

    def update_reconstruction(self):
        if self.F is None:
            return
            
        num_coeff = self.slider.value()
        F_filtered = np.zeros_like(self.F)
        F_filtered[0] = self.F[0]
        half = num_coeff // 2
        F_filtered[1:half+1] = self.F[1:half+1]
        F_filtered[-half:] = self.F[-half:]
        z_recon = np.fft.ifft(F_filtered)
        reconstructed = np.column_stack((z_recon.real, z_recon.imag)) + self.centroid
        # Apply rotation
        theta = np.radians(self.rotation_angle)
        if theta != 0.0:
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            reconstructed = (reconstructed - self.centroid) @ rot_matrix.T + self.centroid
        self.reconstructed_points = reconstructed
        self.update()

    def paintEvent(self, event):
        if self.points is None:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw original points
        pts_orig = transform_points(self.points, self.width(), self.height())
        painter.setPen(QPen(QtCore.Qt.gray, 1, QtCore.Qt.DashLine))
        for i in range(len(pts_orig)-1):
            p1 = QtCore.QPointF(float(pts_orig[i][0]), float(pts_orig[i][1]))
            p2 = QtCore.QPointF(float(pts_orig[i+1][0]), float(pts_orig[i+1][1]))
            painter.drawLine(p1, p2)
        
        # Draw reconstructed points
        if hasattr(self, 'reconstructed_points'):
            pts_recon = transform_points(self.reconstructed_points, self.width(), self.height())
            painter.setPen(QPen(QtCore.Qt.red, 2))
            for i in range(len(pts_recon)-1):
                p1 = QtCore.QPointF(float(pts_recon[i][0]), float(pts_recon[i][1]))
                p2 = QtCore.QPointF(float(pts_recon[i+1][0]), float(pts_recon[i+1][1]))
                painter.drawLine(p1, p2)

    def add_to_scene(self):
        if hasattr(self, 'reconstructed_points') and self.main_window:
            # Overwrite selected geometry if one is selected, else add new
            if (hasattr(self.main_window.shape_widget, 'selected_idx') and
                self.main_window.shape_widget.selected_idx is not None and
                0 <= self.main_window.shape_widget.selected_idx < len(self.main_window.shape_widget.geometries)):
                idx = self.main_window.shape_widget.selected_idx
                self.main_window.shape_widget.geometries[idx] = self.main_window.shape_widget._make_geometry(self.reconstructed_points)
                self.main_window.shape_widget.update()
            else:
                self.main_window.set_geometry(self.reconstructed_points)

    def delete_selected(self):
        if self.main_window and hasattr(self.main_window, 'shape_widget'):
            self.main_window.shape_widget.delete_selected_geometry()
            self.points = None
            self.F = None
            self.centroid = None
            self.reconstructed_points = None
            self.update()

###############################################################################
# Shape Widget Class
###############################################################################

class ShapeWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = None
        self.entries = []
        self.disturb = 2.0
        self.frame = 0
        self.geometries = []  # List of dicts: {shape, pos, scale}
        self.selected_idx = None
        self.buffer = QImage(800, 800, QImage.Format_ARGB32)
        self.buffer.fill(QtCore.Qt.black)
        self.main_window = None
        
        # Audio-related properties
        self.audio_duration = 0
        self.new_num_frames = 1024
        self.new_times = None
        self.centroid_interp = None
        self.c_min = 0
        self.c_max = 1
        
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.updateFromPlayer)
        self.timer.start(16)

        # Mouse interaction
        self.dragging = False
        self.drag_offset = None

        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def set_main_window(self, window):
        self.main_window = window

    def resizeEvent(self, ev):
        self.buffer = QImage(self.width(), self.height(), QImage.Format_ARGB32)
        self.buffer.fill(QtCore.Qt.black)

    def setPlayer(self, p):
        self.player = p

    def setDisturb(self, v):
        self.disturb = v

    def add_geometry(self, geometry):
        # Flip y-coordinates to match Qt's coordinate system
        geometry_flipped = geometry.copy()
        geometry_flipped[:, 1] = self.height() - geometry_flipped[:, 1]
        centroid = np.mean(geometry_flipped, axis=0)
        centered = geometry_flipped - centroid
        z0 = centered[:, 0] + 1j * centered[:, 1]
        F_base = np.fft.fft(z0)
        MAX_MODE = len(F_base) - 1
        geom = {
            'shape': geometry_flipped,
            'centroid': centroid,
            'centered': centered,
            'F_base': F_base,
            'MAX_MODE': MAX_MODE,
            'pos': [self.width()//2, self.height()//2],
            'scale': 1.0,
            'rotation': 0.0,
            'modulations': []
        }
        self.geometries.append(geom)
        self.selected_idx = len(self.geometries)-1
        self.update()

    def delete_selected_geometry(self):
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.geometries):
            del self.geometries[self.selected_idx]
            if self.geometries:
                self.selected_idx = min(self.selected_idx, len(self.geometries)-1)
            else:
                self.selected_idx = None
            self.buffer.fill(QtCore.Qt.black)
            self.update()

    def select_geometry_at(self, x, y):
        # Select geometry whose bounding box contains (x, y)
        for idx, geom in enumerate(self.geometries):
            pts = self._transformed_points(geom)
            minx, miny = pts.min(axis=0)
            maxx, maxy = pts.max(axis=0)
            if minx <= x <= maxx and miny <= y <= maxy:
                self.selected_idx = idx
                # Update modulation table
                if self.main_window:
                    self.main_window.update_modulation_table()
                self.update()
                return
        # If we get here, no geometry was selected
        self.selected_idx = None
        if self.main_window:
            self.main_window.update_modulation_table()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            x, y = event.x(), event.y()
            prev_selected = self.selected_idx
            self.select_geometry_at(x, y)
            self.setFocus()  # Ensure widget receives keyboard events
            # Deselect if click is on blank area
            found = False
            for idx, geom in enumerate(self.geometries):
                pts = self._transformed_points(geom)
                minx, miny = pts.min(axis=0)
                maxx, maxy = pts.max(axis=0)
                if minx <= x <= maxx and miny <= y <= maxy:
                    found = True
                    break
            if not found:
                self.selected_idx = None
                if self.main_window:
                    self.main_window.update_modulation_table()
                self.update()
            elif self.selected_idx is not None:
                geom = self.geometries[self.selected_idx]
                pts = self._transformed_points(geom)
                minx, miny = pts.min(axis=0)
                maxx, maxy = pts.max(axis=0)
                if minx <= x <= maxx and miny <= y <= maxy:
                    self.dragging = True
                    self.drag_offset = [x - geom['pos'][0], y - geom['pos'][1]]

    def mouseMoveEvent(self, event):
        if self.dragging and self.selected_idx is not None:
            x, y = event.x(), event.y()
            dx, dy = self.drag_offset
            self.geometries[self.selected_idx]['pos'] = [x - dx, y - dy]
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.dragging = False
            self.drag_offset = None

    def wheelEvent(self, event):
        if self.selected_idx is not None:
            delta = event.angleDelta().y()
            scale = self.geometries[self.selected_idx]['scale']
            scale *= 1.1 if delta > 0 else 0.9
            scale = max(0.1, min(10.0, scale))
            self.geometries[self.selected_idx]['scale'] = scale
            self.update()

    def _transformed_points(self, geom):
        # Apply scale, rotation, and position to centered shape
        pts = geom['centered'] * geom['scale']
        theta = math.radians(geom.get('rotation', 0.0))
        if theta != 0.0:
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            pts = pts @ rot_matrix.T
        pts = pts + np.array(geom['pos'])
        return pts

    def updateFromPlayer(self):
        if not self.geometries or self.audio_duration <= 0:
            return

        if self.player and self.player.state() == QMediaPlayer.PlayingState:
            t = (self.player.position()/1000.0) % self.audio_duration
            self.frame = int(t/self.audio_duration * self.new_num_frames)
        else:
            self.frame = (self.frame + 1) % self.new_num_frames

        # Choose color based on spectral centroid
        if self.centroid_interp is not None:
            cent = self.centroid_interp[self.frame]
            hue = int(((cent-self.c_min)/(self.c_max-self.c_min))*360) % 360
            col = QColor.fromHsv(hue, 255, 255, 200)
        else:
            col = QColor(255, 255, 255, 200)

        p = QPainter(self.buffer)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.buffer.rect(), QColor(0,0,0,15))

        w, h = self.width(), self.height()

        for idx, geom in enumerate(self.geometries):
            F = geom['F_base'].copy()
            # Apply modulations specific to this geometry
            for e in geom['modulations']:
                if e.mode > geom['MAX_MODE']:
                    continue
                a, ph = e.alpha[self.frame], e.phi[self.frame]
                delta = e.scale * a * np.exp(1j*ph)
                F[e.mode] += delta
                F[-e.mode] += np.conj(delta)

            rec = np.fft.ifft(F)
            pts = np.column_stack((rec.real, rec.imag))

            # Radial wobble
            if geom['modulations']:
                a0, p0 = geom['modulations'][0].alpha[self.frame], geom['modulations'][0].phi[self.frame]
            else:
                a0, p0 = 0, 0
            r = np.hypot(pts[:,0], pts[:,1])
            th = np.arctan2(pts[:,1], pts[:,0])
            rd = r + self.disturb * a0 * np.sin(12*th + p0)
            pts = np.column_stack((rd*np.cos(th), rd*np.sin(th)))

            # Apply scale and position
            pts = pts * geom['scale'] + np.array(geom['pos'])

            # Draw geometry
            poly = QtGui.QPolygonF()
            for x_,y_ in pts:
                poly.append(QtCore.QPointF(x_, y_))
            if idx == self.selected_idx:
                p.setPen(QPen(QColor(255,255,0,255), 3))  # Highlight selected
            else:
                p.setPen(QPen(col, 2))
            p.drawPolyline(poly)

            # Draw bounding box for selected
            if idx == self.selected_idx:
                minx, miny = pts.min(axis=0)
                maxx, maxy = pts.max(axis=0)
                p.setPen(QPen(QColor(255,0,0,180), 1, QtCore.Qt.DashLine))
                p.drawRect(QRectF(float(minx), float(miny), float(maxx - minx), float(maxy - miny)))

        p.end()
        self.update()

    def paintEvent(self, ev):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.buffer)
        painter.end()

    def set_audio_properties(self, duration, times, centroid_interp):
        """Set audio-related properties from main window"""
        self.audio_duration = duration
        self.new_times = times
        self.centroid_interp = centroid_interp
        if centroid_interp is not None:
            self.c_min = centroid_interp.min()
            self.c_max = centroid_interp.max()

    def keyPressEvent(self, event):
        if self.selected_idx is not None:
            geom = self.geometries[self.selected_idx]
            if event.key() == QtCore.Qt.Key_Q:
                geom['rotation'] -= 5.0
                self.update()
            elif event.key() == QtCore.Qt.Key_E:
                geom['rotation'] += 5.0
                self.update()
        super().keyPressEvent(event)

    def _make_geometry(self, geometry):
        geometry_flipped = geometry.copy()
        geometry_flipped[:, 1] = self.height() - geometry_flipped[:, 1]
        centroid = np.mean(geometry_flipped, axis=0)
        centered = geometry_flipped - centroid
        z0 = centered[:, 0] + 1j * centered[:, 1]
        F_base = np.fft.fft(z0)
        MAX_MODE = len(F_base) - 1
        return {
            'shape': geometry_flipped,
            'centroid': centroid,
            'centered': centered,
            'F_base': F_base,
            'MAX_MODE': MAX_MODE,
            'pos': [self.width()//2, self.height()//2],
            'scale': 1.0,
            'rotation': 0.0,
            'modulations': []  # Initialize empty modulations list
        }

###############################################################################
# Modulation Entry Class
###############################################################################

class ModulationEntry:
    def __init__(self, freq, mode, scale, main_window):
        self.freq = freq
        self.mode = int(mode)
        self.scale = scale
        self.main_window = main_window
        
        # Get the frequency index
        idx = np.argmin(np.abs(self.main_window.freqs - freq))
        
        # Get magnitude and phase for this frequency
        mag = uniform_filter1d(self.main_window.magnitude[idx], size=3)
        ph = uniform_filter1d(self.main_window.phase[idx], size=3)
        
        # Interpolate to new time points
        self.alpha = np.interp(self.main_window.new_times, 
                             self.main_window.times, mag)
        self.phi = np.interp(self.main_window.new_times, 
                           self.main_window.times, ph)

###############################################################################
# Main Window Class
###############################################################################

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio-Driven Geometry")
        
        # Configure default font
        default_font = QtGui.QFont()
        default_font.setPointSize(12)
        default_font.setBold(True)
        QtWidgets.QApplication.setFont(default_font)
        
        # Create central widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QHBoxLayout(central_widget)
        
        # Left side
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        
        # Drawing interface
        self.drawing_canvas = DrawingCanvas()
        self.drawing_canvas.setFixedHeight(self.screen().size().height() // 2)
        left_layout.addWidget(self.drawing_canvas)
        
        # Drawing controls
        drawing_controls = QtWidgets.QHBoxLayout()
        self.load_button = QtWidgets.QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        self.delete_button = QtWidgets.QPushButton("Delete Last")
        self.delete_button.clicked.connect(self.drawing_canvas.delete_last_stroke)
        self.done_drawing_button = QtWidgets.QPushButton("Done Drawing")
        self.done_drawing_button.clicked.connect(self.on_done_drawing)
        drawing_controls.addWidget(self.load_button)
        drawing_controls.addWidget(self.delete_button)
        drawing_controls.addWidget(self.done_drawing_button)
        left_layout.addLayout(drawing_controls)
        
        # Add Clear Canvas button below drawing controls
        self.clear_canvas_button = QtWidgets.QPushButton("Clear Canvas")
        self.clear_canvas_button.clicked.connect(self.drawing_canvas.clear)
        left_layout.addWidget(self.clear_canvas_button)
        
        # Add Load Geometry and Load Audio buttons
        file_buttons_layout = QtWidgets.QHBoxLayout()
        self.load_geom_button = QtWidgets.QPushButton("Load Geometry")
        self.load_geom_button.clicked.connect(self.load_geometry_file)
        self.load_audio_button = QtWidgets.QPushButton("Load Audio")
        self.load_audio_button.clicked.connect(self.load_audio_file)
        file_buttons_layout.addWidget(self.load_geom_button)
        file_buttons_layout.addWidget(self.load_audio_button)
        left_layout.addLayout(file_buttons_layout)
        
        # Fourier selection
        self.fourier_widget = FourierSelectionWidget()
        self.fourier_widget.set_main_window(self)
        self.fourier_widget.setVisible(False)
        left_layout.addWidget(self.fourier_widget)
        
        left_layout.addStretch(1)
        layout.addWidget(left_widget)
        
        # Right side
        self.animation_widget = QtWidgets.QWidget()
        layout.addWidget(self.animation_widget, stretch=2)
        
        # Initialize animation interface
        self.init_animation_interface()
        
        # Set window size
        self.resize(1200, 800)
        
        # Load default audio file
        self.load_default_audio()
        
        # Start in fullscreen
        self.showFullScreen()

    def load_image(self):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*.*)"
        )
        if filepath:
            self.drawing_canvas.load_background_image(filepath)

    def on_done_drawing(self):
        points = self.drawing_canvas.get_total_points()
        if len(points) < 4:
            QtWidgets.QMessageBox.warning(self, "Error", "Not enough points to reconstruct a shape.")
            return
        # Transform points to Cartesian coordinates
        transformed = [(x, CANVAS_HEIGHT - y) for (x, y) in points]
        if transformed and transformed[0] != transformed[-1]:
            transformed.append(transformed[0])
        # Show Fourier selection interface
        self.fourier_widget.setVisible(True)
        self.fourier_widget.set_points(transformed)

    def init_animation_interface(self):
        # Create layout for animation interface
        layout = QtWidgets.QVBoxLayout(self.animation_widget)
        
        # Create shape widget for animation
        self.shape_widget = ShapeWidget()
        self.shape_widget.set_main_window(self)  # Set main window reference
        layout.addWidget(self.shape_widget, stretch=2)
        
        # Create spectrogram widget
        self._initSpectrumPlot()
        layout.addWidget(self.spec_canvas, stretch=1)
        
        # Create time-frequency plot
        self._initTimeFreqPlot()
        layout.addWidget(self.time_canvas, stretch=1)
        
        # Add modulation table
        self._initModulationTable(layout)
        
        # Create controls
        controls_layout = QtWidgets.QHBoxLayout()
        
        # Add/Remove buttons
        self.btn_add = QtWidgets.QPushButton("Add Modulation")
        self.btn_rem = QtWidgets.QPushButton("Remove Modulation")
        controls_layout.addWidget(self.btn_add)
        controls_layout.addWidget(self.btn_rem)
        
        # Disturbance slider
        self.disturb_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.disturb_slider.setRange(1, 50)
        self.disturb_slider.setValue(20)
        self.disturb_slider.valueChanged.connect(
            lambda v: self.shape_widget.setDisturb(v/10))
        controls_layout.addWidget(QtWidgets.QLabel("Disturbance:"))
        controls_layout.addWidget(self.disturb_slider)
        
        layout.addLayout(controls_layout)
        
        # Connect signals
        self.btn_add.clicked.connect(self.add_modulation)
        self.btn_rem.clicked.connect(self.remove_modulation)
        
        # Initialize audio
        self.init_audio()

    def _initModulationTable(self, layout):
        self.table = QtWidgets.QTableWidget(0,3)
        self.table.setHorizontalHeaderLabels(["Freq","Mode","Scale"])
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked |
            QtWidgets.QAbstractItemView.SelectedClicked
        )
        self.table.cellChanged.connect(self.onCellChanged)
        
        # Set table font
        font = self.table.font()
        font.setPointSize(12)
        font.setBold(True)
        self.table.setFont(font)
        
        # Set header font
        header = self.table.horizontalHeader()
        header_font = header.font()
        header_font.setPointSize(12)
        header_font.setBold(True)
        header.setFont(header_font)
        
        layout.addWidget(self.table, stretch=0)

    def add_modulation(self):
        if not self.shape_widget.geometries or self.shape_widget.selected_idx is None:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select a geometry first.")
            return
        self._showAddDialog()

    def remove_modulation(self):
        if not self.shape_widget.geometries or self.shape_widget.selected_idx is None:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select a geometry first.")
            return
            
        idxs = self.table.selectedIndexes()
        if not idxs: return
        r = idxs[0].row()
        self.table.removeRow(r)
        selected_geom = self.shape_widget.geometries[self.shape_widget.selected_idx]
        del selected_geom['modulations'][r]

    def onCellChanged(self, row, col):
        if not self.shape_widget.geometries or self.shape_widget.selected_idx is None:
            return
            
        txt = self.table.item(row,col).text()
        selected_geom = self.shape_widget.geometries[self.shape_widget.selected_idx]
        entry = selected_geom['modulations'][row]
        try:
            if col==0:
                fr = float(txt); entry.freq=fr
                i  = np.argmin(np.abs(self.freqs-fr))
                mag= uniform_filter1d(self.magnitude[i],size=3)
                ph = uniform_filter1d(self.phase[i],   size=3)
                entry.alpha = np.interp(self.new_times,self.times,mag)
                entry.phi   = np.interp(self.new_times,self.times,ph)
            elif col==1:
                nm=int(txt)
                # Clamp mode to max_mode
                nm = min(nm, selected_geom['MAX_MODE'])
                entry.mode=nm
                self.table.blockSignals(True)
                self.table.item(row,col).setText(str(nm))
                self.table.blockSignals(False)
            else:
                entry.scale=float(txt)
        except ValueError:
            pass

    def _showAddDialog(self, preset_freq=None):
        if not self.shape_widget.geometries or self.shape_widget.selected_idx is None:
            return
            
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Add Modulation")
        form = QtWidgets.QFormLayout(dlg)
        
        freq_edit = QtWidgets.QLineEdit()
        mode_edit = QtWidgets.QLineEdit()
        scale_edit = QtWidgets.QLineEdit()
        
        if preset_freq is not None:
            freq_edit.setText(f"{preset_freq:.2f}")
        
        # Use the current geometry's max_mode for the validator
        selected_geom = self.shape_widget.geometries[self.shape_widget.selected_idx]
        max_mode = selected_geom['MAX_MODE']
        mode_edit.setValidator(QIntValidator(0, max_mode, self))
        
        form.addRow("Frequency:", freq_edit)
        form.addRow("Mode:", mode_edit)
        form.addRow("Scale:", scale_edit)
        
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        form.addWidget(buttons)
        
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            try:
                freq = float(freq_edit.text())
                mode = int(mode_edit.text())
                scale = float(scale_edit.text())
                # Clamp mode to max_mode
                mode = min(mode, max_mode)
                entry = ModulationEntry(freq, mode, scale, self)
                selected_geom['modulations'].append(entry)
                # Add to table
                r = self.table.rowCount()
                self.table.blockSignals(True)
                self.table.insertRow(r)
                for c,val in enumerate([freq,mode,scale]):
                    itm = QtWidgets.QTableWidgetItem(str(val))
                    itm.setFlags(itm.flags()|QtCore.Qt.ItemIsEditable)
                    self.table.setItem(r,c,itm)
                self.table.blockSignals(False)
            except ValueError:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Input", "Please enter valid numbers."
                )

    def _initSpectrumPlot(self):
        fig = Figure(figsize=(6,2.2))
        fig.subplots_adjust(bottom=0.25)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Plot will be updated when audio is loaded
        ax.set_xlabel("Frequency (Hz)", color='white', fontsize=14, fontweight='bold')
        ax.set_ylabel("Amplitude (dB)", color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='white', labelsize=12)
        for sp in ax.spines.values():
            sp.set_color('white')
        ax.grid(False)
        
        self.spec_canvas = FigureCanvas(fig)
        self.spec_ax = ax
        
        # Connect click event
        self.spec_canvas.mpl_connect('button_press_event', self.onSpectrumClick)

    def onSpectrumClick(self, event):
        if event.inaxes is not self.spec_ax:
            return
        if not self.shape_widget.geometries or self.shape_widget.selected_idx is None:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select a geometry first.")
            return
        f = event.xdata
        idx = np.argmin(np.abs(self.freqs - f))
        self._showAddDialog(self.freqs[idx])

    def _initTimeFreqPlot(self):
        fig = Figure(figsize=(6,2.2))
        fig.subplots_adjust(bottom=0.25)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Plot will be updated when audio is loaded
        ax.set_xlabel("Time (s)", color='white', fontsize=14, fontweight='bold')
        ax.set_ylabel("Frequency (Hz)", color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='white', labelsize=12)
        for sp in ax.spines.values():
            sp.set_color('white')
        
        # Playhead line
        self.time_ax = ax
        self.time_ax_line = ax.axvline(0, color='white', linewidth=1)
        
        canvas = FigureCanvas(fig)
        canvas.draw()
        self.tf_bg = canvas.copy_from_bbox(ax.bbox)
        canvas.mpl_connect('resize_event', self._onSpectrogramResize)
        
        self.time_canvas = canvas

    def _onSpectrogramResize(self, event):
        self.time_canvas.draw()
        self.tf_bg = self.time_canvas.copy_from_bbox(self.time_ax.bbox)

    def init_audio(self):
        # Initialize audio player with looping playlist
        self.playlist = QMediaPlaylist()
        self.player = QMediaPlayer()
        self.player.setVolume(50)
        self.player.setPlaylist(self.playlist)
        self.playlist.setPlaybackMode(QMediaPlaylist.Loop)
        
        # Connect player to shape widget
        self.shape_widget.setPlayer(self.player)
        
        # Connect player position updates
        self.player.positionChanged.connect(self._updateTimeLine)

    def _updateTimeLine(self, position):
        if not hasattr(self, 'audio_duration'):
            return
            
        t = (position/1000.0) % self.audio_duration
        self.time_ax_line.set_xdata([t,t])
        
        c = self.time_canvas
        c.restore_region(self.tf_bg)
        self.time_ax.draw_artist(self.time_ax_line)
        c.blit(self.time_ax.bbox)

    def load_audio(self, filepath):
        # Load and process audio file
        self.audio, self.sr = librosa.load(filepath, sr=None)
        self.audio_duration = len(self.audio) / self.sr
        
        # Compute STFT
        n_fft, hop_length = 2048, 512
        D = librosa.stft(self.audio, n_fft=n_fft, hop_length=hop_length)
        self.magnitude = np.abs(D)
        self.phase = np.angle(D)
        
        # Convert to dB
        self.S_db = librosa.amplitude_to_db(self.magnitude, ref=np.max)
        
        # Get frequencies and times
        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        self.times = librosa.frames_to_time(
            np.arange(self.magnitude.shape[1]), 
            sr=self.sr, 
            hop_length=hop_length, 
            n_fft=n_fft
        )
        
        # Compute spectral centroid
        centroid = librosa.feature.spectral_centroid(
            y=self.audio, sr=self.sr, n_fft=n_fft, hop_length=hop_length
        )[0]
        centroid_sm = uniform_filter1d(centroid, size=3)
        
        # Create new time points for interpolation
        self.new_num_frames = 1024
        self.new_times = np.linspace(0, self.audio_duration, self.new_num_frames)
        self.centroid_interp = np.interp(self.new_times, self.times, centroid_sm)
        
        # Update shape widget with audio properties
        self.shape_widget.set_audio_properties(
            self.audio_duration,
            self.new_times,
            self.centroid_interp
        )
        
        # Update plots
        self._update_plots()
        
        # Set up media player with looping playlist
        url = QUrl.fromLocalFile(os.path.abspath(filepath))
        self.playlist.clear()
        self.playlist.addMedia(QMediaContent(url))
        self.playlist.setPlaybackMode(QMediaPlaylist.Loop)
        self.player.setPlaylist(self.playlist)
        self.player.play()  # Always start playback after loading

    def _update_plots(self):
        # Update spectrum plot
        self.spec_ax.clear()
        
        # Compute average magnitude
        avg_mag = self.magnitude.mean(axis=1)
        
        # Find peaks
        threshold = avg_mag.max() * 0.03  # 3% of max as threshold
        peaks, _ = find_peaks(avg_mag, height=threshold, distance=5)
        if len(peaks) > 0:
            peak_freqs = self.freqs[peaks]
            freq_min = peak_freqs.min()
            freq_max = peak_freqs.max()
        else:
            freq_min = self.freqs.min()
            freq_max = self.freqs.max()
        
        # Plot spectrum
        self.spec_ax.plot(self.freqs, avg_mag, color='cyan', linewidth=1)
        self.spec_ax.set_xlim(freq_min, freq_max)
        self.spec_ax.set_xlabel("Frequency (Hz)", color='white', fontsize=14, fontweight='bold')
        self.spec_ax.set_ylabel("Amplitude", color='white', fontsize=14, fontweight='bold')
        self.spec_ax.tick_params(colors='white', labelsize=12)
        for sp in self.spec_ax.spines.values():
            sp.set_color('white')
        self.spec_ax.grid(False)
        self.spec_canvas.draw()
        
        # Update spectrogram with same frequency range
        self.time_ax.clear()
        im = self.time_ax.imshow(
            self.S_db, aspect='auto', origin='lower',
            extent=[0, self.audio_duration, freq_min, freq_max],
            cmap='magma', vmin=-60, vmax=0
        )
        self.time_ax.set_xlabel("Time (s)", color='white', fontsize=14, fontweight='bold')
        self.time_ax.set_ylabel("Frequency (Hz)", color='white', fontsize=14, fontweight='bold')
        self.time_ax.tick_params(colors='white', labelsize=12)
        for sp in self.time_ax.spines.values():
            sp.set_color('white')
        self.time_canvas.draw()
        self.tf_bg = self.time_canvas.copy_from_bbox(self.time_ax.bbox)

    def load_audio_file(self):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Audio", "", "Audio Files (*.wav *.mp3);;All Files (*.*)"
        )
        if filepath:
            self.load_audio(filepath)
            self.player.play()

    def load_default_audio(self):
        """Load the default audio file 'My_Song.wav'"""
        try:
            if os.path.exists("My_Song.wav"):
                self.load_audio("My_Song.wav")
                self.player.play()
            else:
                QtWidgets.QMessageBox.warning(self, "Missing Audio", "Default audio file 'My_Song.wav' not found.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Error loading default audio: {e}")

    def set_geometry(self, geometry):
        if hasattr(self, 'shape_widget'):
            self.shape_widget.add_geometry(geometry)
            self.shape_widget.show()
            self.shape_widget.update()
            # Update the modulation table for the newly added geometry
            self.update_modulation_table()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            QtWidgets.QApplication.quit()
        else:
            # Forward key events to shape widget for rotation
            if hasattr(self, 'shape_widget'):
                self.shape_widget.keyPressEvent(event)
            super().keyPressEvent(event)

    def load_geometry_file(self):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Geometry", "", "NPZ Files (*.npz);;All Files (*.*)"
        )
        if filepath:
            try:
                data = np.load(filepath)
                if 'refined_geometry' in data:
                    geometry = data['refined_geometry']
                elif 'geometry' in data:
                    geometry = data['geometry']
                else:
                    QtWidgets.QMessageBox.warning(self, "Invalid File", "No geometry found in file.")
                    return
                self.set_geometry(geometry)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load geometry: {e}")

    def update_modulation_table(self):
        if not self.shape_widget.geometries or self.shape_widget.selected_idx is None:
            self.table.clearContents()
            self.table.setRowCount(0)
            return
            
        geom = self.shape_widget.geometries[self.shape_widget.selected_idx]
        self.table.clearContents()
        self.table.setRowCount(len(geom['modulations']))
        for row, entry in enumerate(geom['modulations']):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{entry.freq:.2f}"))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(entry.mode)))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{entry.scale:.2f}"))

    def delete_selected_geometry(self):
        if self.shape_widget.selected_idx is not None and 0 <= self.shape_widget.selected_idx < len(self.shape_widget.geometries):
            del self.shape_widget.geometries[self.shape_widget.selected_idx]
            if self.shape_widget.geometries:
                self.shape_widget.selected_idx = min(self.shape_widget.selected_idx, len(self.shape_widget.geometries)-1)
            else:
                self.shape_widget.selected_idx = None
            self.shape_widget.buffer.fill(QtCore.Qt.black)  # Clear the buffer
            self.shape_widget.update()
            # Update the modulation table after deletion
            self.update_modulation_table()

###############################################################################
# Main Entry Point
###############################################################################

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 