#!/usr/bin/env python3
"""
Atomy Data Decoder
==================
IMU / GPS CSV visualizer with activity-segment timeline.

Requirements (all standard or pip-installable):
    pip install matplotlib numpy pandas
    tkinter is part of the Python standard library.
"""

import math
import os
from datetime import datetime, timezone

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ─── Decode constants ──────────────────────────────────────────────────────────
# User-specified:
ACC_SCALE     = 16 * 9.8 / 32768   # raw → m/s²
LAT_LON_SCALE = 1e-7                # lat/lon_e7 → degrees

# Reasonable defaults for common IMU chips (MPU-6050 / ICM-42688):
GYRO_SCALE = 250.0 / 32768         # raw → °/s   (±250 dps full-scale)
ATT_SCALE  = 180.0 / 32768         # raw → °      (±180° full-scale)


# ─── GCJ-02 → WGS-84 coordinate conversion ────────────────────────────────────
# Chinese devices often output GCJ-02 ("Mars coordinates") instead of WGS-84.
# The offset is ~200-500 m in China/HK. Apply this before exporting GPX.

_GCJ_A  = 6378245.0
_GCJ_EE = 0.00669342162296594323

def _in_china(lat: float, lon: float) -> bool:
    return 0.8293 <= lat <= 55.8271 and 72.004 <= lon <= 137.8347

def _gcj_transform_lat(x: float, y: float) -> float:
    ret = (-100.0 + 2.0*x + 3.0*y + 0.2*y*y + 0.1*x*y
           + 0.2*math.sqrt(abs(x)))
    ret += (20.0*math.sin(6.0*x*math.pi) + 20.0*math.sin(2.0*x*math.pi)) * 2/3
    ret += (20.0*math.sin(y*math.pi) + 40.0*math.sin(y/3.0*math.pi))      * 2/3
    ret += (160.0*math.sin(y/12.0*math.pi) + 320*math.sin(y*math.pi/30.0))* 2/3
    return ret

def _gcj_transform_lon(x: float, y: float) -> float:
    ret = (300.0 + x + 2.0*y + 0.1*x*x + 0.1*x*y
           + 0.1*math.sqrt(abs(x)))
    ret += (20.0*math.sin(6.0*x*math.pi) + 20.0*math.sin(2.0*x*math.pi)) * 2/3
    ret += (20.0*math.sin(x*math.pi) + 40.0*math.sin(x/3.0*math.pi))     * 2/3
    ret += (150.0*math.sin(x/12.0*math.pi) + 300.0*math.sin(x/30.0*math.pi))* 2/3
    return ret

def gcj02_to_wgs84(lat: float, lon: float) -> tuple[float, float]:
    """Convert GCJ-02 (Mars / 火星坐标) to WGS-84.
    Points outside China bounding box are returned unchanged."""
    if not _in_china(lat, lon):
        return lat, lon
    dlat = _gcj_transform_lat(lon - 105.0, lat - 35.0)
    dlon = _gcj_transform_lon(lon - 105.0, lat - 35.0)
    rad  = lat / 180.0 * math.pi
    magic = 1 - _GCJ_EE * math.sin(rad) ** 2
    sq    = math.sqrt(magic)
    dlat  = dlat * 180.0 / ((_GCJ_A * (1 - _GCJ_EE)) / (magic * sq) * math.pi)
    dlon  = dlon * 180.0 / (_GCJ_A / sq * math.cos(rad) * math.pi)
    # Inverse: WGS84 ≈ 2×GCJ02 − forward(GCJ02)
    return lat * 2 - (lat + dlat), lon * 2 - (lon + dlon)

# ─── Decoder ──────────────────────────────────────────────────────────────────
def decode_csv(path: str) -> pd.DataFrame:
    """Load a CSV and produce decoded / renamed columns alongside the raw ones."""
    df = pd.read_csv(path)

    # Accelerometer  →  m/s²
    for ax in ('ax', 'ay', 'az'):
        raw = f'raw_{ax}'
        if raw in df.columns:
            df[f'{ax} [m/s²]'] = df[raw] * ACC_SCALE

    # Gyroscope  →  °/s
    for ax in ('gx', 'gy', 'gz'):
        raw = f'raw_{ax}'
        if raw in df.columns:
            df[f'{ax} [°/s]'] = df[raw] * GYRO_SCALE

    # Attitude  →  °
    for ax in ('roll', 'pitch', 'yaw'):
        raw = f'raw_{ax}'
        if raw in df.columns:
            df[f'{ax} [°]'] = df[raw] * ATT_SCALE

    # GPS
    if 'lat_e7' in df.columns:
        df['latitude']  = df['lat_e7'] * LAT_LON_SCALE
    if 'lon_e7' in df.columns:
        df['longitude'] = df['lon_e7'] * LAT_LON_SCALE

    # Relative time in seconds
    df['time_s'] = (df['timestamp_ms'] - df['timestamp_ms'].iloc[0]) / 1000.0

    return df


# Column groups shown in the left panel (only existing columns are displayed)
COL_GROUPS = [
    ('Accelerometer', ['ax [m/s²]', 'ay [m/s²]', 'az [m/s²]']),
    ('Gyroscope',     ['gx [°/s]',  'gy [°/s]',  'gz [°/s]']),
    ('Attitude',      ['roll [°]',  'pitch [°]', 'yaw [°]']),
    ('GPS',           ['latitude',  'longitude']),
]

# Columns fed into the activity detector
IMU_DETECT_COLS = [
    'ax [m/s²]', 'ay [m/s²]', 'az [m/s²]',
    'gx [°/s]',  'gy [°/s]',  'gz [°/s]',
]

DEFAULT_CHECKED = {'ax [m/s²]', 'ay [m/s²]', 'az [m/s²]'}


# ─── IMU Activity Detector ────────────────────────────────────────────────────
def detect_segments(df: pd.DataFrame, n_blocks: int = 60,
                    threshold_factor: float = 2.0) -> list[dict]:
    """
    Split data into n_blocks equal-sized segments.
    Activity score = mean per-column std within the segment.
    Using std (not raw RMS) removes constant offsets like gravity,
    so only genuine motion/vibration raises the score.
    A segment is 'active' if score > median_score × threshold_factor.

    Returns a list of dicts: {start, end, score, active}
    """
    cols = [c for c in IMU_DETECT_COLS if c in df.columns]
    if not cols:
        return []

    data       = df[cols].values
    block_size = max(1, len(df) // n_blocks)

    segs = []
    for i in range(n_blocks):
        s = i * block_size
        e = min(s + block_size, len(df))
        if s >= len(df):
            break
        score = float(np.mean(np.std(data[s:e], axis=0)))
        segs.append({'start': s, 'end': e, 'rms': score, 'active': False})

    if not segs:
        return []

    median_score = np.median([seg['rms'] for seg in segs])
    threshold    = median_score * threshold_factor
    for seg in segs:
        seg['active'] = bool(seg['rms'] > threshold)

    return segs


# ─── GPX Writer (no external library needed) ──────────────────────────────────
def write_gpx(df: pd.DataFrame, path: str, convert_gcj02: bool = False) -> int:
    """Write GPS track to a GPX file. Returns the number of track points saved.

    Args:
        convert_gcj02: If True, convert GCJ-02 (火星坐标) → WGS-84 before writing.
    """
    gps = df.copy()
    if 'gps_fix' in gps.columns:
        gps = gps[gps['gps_fix'] == 1]
    gps = gps[gps['latitude'].notna() & gps['longitude'].notna()]

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="Atomy Data Decoder"',
        '     xmlns="http://www.topografix.com/GPX/1/1">',
        '  <trk><name>Track</name><trkseg>',
    ]
    for _, row in gps.iterrows():
        lat, lon = row['latitude'], row['longitude']
        if convert_gcj02:
            lat, lon = gcj02_to_wgs84(lat, lon)
        t = datetime.fromtimestamp(row['timestamp_ms'] / 1000.0, tz=timezone.utc)
        lines.append(
            f'    <trkpt lat="{lat:.7f}" lon="{lon:.7f}">'
            f'<time>{t.strftime("%Y-%m-%dT%H:%M:%SZ")}</time></trkpt>'
        )
    lines += ['  </trkseg></trk>', '</gpx>']

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return len(gps)


# ─── Detail Popup Window ──────────────────────────────────────────────────────
def open_detail(parent, df_slice: pd.DataFrame, cols: list[str], title: str):
    """Open a Toplevel window showing a segment's data in its own plot."""
    win = tk.Toplevel(parent)
    win.title(title)
    win.geometry('960x520')

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for col in cols:
        if col in df_slice.columns:
            ax.plot(df_slice['time_s'], df_slice[col], label=col, linewidth=0.9)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Value')
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.3)

    # Summary stats in the title area
    if cols:
        stats_parts = []
        for col in cols:
            if col in df_slice.columns:
                v = df_slice[col]
                stats_parts.append(f'{col}: μ={v.mean():.3g}  σ={v.std():.3g}')
        ax.set_title('\n'.join(stats_parts), fontsize=7, loc='left')

    fig.tight_layout()

    canvas  = FigureCanvasTkAgg(fig, master=win)
    toolbar = NavigationToolbar2Tk(canvas, win)
    toolbar.update()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()

    win.protocol('WM_DELETE_WINDOW', lambda: [plt.close(fig), win.destroy()])


# ─── Main Application ─────────────────────────────────────────────────────────
class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('Atomy Data Decoder')
        self.geometry('1440x900')
        self.minsize(900, 600)

        self.df: pd.DataFrame | None = None
        self.segments: list[dict]    = []
        self._suppress = False        # guard against recursive slider callbacks

        self._build_ui()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top toolbar ───────────────────────────────────────────────────────
        top = tk.Frame(self, bd=1, relief='raised', pady=3)
        top.pack(fill='x')

        tk.Button(top, text='📂  Load CSV', width=14,
                  command=self._load).pack(side='left', padx=6)
        tk.Button(top, text='💾  Export GPX', width=14,
                  command=self._export_gpx).pack(side='left', padx=2)

        self.lbl_file = tk.Label(top, text='No file loaded', fg='gray',
                                  font=('Courier', 9))
        self.lbl_file.pack(side='left', padx=12)

        self.lbl_rows = tk.Label(top, text='', fg='gray', font=('', 9))
        self.lbl_rows.pack(side='right', padx=10)

        # ── Main split: sidebar | chart area ──────────────────────────────────
        pane = tk.PanedWindow(self, orient='horizontal', sashwidth=4,
                               sashrelief='sunken')
        pane.pack(fill='both', expand=True)

        # Left sidebar
        sidebar = self._make_sidebar(pane)
        pane.add(sidebar, minsize=190, width=210)

        # Right chart + controls
        right = self._make_right(pane)
        pane.add(right, minsize=600)

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _make_sidebar(self, parent) -> tk.Frame:
        sb = tk.Frame(parent, bd=1, relief='sunken')

        tk.Label(sb, text='Plot Columns', font=('', 10, 'bold'),
                 pady=4).pack(fill='x')

        # Scrollable checkbox list
        wrap = tk.Frame(sb)
        wrap.pack(fill='both', expand=True)

        vsb = tk.Scrollbar(wrap, orient='vertical')
        vsb.pack(side='right', fill='y')

        self.col_canvas = tk.Canvas(wrap, yscrollcommand=vsb.set,
                                     highlightthickness=0)
        self.col_canvas.pack(side='left', fill='both', expand=True)
        vsb.config(command=self.col_canvas.yview)

        self.col_frame = tk.Frame(self.col_canvas)
        self._col_win  = self.col_canvas.create_window(
            (0, 0), window=self.col_frame, anchor='nw')
        self.col_frame.bind('<Configure>', self._on_col_frame_resize)
        self.col_canvas.bind('<Configure>', self._on_col_canvas_resize)

        self.checkboxes: dict[str, tk.BooleanVar] = {}

        # Bind mouse-wheel scrolling
        for w in (self.col_canvas, wrap):
            w.bind('<MouseWheel>',
                   lambda e: self.col_canvas.yview_scroll(
                       int(-e.delta / 120), 'units'))

        tk.Button(sb, text='▶  Plot Selected', pady=3,
                  command=self._plot).pack(fill='x', padx=6, pady=(4, 2))
        tk.Button(sb, text='Select All', pady=2,
                  command=lambda: self._toggle_all(True)
                  ).pack(fill='x', padx=6, pady=1)
        tk.Button(sb, text='Deselect All', pady=2,
                  command=lambda: self._toggle_all(False)
                  ).pack(fill='x', padx=6, pady=(1, 6))
        return sb

    def _on_col_frame_resize(self, _):
        self.col_canvas.configure(scrollregion=self.col_canvas.bbox('all'))

    def _on_col_canvas_resize(self, event):
        self.col_canvas.itemconfig(self._col_win, width=event.width)

    # ── Right panel ───────────────────────────────────────────────────────────

    def _make_right(self, parent) -> tk.Frame:
        right = tk.Frame(parent)

        # ── Matplotlib figure (data chart + timeline) ──────────────────────
        self.fig = plt.figure(figsize=(10, 6.5))
        gs = gridspec.GridSpec(
            2, 1, height_ratios=[5, 1.3], hspace=0.06,
            left=0.07, right=0.98, top=0.97, bottom=0.06)
        self.ax_data     = self.fig.add_subplot(gs[0])
        self.ax_timeline = self.fig.add_subplot(gs[1])

        self.canvas  = FigureCanvasTkAgg(self.fig, master=right)
        nav_bar      = NavigationToolbar2Tk(self.canvas, right)
        nav_bar.update()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)

        # ── Controls below the figure ──────────────────────────────────────
        ctrl = tk.Frame(right)
        ctrl.pack(fill='x', padx=6, pady=4)

        # Time range sliders
        rng = tk.LabelFrame(ctrl, text='Time Range', padx=4, pady=3)
        rng.pack(fill='x', pady=(0, 4))

        tk.Label(rng, text='Start:', width=6, anchor='e').grid(
            row=0, column=0, padx=(0, 4))
        self.sv_start = tk.DoubleVar(value=0.0)
        ttk.Scale(rng, from_=0.0, to=1.0, variable=self.sv_start,
                  orient='horizontal',
                  command=self._on_start_slide).grid(
            row=0, column=1, sticky='ew', padx=2)
        self.lbl_start = tk.Label(rng, text='0.00 s', width=10, anchor='w')
        self.lbl_start.grid(row=0, column=2, padx=4)

        tk.Label(rng, text='End:', width=6, anchor='e').grid(
            row=1, column=0, padx=(0, 4))
        self.sv_end = tk.DoubleVar(value=1.0)
        ttk.Scale(rng, from_=0.0, to=1.0, variable=self.sv_end,
                  orient='horizontal',
                  command=self._on_end_slide).grid(
            row=1, column=1, sticky='ew', padx=2)
        self.lbl_end = tk.Label(rng, text='0.00 s', width=10, anchor='w')
        self.lbl_end.grid(row=1, column=2, padx=4)
        rng.columnconfigure(1, weight=1)

        # Activity sensitivity slider
        sens = tk.LabelFrame(ctrl, text='Activity Detection Sensitivity',
                              padx=4, pady=3)
        sens.pack(fill='x')

        tk.Label(sens, text='Low', width=4).pack(side='left')
        self.sv_sens = tk.DoubleVar(value=2.0)
        ttk.Scale(sens, from_=1.0, to=5.0, variable=self.sv_sens,
                  orient='horizontal',
                  command=self._on_sens_slide).pack(
            side='left', fill='x', expand=True, padx=4)
        tk.Label(sens, text='High', width=4).pack(side='left')
        self.lbl_sens = tk.Label(sens, text='threshold ×2.0', width=14,
                                  anchor='w')
        self.lbl_sens.pack(side='left', padx=6)

        return right

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        path = filedialog.askopenfilename(
            title='Open CSV',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])
        if not path:
            return

        try:
            self.df = decode_csv(path)
        except Exception as exc:
            messagebox.showerror('Error', f'Failed to load CSV:\n{exc}')
            return

        self.lbl_file.config(text=os.path.basename(path), fg='black')
        self.lbl_rows.config(text=f'{len(self.df):,} rows  |  '
                                   f'{self.df["time_s"].iloc[-1]:.1f} s total')
        self._populate_cols()
        self._reset_sliders()
        self._update_segments()
        self._plot()

    def _populate_cols(self):
        for w in self.col_frame.winfo_children():
            w.destroy()
        self.checkboxes.clear()

        for group, cols in COL_GROUPS:
            present = [c for c in cols if c in self.df.columns]
            if not present:
                continue
            tk.Label(self.col_frame, text=group, font=('', 9, 'bold'),
                     anchor='w', fg='#333').pack(fill='x', padx=4,
                                                  pady=(8, 1))
            for col in present:
                var = tk.BooleanVar(value=(col in DEFAULT_CHECKED))
                cb  = tk.Checkbutton(
                    self.col_frame, text=col, variable=var,
                    anchor='w', command=self._plot)
                cb.pack(fill='x', padx=16)
                self.checkboxes[col] = var

    def _toggle_all(self, state: bool):
        for var in self.checkboxes.values():
            var.set(state)
        self._plot()

    # ── Slider helpers ────────────────────────────────────────────────────────

    def _frac_to_time(self, frac: float) -> float:
        if self.df is None or len(self.df) == 0:
            return 0.0
        idx = int(np.clip(frac, 0.0, 1.0) * (len(self.df) - 1))
        return float(self.df['time_s'].iloc[idx])

    def _frac_to_idx(self, frac: float) -> int:
        if self.df is None:
            return 0
        return int(np.clip(frac, 0.0, 1.0) * (len(self.df) - 1))

    def _reset_sliders(self):
        self._suppress = True
        self.sv_start.set(0.0)
        self.sv_end.set(1.0)
        self._suppress = False
        t_max = float(self.df['time_s'].iloc[-1])
        self.lbl_start.config(text='0.00 s')
        self.lbl_end.config(text=f'{t_max:.2f} s')

    def _on_start_slide(self, _=None):
        if self._suppress or self.df is None:
            return
        v = self.sv_start.get()
        if v >= self.sv_end.get() - 0.001:
            self._suppress = True
            self.sv_start.set(self.sv_end.get() - 0.001)
            self._suppress = False
            return
        self.lbl_start.config(text=f'{self._frac_to_time(v):.2f} s')
        self._plot()

    def _on_end_slide(self, _=None):
        if self._suppress or self.df is None:
            return
        v = self.sv_end.get()
        if v <= self.sv_start.get() + 0.001:
            self._suppress = True
            self.sv_end.set(self.sv_start.get() + 0.001)
            self._suppress = False
            return
        self.lbl_end.config(text=f'{self._frac_to_time(v):.2f} s')
        self._plot()

    def _on_sens_slide(self, _=None):
        if self.df is None:
            return
        factor = self.sv_sens.get()
        self.lbl_sens.config(text=f'threshold ×{factor:.1f}')
        self._update_segments()

    # ── Segment detection ─────────────────────────────────────────────────────

    def _update_segments(self):
        if self.df is None:
            return
        factor = self.sv_sens.get()
        self.segments = detect_segments(self.df, n_blocks=60,
                                         threshold_factor=factor)
        self._draw_timeline()

    # ── Timeline ──────────────────────────────────────────────────────────────

    def _draw_timeline(self):
        ax = self.ax_timeline
        ax.clear()
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=7)

        if not self.segments:
            self.canvas.draw_idle()
            return

        n = len(self.segments)
        for i, seg in enumerate(self.segments):
            color = '#e74c3c' if seg['active'] else '#2ecc71'
            alpha = 0.88 if seg['active'] else 0.55
            ax.barh(0, 1, left=i, color=color, edgecolor='white',
                    linewidth=0.3, height=0.8, alpha=alpha)

        # X-axis ticks aligned to real time
        tick_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
        tick_pos   = [frac * n for frac in tick_fracs]
        tick_lbls  = [f'{self._frac_to_time(frac):.0f}s'
                      for frac in tick_fracs]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbls)
        ax.set_xlim(0, n)
        ax.set_ylim(-0.5, 0.5)

        # Highlight current view range in the timeline
        si_frac = self.sv_start.get()
        ei_frac = self.sv_end.get()
        ax.axvspan(si_frac * n, ei_frac * n, color='#3498db', alpha=0.18,
                   label='current view')

        ax.set_xlabel(
            '🔴 High IMU activity  /  🟢 Normal  —  click any block to open detail',
            fontsize=8)

        self.canvas.draw_idle()

    # ── Canvas click: route to timeline or ignore ─────────────────────────────

    def _on_canvas_click(self, event):
        if event.inaxes is not self.ax_timeline:
            return
        if event.xdata is None or not self.segments:
            return

        idx = int(event.xdata)
        if not (0 <= idx < len(self.segments)):
            return

        seg = self.segments[idx]
        si, ei = seg['start'], seg['end']

        # Add context padding (50 % of block size on each side)
        pad = max(1, (ei - si) // 2)
        si  = max(0, si - pad)
        ei  = min(len(self.df), ei + pad)

        df_slice = self.df.iloc[si:ei]
        t0 = df_slice['time_s'].iloc[0]
        t1 = df_slice['time_s'].iloc[-1]
        tag = '⚠ HIGH ACTIVITY' if seg['active'] else 'normal'

        title = (f'Segment {idx}  —  t = {t0:.2f}s → {t1:.2f}s  '
                 f'[{tag}]  activity={seg["rms"]:.4g}')
        open_detail(self, df_slice, self._selected_cols(), title)

    # ── Main plot ─────────────────────────────────────────────────────────────

    def _selected_cols(self) -> list[str]:
        return [c for c, var in self.checkboxes.items() if var.get()]

    def _plot(self):
        if self.df is None:
            return

        cols = self._selected_cols()
        si   = self._frac_to_idx(self.sv_start.get())
        ei   = min(self._frac_to_idx(self.sv_end.get()) + 1, len(self.df))
        df_v = self.df.iloc[si:ei]

        ax = self.ax_data
        ax.clear()

        if cols and len(df_v) > 0:
            time = df_v['time_s']
            for col in cols:
                if col in df_v.columns:
                    ax.plot(time, df_v[col], label=col, linewidth=0.8)
            ax.legend(loc='upper right', fontsize=8, framealpha=0.7)

        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelbottom=False)

        # Redraw timeline to refresh the view-range indicator too
        self._draw_timeline()

    # ── GPX export ────────────────────────────────────────────────────────────

    def _export_gpx(self):
        if self.df is None:
            messagebox.showwarning('Warning', 'No data loaded.')
            return
        if 'latitude' not in self.df.columns:
            messagebox.showwarning('Warning',
                                   'No GPS data found in this file.\n'
                                   '(Need lat_e7 / lon_e7 columns.)')
            return

        # Ask about coordinate system
        ans = messagebox.askyesnocancel(
            'Coordinate System',
            'Does the GPS data use GCJ-02 (火星坐标)?\n\n'
            '  Yes  → Convert GCJ-02 → WGS-84 before saving\n'
            '  No   → Save as-is (already WGS-84)\n'
            '  Cancel → Abort\n\n'
            'Tip: Chinese devices typically output GCJ-02,\n'
            'which causes ~200-500 m offset on standard maps.')
        if ans is None:          # Cancel
            return
        convert = bool(ans)      # True = Yes (apply conversion)

        default = os.path.splitext(
            os.path.basename(self.lbl_file.cget('text')))[0] + '.gpx'
        path = filedialog.asksaveasfilename(
            initialfile=default,
            defaultextension='.gpx',
            filetypes=[('GPX files', '*.gpx'), ('All files', '*.*')])
        if not path:
            return

        try:
            n = write_gpx(self.df, path, convert_gcj02=convert)
            note = '(GCJ-02 → WGS-84 applied)' if convert else '(no conversion)'
            messagebox.showinfo('Exported',
                                f'Saved {n:,} GPS track points to:\n{path}\n{note}')
        except Exception as exc:
            messagebox.showerror('Error', str(exc))


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app = App()
    app.mainloop()
