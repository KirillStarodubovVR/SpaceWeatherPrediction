#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация результатов sat_coords.py: Земля (3D) с орбитой и положением КА,
а также трасса на карте.

    from plot_orbit import plot_satellite, plot_ground_track
    plot_satellite("R3D21_xyz.csv", time="2026-10-12T09:00:00Z", out="earth.png")
    plot_ground_track("R3D21_xyz.csv", time="2026-10-12T09:00:00Z", out="track.png")

или из командной строки:

    python plot_orbit.py R3D21_xyz.csv --time 2026-10-12T09:00:00Z --out earth.png --track track.png

Контуры материков: Natural Earth (ne_110m_coastline.geojson, общественное достояние).
Файл ищется рядом со скриптом и в текущей папке; если его нет, один раз скачивается
с GitHub. Без него Земля рисуется только с сеткой координат.

Зависимости: numpy, pandas, matplotlib, pyerfa
"""
import argparse
import json
import math
import os
import urllib.request

import numpy as np
import pandas as pd
import erfa
import matplotlib
import matplotlib.pyplot as plt

A_WGS84 = 6378.137                  # км
B_WGS84 = 6356.752314245            # км
MU = 398600.4415

COAST_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
             "master/geojson/ne_110m_coastline.geojson")
COAST_FILE = "ne_110m_coastline.geojson"

# цвета (светлая тема)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
EARTH = "#dfe7ef"
GRID = "#a9b4bf"
COAST = "#6b7684"
ORBIT = "#2a78d6"                   # орбита / трасса
SAT = "#eb6834"                     # КА


# --------------------------------------------------------------------------- данные
def load_coastlines(coastline="auto"):
    """Список массивов (N,2) [lon, lat] в градусах или [] если данных нет."""
    if coastline is None or coastline is False:
        return []
    path = None
    if coastline != "auto":
        path = coastline
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in (os.path.join(here, COAST_FILE), os.path.join(os.getcwd(), COAST_FILE)):
            if os.path.exists(cand):
                path = cand
                break
        if path is None:
            try:
                path = os.path.join(here, COAST_FILE)
                urllib.request.urlretrieve(COAST_URL, path)
            except Exception as exc:                       # нет сети -- рисуем без материков
                print(f"Контуры материков недоступны ({exc}); рисую без них.")
                return []
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    lines = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        parts = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
        lines += [np.asarray(p, dtype=float) for p in parts]
    return lines


def _pick_index(df, time):
    if time is None:
        return len(df) - 1
    if isinstance(time, (int, float, np.floating)):
        return int(np.argmin(np.abs(df["t_s"].to_numpy() - float(time))))
    ts = pd.to_datetime(df["time_utc"], utc=True, format="ISO8601")
    target = pd.Timestamp(time)
    target = target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    return int(np.argmin(np.abs((ts - target).dt.total_seconds().to_numpy())))


def _cols(frame, kind="r"):
    f = frame.lower()
    if kind == "r":
        return [f"x_{f}_km", f"y_{f}_km", f"z_{f}_km"]
    return [f"vx_{f}_kms", f"vy_{f}_kms", f"vz_{f}_kms"]


def geodetic(r_itrf_km):
    """ITRF (км) -> долгота, широта (град), высота (км) над эллипсоидом WGS-84."""
    lon, lat, h = erfa.gc2gd(1, np.asarray(r_itrf_km, dtype=float) * 1e3)
    return np.degrees(lon), np.degrees(lat), np.asarray(h) / 1e3


def itrf_to_gcrf_matrix(time_utc, dut1=0.0):
    """Матрица M: r_gcrf = M @ r_itrf (без движения полюса), как в sat_coords.py."""
    t = pd.Timestamp(time_utc)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    u1, u2 = erfa.dtf2d("UTC", t.year, t.month, t.day, t.hour, t.minute,
                        t.second + t.microsecond * 1e-6)
    tt1, tt2 = erfa.taitt(*erfa.utctai(u1, u2))
    C = erfa.c2i06a(tt1, tt2)                          # GCRF -> CIRS
    era = erfa.era00(*erfa.utcut1(u1, u2, dut1))
    c, s = math.cos(era), math.sin(era)
    R3 = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])   # CIRS -> ITRF
    return C.T @ R3.T


def _ll_to_xyz(lon_deg, lat_deg, h=0.0):
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return np.stack([(A_WGS84 + h) * np.cos(lat) * np.cos(lon),
                     (A_WGS84 + h) * np.cos(lat) * np.sin(lon),
                     (B_WGS84 + h) * np.sin(lat)], axis=-1)


# --------------------------------------------------------------------------- 3D
def _hide(points, view, on_surface):
    """Скрывает (NaN) точки, закрытые Землёй для наблюдателя в направлении view."""
    p = points.copy()
    along = p @ view
    if on_surface:
        hidden = along < 0
    else:
        perp = np.linalg.norm(p - np.outer(along, view), axis=1)
        hidden = (along < 0) & (perp < A_WGS84)
    p[hidden] = np.nan
    return p


def plot_satellite(csv_path, time=None, frame="itrf", past_rev=1.0, future_rev=0.5,
                   coastline="auto", title=None, out=None, elev=None, azim=None,
                   ax=None, dut1=0.0):
    """Рисует Землю, участок орбиты и положение КА на момент `time`.

    csv_path   -- файл, созданный sat_coords.py (или DataFrame с теми же столбцами)
    time       -- момент: строка ISO UTC, секунды от начала (t_s) или None (последняя точка)
    frame      -- 'itrf' (Земля неподвижна, видны материки под КА) или 'gcrf' (инерциальная)
    past_rev   -- сколько витков пройденной орбиты показать (сплошная линия)
    future_rev -- сколько витков предстоящей орбиты показать (пунктир)
    elev, azim -- угол обзора; по умолчанию камера смотрит на КА
    Возвращает (fig, ax).
    """
    df = csv_path if isinstance(csv_path, pd.DataFrame) else pd.read_csv(csv_path)
    i0 = _pick_index(df, time)
    R = df[_cols(frame)].to_numpy()
    V = df[_cols(frame, "v")].to_numpy()
    t = df["t_s"].to_numpy()
    r_sat, v_sat = R[i0], V[i0]
    T = 2 * math.pi * math.sqrt(np.linalg.norm(r_sat) ** 3 / MU)
    when = df["time_utc"].iloc[i0]

    lon_s, lat_s, h_s = geodetic(df[_cols("itrf")].to_numpy()[i0])

    # камера: по умолчанию прямо на КА, чуть сверху
    if azim is None:
        azim = math.degrees(math.atan2(r_sat[1], r_sat[0]))
    if elev is None:
        elev = float(np.clip(math.degrees(math.asin(r_sat[2] / np.linalg.norm(r_sat))), -60, 60))
    ea, aa = math.radians(elev), math.radians(azim)
    view = np.array([math.cos(ea) * math.cos(aa), math.cos(ea) * math.sin(aa), math.sin(ea)])

    # поворот «земных» объектов в систему рисования
    M = np.eye(3) if frame.lower() == "itrf" else itrf_to_gcrf_matrix(when, dut1)

    if ax is None:
        fig = plt.figure(figsize=(8, 8), facecolor=SURFACE)
        ax = fig.add_axes([0, 0, 1, 0.92], projection="3d", facecolor=SURFACE)
    else:
        fig = ax.figure
    ax.computed_zorder = False

    # Земля
    lo, la = np.meshgrid(np.linspace(-180, 180, 73), np.linspace(-90, 90, 37))
    E = _ll_to_xyz(lo, la) @ M.T
    # мягкое освещение со стороны наблюдателя (без тёмной «ночной» половины)
    nrm = E / np.linalg.norm(E, axis=-1, keepdims=True)
    light = np.clip(nrm @ view, 0, 1)[..., None]
    base, rim = np.array(matplotlib.colors.to_rgb(EARTH)), np.array(matplotlib.colors.to_rgb("#c3ccd6"))
    fc = rim + (base - rim) * light ** 0.6
    ax.plot_surface(E[..., 0], E[..., 1], E[..., 2], facecolors=fc, linewidth=0,
                    antialiased=False, shade=False, rstride=1, cstride=1, zorder=1)

    # сетка 30°, экватор и гринвичский меридиан толще
    for lat in range(-60, 90, 30):
        p = _hide(_ll_to_xyz(np.linspace(-180, 180, 361), np.full(361, lat), 5) @ M.T, view, True)
        ax.plot(*p.T, color=GRID, lw=1.0 if lat == 0 else 0.5, zorder=2)
    for lon in range(-180, 180, 30):
        p = _hide(_ll_to_xyz(np.full(181, lon), np.linspace(-90, 90, 181), 5) @ M.T, view, True)
        ax.plot(*p.T, color=GRID, lw=1.0 if lon == 0 else 0.5, zorder=2)

    # материки
    for line in load_coastlines(coastline):
        p = _hide(_ll_to_xyz(line[:, 0], line[:, 1], 8) @ M.T, view, True)
        ax.plot(*p.T, color=COAST, lw=0.8, zorder=3)

    # орбита: пройденная (сплошная) и предстоящая (пунктир)
    past = (t >= t[i0] - past_rev * T) & (t <= t[i0])
    fut = (t >= t[i0]) & (t <= t[i0] + future_rev * T)
    ax.plot(*_hide(R[past], view, False).T, color=ORBIT, lw=2.0, zorder=4,
            label=f"пройдено ({past_rev:g} вит.)")
    ax.plot(*_hide(R[fut], view, False).T, color=ORBIT, lw=1.5, ls=(0, (4, 3)), zorder=4,
            label=f"впереди ({future_rev:g} вит.)")

    # подспутниковая точка и линия надира
    sub = _ll_to_xyz(lon_s, lat_s, 0.0) @ M.T
    ax.plot(*np.stack([sub, r_sat]).T, color=SAT, lw=1.0, zorder=5)
    ax.scatter(*sub, s=12, color=SAT, zorder=5)

    # КА и вектор скорости
    ax.scatter(*r_sat, s=90, color=SAT, edgecolors=SURFACE, linewidths=2, zorder=6, label="КА")
    vdir = v_sat / np.linalg.norm(v_sat) * 1500.0
    ax.quiver(*r_sat, *vdir, color=SAT, lw=1.5, arrow_length_ratio=0.25, zorder=6)

    lim = max(np.linalg.norm(r_sat), A_WGS84) * 1.02
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    head = title or "Положение КА"
    fig.suptitle(f"{head}\n{when}", color=TEXT, fontsize=12, y=0.98)
    info = (f"система: {frame.upper()}\n"
            f"r = ({r_sat[0]:.1f}, {r_sat[1]:.1f}, {r_sat[2]:.1f}) км\n"
            f"широта {lat_s:+.3f}°, долгота {lon_s:+.3f}°\n"
            f"высота {h_s:.2f} км")
    ax.text2D(0.02, 0.02, info, transform=ax.transAxes, color=TEXT_2, fontsize=9,
              family="monospace", va="bottom")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(TEXT_2)

    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    return fig, ax


# --------------------------------------------------------------------------- карта
def plot_ground_track(csv_path, time=None, past_rev=3.0, future_rev=1.0, coastline="auto",
                      title=None, out=None, ax=None):
    """Трасса КА (долгота/широта) с положением на момент `time`."""
    df = csv_path if isinstance(csv_path, pd.DataFrame) else pd.read_csv(csv_path)
    i0 = _pick_index(df, time)
    t = df["t_s"].to_numpy()
    lon, lat, h = geodetic(df[_cols("itrf")].to_numpy())
    T = 2 * math.pi * math.sqrt(np.linalg.norm(df[_cols("itrf")].to_numpy()[i0]) ** 3 / MU)

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5.8), facecolor=SURFACE)
    else:
        fig = ax.figure
    ax.set_facecolor(SURFACE)
    for line in load_coastlines(coastline):
        ax.plot(line[:, 0], line[:, 1], color=COAST, lw=0.6)

    def segment(mask, **kw):
        lo, la = lon[mask].copy(), lat[mask].copy()
        jump = np.where(np.abs(np.diff(lo)) > 180)[0] + 1      # разрыв на ±180°
        lo, la = np.insert(lo, jump, np.nan), np.insert(la, jump, np.nan)
        ax.plot(lo, la, **kw)

    segment((t >= t[i0] - past_rev * T) & (t <= t[i0]), color=ORBIT, lw=1.6,
            label=f"пройдено ({past_rev:g} вит.)")
    segment((t >= t[i0]) & (t <= t[i0] + future_rev * T), color=ORBIT, lw=1.2,
            ls=(0, (4, 3)), label=f"впереди ({future_rev:g} вит.)")
    ax.scatter(lon[i0], lat[i0], s=90, color=SAT, edgecolors=SURFACE, linewidths=2,
               zorder=5, label="КА")

    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 30)); ax.set_yticks(range(-90, 91, 30))
    ax.grid(color=GRID, lw=0.5, alpha=0.6)
    ax.set_aspect("equal")
    ax.tick_params(colors=TEXT_2, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_xlabel("долгота, °", color=TEXT_2); ax.set_ylabel("широта, °", color=TEXT_2)
    ax.set_title(f"{title or 'Трасса КА'} — {df['time_utc'].iloc[i0]}\n"
                 f"широта {lat[i0]:+.3f}°, долгота {lon[i0]:+.3f}°, высота {h[i0]:.2f} км",
                 color=TEXT, fontsize=11)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(TEXT_2)
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    return fig, ax


def main():
    ap = argparse.ArgumentParser(description="Земля с орбитой и положением КА")
    ap.add_argument("csv", help="файл координат от sat_coords.py")
    ap.add_argument("--time", default=None, help="момент UTC, напр. 2026-10-12T09:00:00Z (по умолчанию последняя точка)")
    ap.add_argument("--frame", default="itrf", choices=["itrf", "gcrf"])
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default="earth.png", help="PNG с 3D-видом")
    ap.add_argument("--track", default=None, help="PNG с трассой на карте")
    ap.add_argument("--no-coast", action="store_true", help="не рисовать материки")
    ap.add_argument("--show", action="store_true", help="открыть интерактивное окно")
    a = ap.parse_args()
    coast = None if a.no_coast else "auto"
    if not a.show:
        matplotlib.use("Agg")
    plot_satellite(a.csv, a.time, a.frame, coastline=coast, title=a.title, out=a.out)
    if a.track:
        plot_ground_track(a.csv, a.time, coastline=coast, title=a.title, out=a.track)
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
