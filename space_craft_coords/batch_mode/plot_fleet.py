#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Отображение группы КА после пакетного расчёта (batch_coords.py).

    from plot_fleet import load_fleet, plot_fleet_3d, plot_fleet_track, plot_fleet_altitude
    fleet = load_fleet("results/summary.csv")
    plot_fleet_3d(fleet, time=None, out="fleet_3d.png")          # Земля, плоскости орбит, КА
    plot_fleet_track(fleet, time=None, out="fleet_track.png")    # трассы на карте
    plot_fleet_altitude(fleet, out="fleet_altitude.png")         # средняя высота во времени

или из командной строки:

    python plot_fleet.py results/summary.csv --time 2026-10-12T09:00:00Z --out-dir results

Момент `time` по умолчанию -- последний момент, рассчитанный для ВСЕХ КА
(минимум из времён окончания). КА, у которых нет данных на этот момент, на 3D-виде
и карте пропускаются, о чём пишется в подписи.

Положение КА на произвольный момент берётся кубической эрмитовой интерполяцией
по соседним точкам CSV (положение + скорость), ошибка -- метры при шаге 60 с.
"""
import argparse
import math
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from plot_orbit import (A_WGS84, MU, SURFACE, TEXT, TEXT_2, GRID, COAST,
                        _cols, _hide, _draw_globe, geodetic, itrf_to_gcrf_matrix,
                        load_coastlines)

# категориальная палитра (фиксированный порядок, проверена на различимость при
# дальтонизме); при числе КА больше 8 все КА рисуются одним цветом с подписями.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]
SINGLE = "#2a78d6"
MARKER_EDGE = SURFACE


# --------------------------------------------------------------------------- данные
def _utc_seconds(series):
    ts = pd.to_datetime(series, utc=True, format="ISO8601")
    return (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def _to_seconds(time):
    t = pd.Timestamp(time)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return (t - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds()


def _fmt(sec):
    return (pd.Timestamp("1970-01-01", tz="UTC") + pd.Timedelta(seconds=float(sec))) \
        .strftime("%Y-%m-%dT%H:%M:%SZ")


def load_fleet(summary, out_dir=None):
    """Список КА из summary.csv (или DataFrame) пакетного расчёта.
    Берутся только КА со статусом ok. Цвет назначается в порядке следования в таблице."""
    sm = pd.read_csv(summary) if not isinstance(summary, pd.DataFrame) else summary
    base = out_dir or (os.path.dirname(os.path.abspath(summary)) if isinstance(summary, str) else ".")
    sm = sm[sm["status"] == "ok"].reset_index(drop=True)
    many = len(sm) > len(CATEGORICAL)
    fleet = []
    for k, row in sm.iterrows():
        path = row["coords_file"]
        path = path if os.path.isabs(path) else os.path.join(base, os.path.basename(path))
        df = pd.read_csv(path)
        df["_ts"] = _utc_seconds(df["time_utc"])
        chk_path = row.get("check_file", "")
        chk = None
        if isinstance(chk_path, str) and chk_path:
            cp = chk_path if os.path.isabs(chk_path) else os.path.join(base, os.path.basename(chk_path))
            if os.path.exists(cp):
                chk = pd.read_csv(cp)
        fleet.append({
            "name": str(row["scname"]), "scid": str(row["scid"]),
            "color": SINGLE if many else CATEGORICAL[k],
            "df": df, "chk": chk,
        })
    return fleet


def state_at(df, ts, frame="itrf"):
    """Положение и скорость на момент ts (секунды UTC от 1970) эрмитовой интерполяцией.
    None, если момент вне рассчитанного интервала."""
    t = df["_ts"].to_numpy()
    if ts < t[0] - 1e-6 or ts > t[-1] + 1e-6:
        return None
    j = int(np.clip(np.searchsorted(t, ts) - 1, 0, len(t) - 2))
    R, V = df[_cols(frame)].to_numpy(), df[_cols(frame, "v")].to_numpy()
    h = t[j + 1] - t[j]
    s = (ts - t[j]) / h
    h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
    h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
    r = h00 * R[j] + h10 * h * V[j] + h01 * R[j + 1] + h11 * h * V[j + 1]
    dh00, dh10 = (6 * s**2 - 6 * s) / h, 3 * s**2 - 4 * s + 1
    dh01, dh11 = (-6 * s**2 + 6 * s) / h, 3 * s**2 - 2 * s
    v = dh00 * R[j] + dh10 * V[j] + dh01 * R[j + 1] + dh11 * V[j + 1]
    return r, v


def common_time(fleet, time=None):
    """Момент отображения: заданный или последний, рассчитанный для всех КА."""
    if time is not None:
        return _to_seconds(time)
    return min(sc["df"]["_ts"].iloc[-1] for sc in fleet)


def _style_legend(leg):
    for txt in leg.get_texts():
        txt.set_color(TEXT_2)


# --------------------------------------------------------------------------- 3D
def plot_fleet_3d(fleet, time=None, frame="gcrf", coastline="auto", labels=True,
                  title="Группа КА", out=None, elev=25.0, azim=None, dut1=0.0):
    """Земля с орбитами и положениями всех КА, два полушария (вид и обратная сторона).

    frame='gcrf' -- инерциальная система: орбиты замкнуты, видно расположение плоскостей;
    frame='itrf' -- Земля неподвижна, орбиты закручены вращением Земли.
    Орбита каждого КА -- один виток, центрированный на моменте time.
    """
    ts = common_time(fleet, time)
    when = _fmt(ts)
    M = np.eye(3) if frame.lower() == "itrf" else itrf_to_gcrf_matrix(when, dut1)
    coast = load_coastlines(coastline)

    states, missing = [], []
    for sc in fleet:
        st = state_at(sc["df"], ts, frame)
        (states.append((sc, st)) if st is not None else missing.append(sc["name"]))
    if not states:
        raise ValueError(f"Ни у одного КА нет данных на {when}")

    if azim is None:                                     # смотрим на «центр масс» группы
        mean_dir = np.mean([st[0] / np.linalg.norm(st[0]) for _, st in states], axis=0)
        azim = math.degrees(math.atan2(mean_dir[1], mean_dir[0])) if np.linalg.norm(mean_dir[:2]) > 1e-3 else 0.0
    rmax = max(np.linalg.norm(st[0]) for _, st in states)
    lim = max(rmax, A_WGS84) * 1.02

    fig = plt.figure(figsize=(13, 7.0), facecolor=SURFACE)
    fig.suptitle(f"{title}: {len(states)} КА, {when}  ({frame.upper()})", color=TEXT, fontsize=13, y=0.98)
    for p, az in enumerate((azim, azim + 180.0)):
        ax = fig.add_axes([-0.02 + 0.5 * p, 0.07, 0.54, 0.86], projection="3d", facecolor=SURFACE)
        ax.computed_zorder = False
        ea, aa = math.radians(elev), math.radians(az)
        view = np.array([math.cos(ea) * math.cos(aa), math.cos(ea) * math.sin(aa), math.sin(ea)])
        _draw_globe(ax, M, view, coast_lines=coast)
        for sc, (r, v) in states:
            df = sc["df"]
            T = 2 * math.pi * math.sqrt(np.linalg.norm(r) ** 3 / MU)
            m = (df["_ts"] >= ts - T / 2) & (df["_ts"] <= ts + T / 2)
            orb = df.loc[m, _cols(frame)].to_numpy()
            ax.plot(*_hide(orb, view, False).T, color=sc["color"], lw=1.4, zorder=4)
            vis = not np.isnan(_hide(r[None, :], view, False)).any()
            if vis:
                ax.scatter(*r, s=60, color=sc["color"], edgecolors=MARKER_EDGE, linewidths=1.5, zorder=6)
                if labels:
                    off = r / np.linalg.norm(r) * 350.0
                    ax.text(*(r + off), sc["name"], color=TEXT, fontsize=8, zorder=7,
                            ha="center", va="center")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1), zoom=1.3)
        ax.view_init(elev=elev, azim=az)
        ax.set_axis_off()
        ax.set_title("вид с азимута {:.0f}°".format(az % 360), color=TEXT_2, fontsize=9, y=0.98)

    handles = [plt.Line2D([], [], color=sc["color"], lw=2, marker="o", markersize=6,
                          markeredgecolor=MARKER_EDGE, label=sc["name"])
               for sc, _ in states] if len(states) <= len(CATEGORICAL) else []
    if handles:
        _style_legend(fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8),
                                 frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.02)))
    note = f"нет данных на этот момент: {', '.join(missing)}" if missing else ""
    if note:
        fig.text(0.01, 0.01, note, color=TEXT_2, fontsize=9)
    if out:
        fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    return fig


# --------------------------------------------------------------------------- карта
def plot_fleet_track(fleet, time=None, past_rev=1.0, coastline="auto", labels=True,
                     title="Трассы группы КА", out=None):
    """Карта: трассы всех КА за past_rev витков до момента time и их текущее положение."""
    ts = common_time(fleet, time)
    fig, ax = plt.subplots(figsize=(13, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for line in load_coastlines(coastline):
        ax.plot(line[:, 0], line[:, 1], color=COAST, lw=0.6)

    shown, missing = [], []
    for sc in fleet:
        df = sc["df"]
        st = state_at(df, ts, "itrf")
        if st is None:
            missing.append(sc["name"])
            continue
        r, _ = st
        T = 2 * math.pi * math.sqrt(np.linalg.norm(r) ** 3 / MU)
        m = (df["_ts"] >= ts - past_rev * T) & (df["_ts"] <= ts)
        lon, lat, _ = geodetic(df.loc[m, _cols("itrf")].to_numpy())
        lon, lat = np.append(lon, geodetic(r)[0]), np.append(lat, geodetic(r)[1])
        jump = np.where(np.abs(np.diff(lon)) > 180)[0] + 1
        ax.plot(np.insert(lon, jump, np.nan), np.insert(lat, jump, np.nan),
                color=sc["color"], lw=1.4, label=sc["name"])
        lo, la, h = geodetic(r)
        ax.scatter(lo, la, s=70, color=sc["color"], edgecolors=MARKER_EDGE, linewidths=1.5, zorder=5)
        if labels:
            ax.annotate(sc["name"], (lo, la), xytext=(6, 6), textcoords="offset points",
                        color=TEXT, fontsize=8, zorder=6)
        shown.append(sc["name"])

    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 30)); ax.set_yticks(range(-90, 91, 30))
    ax.grid(color=GRID, lw=0.5, alpha=0.6)
    ax.set_aspect("equal")
    ax.tick_params(colors=TEXT_2, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_xlabel("долгота, °", color=TEXT_2); ax.set_ylabel("геодезическая широта, °", color=TEXT_2)
    sub = f"трасса за {past_rev:g} вит. до момента; точка -- положение КА"
    if missing:
        sub += f"; нет данных: {', '.join(missing)}"
    ax.set_title(f"{title}: {len(shown)} КА, {_fmt(ts)}\n{sub}", color=TEXT, fontsize=11)
    if len(shown) <= len(CATEGORICAL):
        _style_legend(ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1),
                                ncol=min(len(shown), 8), frameon=False, fontsize=9))
    if out:
        fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    return fig, ax


# --------------------------------------------------------------------------- высота
def plot_fleet_altitude(fleet, time=None, title="Довыведение: средняя высота орбиты по таблице", out=None):
    """Средняя высота (ha+hp)/2 = a_табл − 6378.137 км по эпохам таблицы для всех КА
    (берётся из *_check.csv). Вертикальная линия -- момент, показанный на 3D-виде и карте.
    Расхождение модели с таблицей -- в summary.csv (max_da_m, calib_da_m, warning)."""
    fig, ax = plt.subplots(figsize=(12, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    many = len(fleet) > len(CATEGORICAL)
    for sc in fleet:
        chk = sc["chk"]
        if chk is None or chk.empty:
            continue
        x = pd.to_datetime(chk["time_utc"], utc=True, format="ISO8601")
        y = chk["a_table_km"] - A_WGS84
        ax.plot(x, y, color=sc["color"], lw=1.2 if many else 2.0, alpha=0.8 if many else 1.0,
                label=sc["name"])
        if many or len(fleet) <= 4:                       # прямые подписи на конце линии
            ax.annotate(sc["name"], (x.iloc[-1], y.iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", color=TEXT_2, fontsize=8, va="center")
    ts = common_time(fleet, time)
    ax.axvline(pd.Timestamp("1970-01-01", tz="UTC") + pd.Timedelta(seconds=ts),
               color=TEXT_2, lw=0.8, ls=(0, (3, 3)))
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
    ax.set_ylabel("(ha + hp)/2, км", color=TEXT_2)
    ax.set_title(f"{title}\nпунктир -- момент {_fmt(ts)}", color=TEXT, fontsize=11)
    if not many and len(fleet) >= 2:
        _style_legend(ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
                                ncol=min(len(fleet), 8), frameon=False, fontsize=9))
    if out:
        fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    return fig, ax


def plot_fleet_all(summary_or_fleet, out_dir, time=None, frame="gcrf", coastline="auto"):
    """Три рисунка группы в out_dir: fleet_3d.png, fleet_track.png, fleet_altitude.png."""
    fleet = load_fleet(summary_or_fleet) if not isinstance(summary_or_fleet, list) else summary_or_fleet
    if not fleet:
        print("Нет рассчитанных КА для отображения.")
        return []
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(out_dir, n) for n in ("fleet_3d.png", "fleet_track.png", "fleet_altitude.png")]
    plot_fleet_3d(fleet, time, frame=frame, coastline=coastline, out=paths[0])
    plot_fleet_track(fleet, time, coastline=coastline, out=paths[1])
    plot_fleet_altitude(fleet, time, out=paths[2])
    plt.close("all")
    return paths


def main():
    ap = argparse.ArgumentParser(description="Отображение всех КА после пакетного расчёта")
    ap.add_argument("summary", help="summary.csv от batch_coords.py")
    ap.add_argument("--time", default=None, help="момент UTC (по умолчанию последний общий)")
    ap.add_argument("--frame", default="gcrf", choices=["gcrf", "itrf"])
    ap.add_argument("--out-dir", default=None, help="куда сохранить PNG (по умолчанию рядом с summary)")
    ap.add_argument("--no-coast", action="store_true")
    ap.add_argument("--show", action="store_true", help="открыть интерактивные окна")
    a = ap.parse_args()
    if not a.show:
        matplotlib.use("Agg")
    out_dir = a.out_dir or os.path.dirname(os.path.abspath(a.summary))
    fleet = load_fleet(a.summary)
    coast = None if a.no_coast else "auto"
    if a.show:
        plot_fleet_3d(fleet, a.time, frame=a.frame, coastline=coast)
        plot_fleet_track(fleet, a.time, coastline=coast)
        plot_fleet_altitude(fleet, a.time)
        plt.show()
    else:
        for p in plot_fleet_all(fleet, out_dir, a.time, a.frame, coast):
            print(p)


if __name__ == "__main__":
    main()
