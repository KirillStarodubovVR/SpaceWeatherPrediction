#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Координаты КА (x, y, z) на этапе довыведения по OPM и таблице средних элементов.

Входные данные
  * OPM (CCSDS) на начальную эпоху: вектор состояния r, v в GCRF, GM, масса.
  * Excel/CSV (лист all_sats): scname, scid, date, ha, hp, inc, raan, dRaanDt, mass
    -- средние элементы теории J2 на ряд эпох (ha, hp в м; inc, raan в рад;
       inc/raan отсчитываются от истинного экватора даты, т.е. в системе CIRS).

Идея
  Средние элементы задают размер, форму и плоскость орбиты, но не положение КА
  на орбите. Положение берём из OPM и численно интегрируем движение (центральное
  поле + J2). Тягу двигателя и торможение атмосферы, о которых данных нет, заменяем
  эквивалентным трансверсальным ускорением, подобранным так, чтобы средние a и e
  модели на каждой эпохе таблицы совпадали с таблицей. Наклонение и ДВУ
  получаются из интегрирования и сверяются с таблицей.

Запуск
  python sat_coords.py --opm R3D21_....opm --excel all_sats.xlsx --step 60 --out R3D21_xyz.csv

Зависимости: numpy, scipy, pandas, openpyxl, pyerfa  (pip install numpy scipy pandas openpyxl pyerfa)
"""
import argparse
import math
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import erfa
from scipy.integrate import solve_ivp

R_E_DEFAULT = 6378.137          # км, экваториальный радиус Земли (WGS-84)
J2_DEFAULT = 1.08263e-3
G0 = 9.80665                    # м/с^2
OMEGA_EARTH = 7.292115146706979e-5  # рад/с, скорость вращения Земли (ERA)


# --------------------------------------------------------------------------- время
_TS = re.compile(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2}(?:\.\d*)?)")


def parse_utc(value):
    """Дата UTC -> (datetime с точностью до минуты, секунды float).
    Секунды храним отдельно, т.к. в таблице бывает 15+ знаков после запятой."""
    if isinstance(value, (pd.Timestamp, datetime)):
        ts = pd.Timestamp(value)
        return (datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute),
                ts.second + ts.microsecond * 1e-6 + ts.nanosecond * 1e-9)
    m = _TS.search(str(value))
    if not m:
        raise ValueError(f"Не удалось разобрать дату: {value!r}")
    y, mo, d, h, mi = map(int, m.groups()[:5])
    return datetime(y, mo, d, h, mi), float(m.group(6))


def seconds_between(t, t0):
    (b, s), (b0, s0) = t, t0
    return (b - b0).total_seconds() + (s - s0)


def fmt_utc(t0, dt):
    b0, s0 = t0
    return (b0 + timedelta(seconds=s0 + dt)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# --------------------------------------------------------------------------- OPM
def read_opm(path):
    kv = {}
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("COMMENT") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            val = re.sub(r"\[.*?\]\s*$", "", val).strip()   # убираем [km] и т.п.
            kv[key.strip().upper()] = val
    need = ["EPOCH", "X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT"]
    miss = [k for k in need if k not in kv]
    if miss:
        raise ValueError(f"В OPM нет полей: {miss}")
    if kv.get("TIME_SYSTEM", "UTC").upper() != "UTC":
        raise ValueError("Поддерживается только TIME_SYSTEM = UTC")
    frame = kv.get("REF_FRAME", "GCRF").upper()
    if frame not in ("GCRF", "ICRF", "EME2000", "J2000"):
        raise ValueError(f"REF_FRAME={frame}: ожидается GCRF (или EME2000)")
    return {
        "epoch": parse_utc(kv["EPOCH"]),
        "r": np.array([float(kv["X"]), float(kv["Y"]), float(kv["Z"])]),
        "v": np.array([float(kv["X_DOT"]), float(kv["Y_DOT"]), float(kv["Z_DOT"])]),
        "mu": float(kv.get("GM", 398600.4415)),
        "mass": float(kv["MASS"]) if "MASS" in kv else None,
        "name": kv.get("OBJECT_NAME", ""),
        "id": kv.get("OBJECT_ID", ""),
        "scid": kv.get("USER_DEFINED_SCID", kv.get("OBJECT_ID", "")),
        "file": str(path),
        "frame": frame,
    }


# --------------------------------------------------------------------------- таблица
def load_table(path, sheet=None):
    """Читает таблицу (.xlsx/.csv) целиком, имена столбцов -> нижний регистр."""
    if str(path).lower().endswith((".csv", ".txt")):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ["date", "ha", "hp", "inc", "raan"]:
        if c not in df.columns:
            raise ValueError(f"В таблице нет столбца '{c}'. Есть: {list(df.columns)}")
    return df


def norm_id(value):
    """'27', 27, 27.0 -> '27'."""
    return re.sub(r"\.0$", "", str(value).strip())


def select_sc(df, opm, scid=None):
    """Строки нужного КА: по scid (по умолчанию OBJECT_ID из OPM), иначе по scname."""
    want = norm_id(scid if scid is not None else opm["id"])
    sel = pd.Series(False, index=df.index)
    if "scid" in df.columns and want:
        sel = df["scid"].map(norm_id) == want
    if not sel.any() and "scname" in df.columns and opm["name"]:
        sel = df["scname"].astype(str).str.strip() == opm["name"].strip()
    if not sel.any():
        if "scid" in df.columns and df["scid"].nunique() > 1:
            raise ValueError(f"Не найден КА scid={want} / {opm['name']} в таблице")
        sel[:] = True
    return df[sel].copy()


def prepare_rows(df, opm, r_e=R_E_DEFAULT):
    """Строки одного КА -> время от эпохи OPM, a, e, углы в радианах."""
    df = df.copy()
    times = [parse_utc(v) for v in df["date"]]
    df["t"] = [seconds_between(t, opm["epoch"]) for t in times]
    df = df.sort_values("t").drop_duplicates("t").reset_index(drop=True)

    ha, hp = df["ha"].astype(float).to_numpy(), df["hp"].astype(float).to_numpy()
    if np.nanmedian(ha) > 1e4:                     # высоты в метрах
        ha, hp = ha / 1000.0, hp / 1000.0
    inc, raan = df["inc"].astype(float).to_numpy(), df["raan"].astype(float).to_numpy()
    deg = np.nanmax(np.abs(inc)) > math.pi + 1e-9 or np.nanmax(np.abs(raan)) > 2 * math.pi + 1e-9
    if deg:
        inc, raan = np.radians(inc), np.radians(raan)
    df["a"] = r_e + (ha + hp) / 2.0
    df["e"] = np.abs(ha - hp) / (2.0 * df["a"])
    df["inc_rad"] = inc
    df["raan_rad"] = np.unwrap(raan)
    if "draandt" in df.columns:
        d = df["draandt"].astype(float).to_numpy()
        df["draandt_rad"] = np.radians(d) if deg else d
    else:
        df["draandt_rad"] = np.nan
    if "mass" not in df.columns:
        df["mass"] = np.nan
    return df


def read_table(path, sheet, opm, scid=None, r_e=R_E_DEFAULT):
    """Совместимость: таблица -> подготовленные строки одного КА."""
    return prepare_rows(select_sc(load_table(path, sheet), opm, scid), opm, r_e)


def estimate_j2(df, mu, r_e):
    """J2, при котором dRaanDt таблицы = -3/2 n J2 (R/p)^2 cos i."""
    d = df["draandt_rad"].to_numpy()
    a, e, i = df["a"].to_numpy(), df["e"].to_numpy(), df["inc_rad"].to_numpy()
    n = np.sqrt(mu / a**3)
    p = a * (1 - e**2)
    den = 1.5 * n * (r_e / p) ** 2 * np.cos(i)
    ok = np.isfinite(d) & (np.abs(den) > 1e-12)
    if not ok.any():
        return None
    return float(np.median(-d[ok] / den[ok]))


# --------------------------------------------------------------------------- механика
def elements(r, v, mu):
    """Оскулирующие элементы (векторизовано): a, ex, ey, i, raan, u.
    ex, ey -- вектор эксцентриситета в осях (линия узлов, перпендикуляр в плоскости)."""
    r, v = np.atleast_2d(r), np.atleast_2d(v)
    rn = np.linalg.norm(r, axis=1)
    a = 1.0 / (2.0 / rn - np.sum(v * v, axis=1) / mu)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=1)
    w = h / hn[:, None]
    inc = np.arccos(np.clip(w[:, 2], -1, 1))
    raan = np.arctan2(h[:, 0], -h[:, 1])
    nv = np.stack([np.cos(raan), np.sin(raan), np.zeros_like(raan)], axis=1)
    mv = np.cross(w, nv)
    u = np.arctan2(np.sum(r * mv, axis=1), np.sum(r * nv, axis=1))
    ev = np.cross(v, h) / mu - r / rn[:, None]
    return a, np.sum(ev * nv, axis=1), np.sum(ev * mv, axis=1), inc, raan, u


class Model:
    def __init__(self, mu, r_e, j2):
        self.mu, self.re, self.j2 = mu, r_e, j2

    def rhs(self, t, y, thrust):
        r, v = y[:3], y[3:]
        x, yy, z = r
        r2 = r @ r
        rn = math.sqrt(r2)
        k = 1.5 * self.j2 * self.mu * self.re**2 / rn**5
        z2 = 5.0 * z * z / r2
        acc = -self.mu * r / (rn**3)
        acc = acc + np.array([-k * x * (1 - z2), -k * yy * (1 - z2), -k * z * (3 - z2)])
        if thrust is not None:
            A, B, phi0, wdot, tref = thrust
            vn = math.sqrt(v @ v)
            at = A
            if B != 0.0:
                h = np.cross(r, v)
                nh = math.hypot(h[0], h[1])
                nvec = np.array([-h[1] / nh, h[0] / nh, 0.0])           # на восх. узел
                mvec = np.cross(h / math.sqrt(h @ h), nvec)
                u = math.atan2(r @ mvec, r @ nvec)
                at += B * math.cos(u - (phi0 + wdot * (t - tref)))
            acc = acc + at * v / vn
        return np.concatenate([v, acc])

    def propagate(self, y0, t0, t1, thrust=None, t_eval=None):
        sol = solve_ivp(self.rhs, (t0, t1), y0, method="DOP853", args=(thrust,),
                        rtol=1e-11, atol=1e-9, dense_output=True, t_eval=t_eval)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol

    def mean_elements(self, y0, t0, nsamp=360):
        """Средние элементы как средние оскулирующих за один виток пассивного полёта
        (короткопериодические возмущения J2 за виток усредняются в ноль)."""
        a0 = elements(y0[:3], y0[3:], self.mu)[0][0]
        T = 2 * math.pi * math.sqrt(a0**3 / self.mu)
        ts = t0 + np.linspace(0.0, T, nsamp, endpoint=False)
        sol = self.propagate(y0, t0, t0 + T, t_eval=ts)
        a, ex, ey, inc, raan, _ = elements(sol.y[:3].T, sol.y[3:].T, self.mu)
        exm, eym = ex.mean(), ey.mean()
        return {"a": a.mean(), "e": math.hypot(exm, eym), "w": math.atan2(eym, exm),
                "inc": inc.mean(), "raan": np.unwrap(raan).mean(), "T": T}


# --------------------------------------------------------------------------- основной расчёт
DEFAULTS = dict(step=60.0, end_seconds=None, re=R_E_DEFAULT, j2=None, dut1=0.0,
                absolute=False, iters=2, verbose=False)


def compute(opm, df, log=print, **opts):
    """Расчёт одного КА.

    opm  -- словарь из read_opm()
    df   -- строки этого КА после prepare_rows()
    opts -- step, end_seconds, re, j2, dut1, absolute, iters, verbose (см. DEFAULTS)
    Возвращает (out, chk): координаты и таблицу сверки (pandas.DataFrame).
    """
    args = argparse.Namespace(**{**DEFAULTS, **opts})
    print = log                                        # noqa: A001 -- весь вывод через log
    mu, r_e = opm["mu"], args.re
    if df.t.iloc[-1] <= 0:
        raise ValueError("все строки таблицы раньше эпохи OPM")
    print(f"КА {opm['name']} (id {opm['id']}): строк в таблице {len(df)}, "
          f"с {fmt_utc(opm['epoch'], df.t.iloc[0])} по {fmt_utc(opm['epoch'], df.t.iloc[-1])}")

    j2 = args.j2 if args.j2 else (estimate_j2(df, mu, r_e) or J2_DEFAULT)
    print(f"J2 = {j2:.8e}" + ("" if args.j2 else "  (оценён по dRaanDt таблицы)"))

    # GCRF -> CIRS (истинный экватор даты): матрица на эпоху OPM
    b, s = opm["epoch"]
    u1, u2 = erfa.dtf2d("UTC", b.year, b.month, b.day, b.hour, b.minute, s)
    tt1, tt2 = erfa.taitt(*erfa.utctai(u1, u2))
    C = erfa.c2i06a(tt1, tt2)
    y = np.concatenate([C @ opm["r"], C @ opm["v"]])

    model = Model(mu, r_e, j2)
    m0 = model.mean_elements(y, 0.0)
    ref = lambda col, t: float(np.interp(t, df.t.to_numpy(), df[col].to_numpy()))

    # Калибровка по первой эпохе: «среднее» таблицы и наше среднее по витку определены
    # немного по-разному. Абсолютный уровень a и e берём из OPM, из таблицы -- их изменение.
    if args.absolute:
        a_bias, e_scale = 0.0, 1.0
    else:
        a_bias = m0["a"] - ref("a", 0.0)
        e_scale = m0["e"] / ref("e", 0.0) if ref("e", 0.0) > 1e-7 else 1.0
    print("Начальная эпоха, модель (среднее по витку из OPM) / таблица:")
    print(f"  a   {m0['a']:.4f} / {ref('a', 0):.4f} км   (смещение {a_bias*1e3:+.1f} м)")
    print(f"  e   {m0['e']:.3e} / {ref('e', 0):.3e}       (масштаб {e_scale:.3f})")
    print(f"  i   {math.degrees(m0['inc']):.5f} / {math.degrees(ref('inc_rad', 0)):.5f} град  (CIRS)")
    a_target = lambda t: ref("a", t) + a_bias
    e_target = lambda t: max(ref("e", t) * e_scale, 0.0)

    t_end = args.end_seconds if args.end_seconds is not None else float(df.t.iloc[-1])
    knots = np.unique(np.concatenate([[0.0], df.t[(df.t > 0) & (df.t < t_end)].to_numpy(), [t_end]]))
    grid = np.unique(np.concatenate([np.arange(0.0, t_end, args.step),
                                     df.t[(df.t >= 0) & (df.t <= t_end)], [t_end]]))
    tt = df.t.to_numpy()

    rows_t, rows_y, checks = [], [], []
    mm = m0
    for k in range(len(knots) - 1):
        t0, t1 = knots[k], knots[k + 1]
        dt = t1 - t0
        # сверка с таблицей, если узел -- строка таблицы
        hit = np.where(np.abs(tt - t0) < 1e-3)[0]
        if hit.size:
            row = df.iloc[hit[0]]
            dr = row.draandt_rad if np.isfinite(row.draandt_rad) else 0.0
            checks.append({
                "time_utc": fmt_utc(opm["epoch"], t0),
                "a_model_km": mm["a"], "a_table_km": row.a,
                "da_m": (mm["a"] - a_target(t0)) * 1e3,
                "e_model": mm["e"], "e_table": row.e,
                "de": mm["e"] - e_target(t0),
                "di_deg": math.degrees(mm["inc"] - row.inc_rad),
                "draan_deg": math.degrees((mm["raan"] - (row.raan_rad + dr * mm["T"] / 2)
                                           + math.pi) % (2 * math.pi) - math.pi),
            })
        # эквивалентное трансверсальное ускорение a_T = A + B*cos(u - w)
        a_tg, e_tg = a_target(t1), e_target(t1)
        am = mm["a"]
        n = math.sqrt(mu / am**3)
        vc = math.sqrt(mu / am)
        kA = math.sqrt(mu) / (2 * am**1.5)                 # da/dt -> a_T (околокруговая)
        A = (a_tg - am) / dt * kA
        steer_e = mm["e"] > 1e-7 and dt >= mm["T"]     # e подстраиваем на отрезках >= витка
        B = vc * (e_tg - mm["e"]) / dt if steer_e else 0.0  # d|e|/dt = B/v
        p = am * (1 - mm["e"] ** 2)
        wdot = 0.75 * n * j2 * (r_e / p) ** 2 * (5 * math.cos(mm["inc"]) ** 2 - 1)

        # стрельба: подправляем A и B, пока средние a, e в конце отрезка не совпадут с целью
        for it in range(args.iters + 1):
            thrust = (A, B, mm["w"], wdot, t0 + mm["T"] / 2)
            sol = model.propagate(y, t0, t1, thrust)
            m_end = model.mean_elements(sol.y[:, -1], t1)
            err_a, err_e = m_end["a"] - a_tg, m_end["e"] - e_tg
            if abs(err_a) < 5e-4 and (not steer_e or abs(err_e) < 2e-7):
                break
            A -= err_a / dt * kA
            if steer_e:
                B -= vc * err_e / dt

        g = grid[(grid >= t0) & (grid < t1)] if k < len(knots) - 2 else grid[(grid >= t0) & (grid <= t1)]
        if g.size:
            rows_t.append(g)
            rows_y.append(sol.sol(g).T)
        y, mm = sol.y[:, -1], m_end

        if args.verbose:
            mass = ref("mass", t0)
            f = f"F_экв~{abs(A) * 1e3 * mass * 1e3:.2f} мН" if np.isfinite(mass) else ""
            print(f"  {fmt_utc(opm['epoch'], t0)}  a_T={A * 1e3:+.3e} м/с2  {f}  итераций {it}")

    # точка t_end тоже участвует в сверке
    hit = np.where(np.abs(tt - t_end) < 1e-3)[0]
    if hit.size:
        row = df.iloc[hit[0]]
        dr = row.draandt_rad if np.isfinite(row.draandt_rad) else 0.0
        checks.append({
            "time_utc": fmt_utc(opm["epoch"], t_end),
            "a_model_km": mm["a"], "a_table_km": row.a, "da_m": (mm["a"] - a_target(t_end)) * 1e3,
            "e_model": mm["e"], "e_table": row.e, "de": mm["e"] - e_target(t_end),
            "di_deg": math.degrees(mm["inc"] - row.inc_rad),
            "draan_deg": math.degrees((mm["raan"] - (row.raan_rad + dr * mm["T"] / 2)
                                       + math.pi) % (2 * math.pi) - math.pi),
        })

    t = np.concatenate(rows_t)
    Y = np.concatenate(rows_y)
    r_cirs, v_cirs = Y[:, :3], Y[:, 3:]
    r_gcrf, v_gcrf = r_cirs @ C, v_cirs @ C           # C^T @ r для каждой строки

    # CIRS -> ITRF (через угол поворота Земли; движение полюса не учитывается)
    ut1a, ut1b = erfa.utcut1(np.full_like(t, u1), u2 + t / 86400.0, args.dut1)
    era = erfa.era00(ut1a, ut1b)
    c, s_ = np.cos(era), np.sin(era)
    x_i = c * r_cirs[:, 0] + s_ * r_cirs[:, 1]
    y_i = -s_ * r_cirs[:, 0] + c * r_cirs[:, 1]
    vx_i = c * v_cirs[:, 0] + s_ * v_cirs[:, 1] + OMEGA_EARTH * y_i
    vy_i = -s_ * v_cirs[:, 0] + c * v_cirs[:, 1] - OMEGA_EARTH * x_i

    out = pd.DataFrame({
        "time_utc": [fmt_utc(opm["epoch"], x) for x in t], "t_s": t,
        "x_gcrf_km": r_gcrf[:, 0], "y_gcrf_km": r_gcrf[:, 1], "z_gcrf_km": r_gcrf[:, 2],
        "vx_gcrf_kms": v_gcrf[:, 0], "vy_gcrf_kms": v_gcrf[:, 1], "vz_gcrf_kms": v_gcrf[:, 2],
        "x_itrf_km": x_i, "y_itrf_km": y_i, "z_itrf_km": r_cirs[:, 2],
        "vx_itrf_kms": vx_i, "vy_itrf_kms": vy_i, "vz_itrf_kms": v_cirs[:, 2],
    })
    chk = pd.DataFrame(checks)
    if len(chk) > 1:
        c2 = chk.iloc[1:]
        print(f"  макс |Δa| = {c2.da_m.abs().max():.1f} м, макс |Δe| = {c2.de.abs().max():.2e}, "
              f"макс |Δi| = {c2.di_deg.abs().max():.4f}°, макс |ΔΩ| = {c2.draan_deg.abs().max():.4f}°")
    return out, chk


def save_results(out, chk, out_path):
    """Пишет координаты и сверку; возвращает путь к файлу сверки."""
    out.to_csv(out_path, index=False, float_format="%.6f")
    chk_path = re.sub(r"(\.csv)?$", "_check.csv", str(out_path), count=1)
    chk.to_csv(chk_path, index=False, float_format="%.6g")
    return chk_path


def run(args):
    opm = read_opm(args.opm)
    df = read_table(args.excel, args.sheet, opm, args.scid, args.re)
    opts = {k: getattr(args, k) for k in DEFAULTS}
    out, chk = compute(opm, df, **opts)
    chk_path = save_results(out, chk, args.out)
    print(f"\nКоординаты: {args.out} ({len(out)} точек, шаг {args.step:g} с)")
    print(f"Сверка со средними элементами таблицы: {chk_path}")
    return out, chk


def main():
    ap = argparse.ArgumentParser(description="Координаты КА по OPM и таблице средних элементов")
    ap.add_argument("--opm", required=True, help="файл OPM (CCSDS)")
    ap.add_argument("--excel", required=True, help="таблица .xlsx/.csv со столбцами date, ha, hp, inc, raan, dRaanDt, mass")
    ap.add_argument("--sheet", default=None, help="имя листа (по умолчанию первый)")
    ap.add_argument("--scid", default=None, help="scid КА (по умолчанию OBJECT_ID из OPM)")
    ap.add_argument("--step", type=float, default=60.0, help="шаг вывода, с")
    ap.add_argument("--end-seconds", type=float, default=None, help="конец расчёта, с от эпохи OPM (по умолчанию последняя строка)")
    ap.add_argument("--re", type=float, default=R_E_DEFAULT, help="экваториальный радиус для ha/hp, км")
    ap.add_argument("--j2", type=float, default=None, help="J2 (по умолчанию оценивается по dRaanDt)")
    ap.add_argument("--dut1", type=float, default=0.0, help="UT1-UTC, с (для ITRF)")
    ap.add_argument("--out", default="coords.csv", help="выходной CSV")
    ap.add_argument("--absolute", action="store_true",
                    help="подгонять под абсолютные a, e таблицы (без калибровки по OPM)")
    ap.add_argument("--iters", type=int, default=2, help="итераций стрельбы на отрезок")
    ap.add_argument("-v", "--verbose", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
