#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пакетный расчёт координат группы КА по набору OPM и одной таблице средних элементов.

Таблица (.xlsx/.csv) содержит данные нескольких КА подряд: сначала все строки одного
КА, сразу за ними строки следующего и т.д. Блоки выделяются по смене scid (и scname).
Каждому блоку подбирается OPM:
  * OBJECT_ID или USER_DEFINED_SCID совпадает со scid, иначе OBJECT_NAME со scname;
  * если подходящих OPM несколько -- берётся OPM с эпохой, ближайшей к первой строке блока.

Для каждого КА выполняется тот же расчёт, что в sat_coords.py (функция compute),
расчёты идут параллельно в нескольких процессах. Затем строятся общие рисунки группы.

Запуск:
    python batch_coords.py --opm-dir opm/ --excel all_sats.xlsx --out-dir results

Результаты в out-dir:
    <scname>_<scid>_xyz.csv        координаты и скорости в GCRF и ITRF
    <scname>_<scid>_xyz_check.csv  сверка средних элементов с таблицей
    <scname>_<scid>.log            журнал расчёта КА
    summary.csv                    сводка: статус, интервал, невязки, итоговая высота
    fleet_3d.png, fleet_track.png, fleet_altitude.png   отображение всех КА
"""
import argparse
import glob
import math
import os
import re
import sys
import time as _time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import sat_coords as sc

R_E = sc.R_E_DEFAULT
# пороги предупреждения о несогласованности OPM и таблицы в начальной точке
# (для R3D21 на реальных данных: 29 м и 0,002°)
CALIB_DA_WARN_M = 1000.0
CALIB_DI_WARN_DEG = 0.05


# --------------------------------------------------------------------------- входные данные
def find_opm_files(opm_path, recursive=False):
    """Путь к папке, маска (opm/*.opm) или один файл -> список файлов OPM."""
    if os.path.isdir(opm_path):
        pattern = os.path.join(opm_path, "**", "*") if recursive else os.path.join(opm_path, "*")
        files = glob.glob(pattern, recursive=recursive)
    else:
        files = glob.glob(opm_path, recursive=recursive)
    return sorted(f for f in files if os.path.isfile(f) and f.lower().endswith(".opm"))


def read_all_opm(files):
    """Читает все OPM; ошибки чтения не прерывают пакет."""
    opms, bad = [], []
    for f in files:
        try:
            opms.append(sc.read_opm(f))
        except Exception as exc:
            bad.append((f, str(exc)))
    return opms, bad


def split_blocks(table):
    """Делит таблицу на блоки КА, идущие подряд. Повторное появление того же КА
    ниже по таблице присоединяется к его блоку (с предупреждением)."""
    t = table.dropna(subset=["date", "ha", "hp"]).copy()
    if "scid" not in t.columns and "scname" not in t.columns:
        return [{"scid": "", "scname": "", "rows": t, "first_row": 0}], []
    sid = t["scid"].map(sc.norm_id) if "scid" in t.columns else pd.Series("", index=t.index)
    snm = t["scname"].astype(str).str.strip() if "scname" in t.columns else pd.Series("", index=t.index)
    key = sid + "|" + snm
    run_id = (key != key.shift()).cumsum()
    blocks, order, warnings = {}, [], []
    for _, part in t.groupby(run_id, sort=True):
        k = key.loc[part.index[0]]
        if k in blocks:
            warnings.append(f"КА {k.replace('|', ' / ')}: строки встречаются в таблице "
                            f"несколькими блоками, объединены")
            blocks[k]["rows"] = pd.concat([blocks[k]["rows"], part])
        else:
            blocks[k] = {"scid": sid.loc[part.index[0]], "scname": snm.loc[part.index[0]],
                         "rows": part, "first_row": int(part.index[0]) + 2}   # +2: строка Excel
            order.append(k)
    return [blocks[k] for k in order], warnings


def match_opm(block, opms):
    """Подбирает OPM к блоку таблицы. Возвращает (opm, причина-или-None)."""
    sid, name = block["scid"], block["scname"]
    cands = [o for o in opms if sid and sid in (sc.norm_id(o["id"]), sc.norm_id(o["scid"]))]
    if not cands and name:
        cands = [o for o in opms if o["name"].strip() == name]
    if not cands:
        return None, "нет OPM"
    t_first = min(sc.parse_utc(v) for v in block["rows"]["date"])
    return min(cands, key=lambda o: abs(sc.seconds_between(t_first, o["epoch"]))), None


def safe_name(s):
    return re.sub(r"[^\w.-]+", "_", str(s)).strip("_") or "sc"


# --------------------------------------------------------------------------- один КА (в процессе)
def run_one(job):
    """Расчёт одного КА. Выполняется в отдельном процессе; ничего не бросает наружу."""
    t_start = _time.time()
    log_lines = []
    res = {"scname": job["scname"], "scid": job["scid"], "opm_file": job["opm"]["file"],
           "status": "error", "error": "", "coords_file": "", "check_file": "",
           "log_file": job["log_file"]}
    try:
        opm = job["opm"]
        df = sc.prepare_rows(job["rows"], opm, job["opts"].get("re", R_E))
        out, chk = sc.compute(opm, df, log=lambda *a, **k: log_lines.append(" ".join(map(str, a))),
                              **job["opts"])
        chk_path = sc.save_results(out, chk, job["out_file"])
        res.update(status="ok", coords_file=job["out_file"], check_file=chk_path)
        res.update(summarize(out, chk, df, opm))
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
        log_lines.append(traceback.format_exc())
    res["runtime_s"] = round(_time.time() - t_start, 1)
    with open(job["log_file"], "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    return res


def summarize(out, chk, df, opm):
    """Строка сводки: интервал, число точек, невязки, итоговое положение и высота."""
    r_end = out[["x_itrf_km", "y_itrf_km", "z_itrf_km"]].to_numpy()[-1]
    rn = float(np.linalg.norm(r_end))
    d = {
        "opm_epoch": sc.fmt_utc(opm["epoch"], 0.0),
        "table_rows": len(df), "t_start": out["time_utc"].iloc[0], "t_end": out["time_utc"].iloc[-1],
        "days": round(float(out["t_s"].iloc[-1]) / 86400.0, 3), "points": len(out),
        "x_end_gcrf_km": out["x_gcrf_km"].iloc[-1], "y_end_gcrf_km": out["y_gcrf_km"].iloc[-1],
        "z_end_gcrf_km": out["z_gcrf_km"].iloc[-1],
        "r_end_minus_R_km": rn - R_E,
    }
    warn = []
    if len(chk):
        d["h_mean_start_km"] = chk["a_table_km"].iloc[0] - R_E      # (ha+hp)/2 по таблице
        d["h_mean_end_km"] = chk["a_table_km"].iloc[-1] - R_E
        # согласованность OPM и таблицы в начальной точке
        d["calib_da_m"] = (chk["a_model_km"].iloc[0] - chk["a_table_km"].iloc[0]) * 1e3
        d["calib_e_ratio"] = (chk["e_model"].iloc[0] / chk["e_table"].iloc[0]
                              if chk["e_table"].iloc[0] > 1e-7 else float("nan"))
        d["di0_deg"] = chk["di_deg"].iloc[0]
        if abs(d["calib_da_m"]) > CALIB_DA_WARN_M:
            warn.append(f"a по OPM и таблице расходятся на {d['calib_da_m'] / 1e3:+.2f} км")
        if abs(d["di0_deg"]) > CALIB_DI_WARN_DEG:
            warn.append(f"наклонение по OPM и таблице расходится на {d['di0_deg']:+.3f}°")
    if len(chk) > 1:
        c2 = chk.iloc[1:]
        d.update(max_da_m=c2.da_m.abs().max(), max_de=c2.de.abs().max(),
                 max_di_deg=c2.di_deg.abs().max(), max_draan_deg=c2.draan_deg.abs().max())
    d["warning"] = "; ".join(warn)
    return d


# --------------------------------------------------------------------------- пакет
def run_batch(opm_path, excel, out_dir="results", sheet=None, workers=None, recursive=False,
              plot=True, plot_time=None, plot_frame="gcrf", coastline="auto", **opts):
    """Пакетный расчёт. Возвращает DataFrame сводки (он же пишется в summary.csv)."""
    os.makedirs(out_dir, exist_ok=True)
    files = find_opm_files(opm_path, recursive)
    opms, bad = read_all_opm(files)
    print(f"OPM: найдено {len(files)}, прочитано {len(opms)}" + (f", с ошибками {len(bad)}" if bad else ""))
    for f, e in bad:
        print(f"  ! {os.path.basename(f)}: {e}")

    table = sc.load_table(excel, sheet)
    blocks, warns = split_blocks(table)
    print(f"Таблица: {len(table)} строк, КА: {len(blocks)}")
    for w in warns:
        print(f"  ! {w}")

    jobs, rows = [], []
    used = set()
    for b in blocks:
        opm, why = match_opm(b, opms)
        if opm is None:
            rows.append({"scname": b["scname"], "scid": b["scid"], "status": "skipped",
                         "error": why, "table_rows": len(b["rows"]), "first_excel_row": b["first_row"]})
            continue
        used.add(opm["file"])
        stem = f"{safe_name(b['scname'])}_{safe_name(b['scid'])}"
        jobs.append({"scname": b["scname"], "scid": b["scid"], "opm": opm, "rows": b["rows"],
                     "opts": opts, "first_row": b["first_row"],
                     "out_file": os.path.join(out_dir, f"{stem}_xyz.csv"),
                     "log_file": os.path.join(out_dir, f"{stem}.log")})
    for o in opms:
        if o["file"] not in used:
            rows.append({"scname": o["name"], "scid": sc.norm_id(o["scid"]), "status": "skipped",
                         "error": "нет строк в таблице (или выбран другой OPM этого КА)",
                         "opm_file": o["file"]})

    workers = workers or min(len(jobs), os.cpu_count() or 1) or 1
    print(f"Расчёт {len(jobs)} КА, процессов: {workers}")
    t0 = _time.time()
    results = []
    if workers == 1:
        for j in jobs:
            results.append(_report(run_one(j), len(results) + 1, len(jobs)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_one, j): j for j in jobs}
            for f in as_completed(futs):
                results.append(_report(f.result(), len(results) + 1, len(jobs)))
    print(f"Готово за {_time.time() - t0:.0f} с")

    order = {(j["scname"], j["scid"]): j["first_row"] for j in jobs}
    results.sort(key=lambda r: order.get((r["scname"], r["scid"]), 0))   # порядок как в таблице
    for r in results:
        r["first_excel_row"] = order.get((r["scname"], r["scid"]))
        if r.get("warning"):
            print(f"  ! {r['scname']} ({r['scid']}): {r['warning']}")
    summary = pd.DataFrame(results + rows)
    lead = ["scname", "scid", "status", "error", "warning", "opm_file", "first_excel_row", "table_rows",
            "t_start", "t_end", "days", "points", "h_mean_start_km", "h_mean_end_km",
            "max_da_m", "max_de", "max_di_deg", "max_draan_deg"]
    summary = summary[[c for c in lead if c in summary] + [c for c in summary if c not in lead]]
    summary_path = os.path.join(out_dir, "summary.csv")
    if "first_excel_row" in summary:
        summary["first_excel_row"] = summary["first_excel_row"].astype("Int64")
    for c in ("table_rows", "points"):
        if c in summary:
            summary[c] = summary[c].astype("Int64")
    summary.to_csv(summary_path, index=False, float_format="%.6g")
    n_ok = int((summary.status == "ok").sum())
    print(f"Сводка: {summary_path}  (успешно {n_ok}, ошибок {int((summary.status == 'error').sum())}, "
          f"пропущено {int((summary.status == 'skipped').sum())})")

    if plot and n_ok:
        import matplotlib
        matplotlib.use("Agg")
        from plot_fleet import plot_fleet_all
        for p in plot_fleet_all(summary_path, out_dir, plot_time, plot_frame, coastline):
            print(f"Рисунок: {p}")
    return summary


def _report(res, k, n):
    if res["status"] == "ok":
        extra = f"{res.get('days', 0):.2f} сут, |Δa|max {res.get('max_da_m', float('nan')):.1f} м"
    else:
        extra = res["error"]
    print(f"  [{k}/{n}] {res['scname']} ({res['scid']}): {res['status']}, {extra}, {res['runtime_s']} с")
    return res


def main():
    ap = argparse.ArgumentParser(description="Пакетный расчёт координат группы КА")
    ap.add_argument("--opm-dir", required=True, help="папка с .opm, маска (opm/*.opm) или файл")
    ap.add_argument("--excel", required=True, help="таблица .xlsx/.csv с блоками КА подряд")
    ap.add_argument("--sheet", default=None, help="лист Excel (по умолчанию первый)")
    ap.add_argument("--out-dir", default="results", help="папка результатов")
    ap.add_argument("--workers", type=int, default=None, help="число процессов (по умолчанию = ядер)")
    ap.add_argument("--recursive", action="store_true", help="искать OPM во вложенных папках")
    ap.add_argument("--step", type=float, default=60.0, help="шаг выдачи координат, с")
    ap.add_argument("--re", type=float, default=R_E, help="экваториальный радиус для ha/hp, км")
    ap.add_argument("--j2", type=float, default=None, help="J2 (по умолчанию по dRaanDt)")
    ap.add_argument("--dut1", type=float, default=0.0, help="UT1-UTC, с")
    ap.add_argument("--absolute", action="store_true", help="без калибровки a, e по OPM")
    ap.add_argument("--iters", type=int, default=2, help="итераций стрельбы на отрезок")
    ap.add_argument("--no-plot", action="store_true", help="не строить рисунки группы")
    ap.add_argument("--plot-time", default=None, help="момент рисунков UTC (по умолчанию последний общий)")
    ap.add_argument("--plot-frame", default="gcrf", choices=["gcrf", "itrf"])
    ap.add_argument("--no-coast", action="store_true", help="рисовать без материков")
    a = ap.parse_args()
    summary = run_batch(a.opm_dir, a.excel, a.out_dir, a.sheet, a.workers, a.recursive,
                        plot=not a.no_plot, plot_time=a.plot_time, plot_frame=a.plot_frame,
                        coastline=None if a.no_coast else "auto",
                        step=a.step, re=a.re, j2=a.j2, dut1=a.dut1, absolute=a.absolute,
                        iters=a.iters)
    if not (summary.status == "ok").any():
        return 2                                   # ничего не рассчитано
    return 0 if (summary.status != "error").all() else 1


if __name__ == "__main__":
    sys.exit(main())
