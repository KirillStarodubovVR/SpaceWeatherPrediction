#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Статистика космической погоды по файлу CelesTrak SpaceWeather-All-v1.2
(формат: https://celestrak.org/SpaceData/SpaceWx-format.php).

Используется только секция OBSERVED. Прогнозные секции игнорируются.

Разброс на заданную дату:
  * Ap (AP_AVG)                     - метод A: календарная климатология
                                      (все годы, окно +-N суток вокруг дня года);
  * F10.7_ADJ и F10.7_ADJ_LAST81    - метод B: по фазе солнечного цикла
                                      (годы от последнего минимума, окно +-M лет).

Пример запуска:
    python spacewx_stats.py SpaceWeather-All-v1_2.txt --date 2031-06-15 --plots

Пример использования как модуля:
    from spacewx_stats import SpaceWeatherStats
    sw = SpaceWeatherStats("SpaceWeather-All-v1_2.txt")
    res = sw.spread_for_date("2031-06-15")
    print(res.summary)
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25

# Позиции колонок в формате legacy (нумерация с 1, границы включительно).
FIELDS = {
    "AP_AVG": (80, 82),
    "ISN": (90, 92),
    "F10.7_ADJ": (94, 98),
    "Q": (100, 100),
    "F10.7_ADJ_CENTER81": (102, 106),
    "F10.7_ADJ_LAST81": (108, 112),
    "F10.7_OBS": (114, 118),
    "F10.7_OBS_CENTER81": (120, 124),
    "F10.7_OBS_LAST81": (126, 130),
}
F107_DAILY_COLS = ("F10.7_ADJ", "F10.7_OBS")
F107_COLS = tuple(c for c in FIELDS if c.startswith("F10.7"))

# Официальные минимумы (SILSO, 13-месячное сглаживание). Первый - минимум
# перед 19-м циклом; он лежит до начала данных и нужен для фазы 1957-1964 гг.
PRE_DATA_MINIMUM = pd.Timestamp("1954-04-15")
REFERENCE_MINIMA = pd.to_datetime([
    "1964-10-15", "1976-03-15", "1986-09-15",
    "1996-08-15", "2008-12-15", "2019-12-15",
])
FIRST_CYCLE_NUMBER = 19          # цикл, начавшийся в PRE_DATA_MINIMUM
MINIMA_TOLERANCE_DAYS = 92       # допустимое расхождение авто- и официальных дат

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
STAT_COLUMNS = ["n", "mean", "std", "min"] + [f"P{p}" for p in PERCENTILES] + ["max"]


# ----------------------------------------------------------------------------
# Чтение файла
# ----------------------------------------------------------------------------
def load_observed(path: str | Path) -> pd.DataFrame:
    """Читает секцию OBSERVED в DataFrame с индексом DATE. Пустые поля -> NaN."""
    rows, inside, found = [], False, False
    with open(path, encoding="ascii", errors="replace") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\r\n")
            tag = line.strip()
            if tag == "BEGIN OBSERVED":
                inside = found = True
                continue
            if tag == "END OBSERVED":
                break
            if not inside or not tag or tag.startswith("#"):
                continue
            try:
                date = pd.Timestamp(int(line[0:4]), int(line[5:7]), int(line[8:10]))
            except ValueError as exc:
                raise ValueError(f"Строка {lineno}: не разобрана дата: {line!r}") from exc
            rec = {"DATE": date}
            for name, (a, b) in FIELDS.items():
                s = line[a - 1:b].strip()
                try:
                    rec[name] = float(s) if s else np.nan
                except ValueError:
                    rec[name] = np.nan
            rows.append(rec)
    if not found or not rows:
        raise ValueError("Секция OBSERVED не найдена или пуста")
    df = pd.DataFrame(rows).set_index("DATE").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


# ----------------------------------------------------------------------------
# Минимумы солнечных циклов
# ----------------------------------------------------------------------------
def smoothed_monthly_isn(df: pd.DataFrame) -> pd.Series:
    """13-месячное сглаженное среднемесячное ISN (стандартная схема SILSO)."""
    monthly = df["ISN"].resample("MS").mean()
    weights = np.r_[0.5, np.ones(11), 0.5] / 12.0
    return monthly.rolling(13, center=True).apply(lambda x: float(np.dot(x, weights)), raw=True)


def detect_minima(df: pd.DataFrame, half_window_months: int = 48) -> pd.DatetimeIndex:
    """Ищет минимумы сглаженного ISN: точка - минимум в окне +-half_window месяцев,
    окно целиком должно лежать внутри данных (чтобы не ловить краевые эффекты)."""
    sm = smoothed_monthly_isn(df)
    valid = np.flatnonzero(sm.notna().values)
    if valid.size == 0:
        return pd.DatetimeIndex([])
    v = sm.values
    lo, hi = valid[0], valid[-1]
    found = []
    for i in range(lo + half_window_months, hi - half_window_months + 1):
        win = v[i - half_window_months:i + half_window_months + 1]
        if np.isfinite(v[i]) and v[i] <= np.nanmin(win):
            if not found or (i - found[-1]) > half_window_months:  # защита от плато
                found.append(i)
    return pd.DatetimeIndex([sm.index[i] + pd.Timedelta(days=14) for i in found])


def choose_minima(df: pd.DataFrame, source: str = "auto"):
    """Возвращает (минимумы вкл. 1954-04, источник, таблица сравнения)."""
    detected = detect_minima(df)
    ref_in_range = REFERENCE_MINIMA[(REFERENCE_MINIMA >= df.index[0]) &
                                    (REFERENCE_MINIMA <= df.index[-1])]
    comp = pd.DataFrame({"официальный (SILSO)": ref_in_range})
    ok = len(detected) == len(ref_in_range)
    if ok:
        comp["найден по данным"] = detected
        comp["разница, сут"] = (detected - ref_in_range).days
        ok = bool((comp["разница, сут"].abs() <= MINIMA_TOLERANCE_DAYS).all())
    if source == "reference":
        chosen, used = ref_in_range, "официальные даты SILSO (задано параметром)"
    elif ok:
        chosen, used = detected, "найдены по данным (совпадают с SILSO)"
    else:
        chosen, used = ref_in_range, "официальные даты SILSO (автопоиск не совпал)"
        warnings.warn(f"Автопоиск минимумов дал {list(detected.date)}; "
                      "используются официальные даты SILSO.")
    minima = pd.DatetimeIndex([PRE_DATA_MINIMUM]).append(pd.DatetimeIndex(chosen))
    return minima, used, comp


def extend_minima(minima: pd.DatetimeIndex, cycle_length: float,
                  until: pd.Timestamp) -> pd.DatetimeIndex:
    """Достраивает будущие минимумы с шагом cycle_length лет."""
    out = list(minima)
    step = pd.Timedelta(days=cycle_length * DAYS_PER_YEAR)
    while out[-1] <= until:
        out.append(out[-1] + step)
    return pd.DatetimeIndex(out)


def phase_table(dates: pd.DatetimeIndex, minima: pd.DatetimeIndex) -> pd.DataFrame:
    """Для каждой даты: годы от предыдущего минимума (>=0), годы до следующего
    (<0) и номер цикла, начавшегося в предыдущем минимуме."""
    m = minima.values
    idx = np.searchsorted(m, dates.values, side="right") - 1
    if (idx < 0).any() or (idx + 1 >= len(m)).any():
        raise ValueError("Даты вне диапазона известных/достроенных минимумов")
    ns_per_year = DAYS_PER_YEAR * 86400e9
    d_prev = (dates.values - m[idx]).astype("timedelta64[ns]").astype(np.int64) / ns_per_year
    d_next = (dates.values - m[idx + 1]).astype("timedelta64[ns]").astype(np.int64) / ns_per_year
    return pd.DataFrame({"phase": d_prev, "to_next": d_next,
                         "cycle": FIRST_CYCLE_NUMBER + idx}, index=dates)


# ----------------------------------------------------------------------------
# Статистика
# ----------------------------------------------------------------------------
def describe(values) -> dict:
    x = pd.Series(values, dtype=float).dropna()
    out = {"n": int(len(x))}
    if len(x) == 0:
        out.update({k: np.nan for k in STAT_COLUMNS[1:]})
        return out
    out["mean"] = x.mean()
    out["std"] = x.std(ddof=1) if len(x) > 1 else np.nan
    out["min"] = x.min()
    for p, v in zip(PERCENTILES, np.percentile(x, PERCENTILES)):
        out[f"P{p}"] = v
    out["max"] = x.max()
    return out


def percentile_of(values, v: float) -> float:
    x = pd.Series(values, dtype=float).dropna().values
    if len(x) == 0 or not np.isfinite(v):
        return np.nan
    return 100.0 * ((x < v).sum() + 0.5 * (x == v).sum()) / len(x)


def noleap_doy(dates: pd.DatetimeIndex) -> np.ndarray:
    """День года 1..365 по невисокосному календарю (29 февраля -> 28 февраля)."""
    month, day = dates.month.values, dates.day.values.copy()
    day[(month == 2) & (day == 29)] = 28
    return pd.to_datetime(pd.DataFrame({"year": 2001, "month": month, "day": day})).dt.dayofyear.values


def circular_doy_distance(doy: np.ndarray, target: int) -> np.ndarray:
    d = np.abs(doy - target)
    return np.minimum(d, 365 - d)


@dataclass
class DateSpread:
    date: pd.Timestamp
    summary: pd.DataFrame          # параметры x статистики (+ факт и его перцентиль)
    per_cycle: pd.DataFrame        # вклад отдельных циклов в выборку метода B
    info: dict                     # фаза, день года, окна и т.п.
    samples: dict = field(repr=False, default_factory=dict)


class SpaceWeatherStats:
    def __init__(self, path, start_year: int | None = None, cycle_length: float = 11.0,
                 exclude_interpolated: bool = False, minima_source: str = "auto",
                 ap_col: str = "AP_AVG", f107_col: str = "F10.7_ADJ",
                 f107avg_col: str = "F10.7_ADJ_LAST81"):
        for c in (ap_col, f107_col, f107avg_col):
            if c not in FIELDS:
                raise ValueError(f"Неизвестная колонка {c}; доступны: {list(FIELDS)}")
        full = load_observed(path)
        # минимумы ищем по полному ряду, до фильтрации по годам
        self.minima, self.minima_source, self.minima_comparison = choose_minima(full, minima_source)
        df = full.copy()
        if exclude_interpolated:
            df.loc[df["Q"] == 4, list(F107_DAILY_COLS)] = np.nan
        if start_year is not None:
            df = df[df.index.year >= start_year]
        if df.empty:
            raise ValueError("После фильтрации не осталось данных")
        self.df = df
        self.cycle_length = cycle_length
        self.ap_col, self.f107_col, self.f107avg_col = ap_col, f107_col, f107avg_col
        self.params = [ap_col, f107_col, f107avg_col]
        self.doy = noleap_doy(df.index)
        self.phases = phase_table(df.index, extend_minima(self.minima, cycle_length, df.index[-1]))

    # --- общая статистика ----------------------------------------------------
    def overall_stats(self) -> pd.DataFrame:
        rows = []
        groups = [("весь период", self.df.index.year.min(), self.df)]
        groups += [("год", y, g) for y, g in self.df.groupby(self.df.index.year)]
        groups += [("цикл", c, g) for c, g in self.df.groupby(self.phases["cycle"])]
        for kind, key, g in groups:
            for p in self.params:
                rows.append({"группа": kind, "ключ": key, "параметр": p, **describe(g[p])})
        return pd.DataFrame(rows)

    # --- метод A ---------------------------------------------------------------
    def _ap_mask(self, target_doy: int, window_days: int) -> np.ndarray:
        return circular_doy_distance(self.doy, target_doy) <= window_days

    # --- метод B ---------------------------------------------------------------
    def _phase_mask(self, target_phase: float, window_years: float):
        """Сутки, чья фаза (от предыдущего минимума или, у начала цикла,
        отрицательная - до следующего) попадает в окно. Возвращает маску и
        номер цикла, по минимуму которого выполнено выравнивание."""
        ph = self.phases
        by_prev = (ph["phase"] - target_phase).abs().values <= window_years
        by_next = (ph["to_next"] - target_phase).abs().values <= window_years
        aligned_cycle = np.where(by_prev, ph["cycle"].values, ph["cycle"].values + 1)
        return by_prev | by_next, aligned_cycle

    # --- разброс на дату -------------------------------------------------------
    def spread_for_date(self, date, ap_window_days: int = 15,
                        phase_window_years: float = 0.5) -> DateSpread:
        date = pd.Timestamp(date).normalize()
        if date < PRE_DATA_MINIMUM:
            raise ValueError(f"Дата раньше {PRE_DATA_MINIMUM.date()} не поддерживается")
        minima_ext = extend_minima(self.minima, self.cycle_length, max(date, self.df.index[-1]))
        tph = phase_table(pd.DatetimeIndex([date]), minima_ext).iloc[0]
        target_doy = int(noleap_doy(pd.DatetimeIndex([date]))[0])
        prev_min = minima_ext[np.searchsorted(minima_ext.values, date.to_datetime64(), side="right") - 1]
        assumed = prev_min > self.minima[-1]

        ap_mask = self._ap_mask(target_doy, ap_window_days)
        ph_mask, aligned_cycle = self._phase_mask(tph["phase"], phase_window_years)
        samples = {
            self.ap_col: self.df.loc[ap_mask, self.ap_col],
            self.f107_col: self.df.loc[ph_mask, self.f107_col],
            self.f107avg_col: self.df.loc[ph_mask, self.f107avg_col],
        }
        actual = self.df.loc[date] if date in self.df.index else None

        rows = []
        for p, s in samples.items():
            method = "A (день года)" if p == self.ap_col else "B (фаза цикла)"
            row = {"параметр": p, "метод": method, **describe(s)}
            if actual is not None:
                row["факт"] = actual[p]
                row["перцентиль факта"] = percentile_of(s, actual[p])
            rows.append(row)
        summary = pd.DataFrame(rows).set_index("параметр")

        pc = pd.DataFrame({"цикл": aligned_cycle[ph_mask],
                           self.f107_col: self.df.loc[ph_mask, self.f107_col].values,
                           self.f107avg_col: self.df.loc[ph_mask, self.f107avg_col].values})
        per_cycle = pc.groupby("цикл").agg(
            n=(self.f107_col, "count"),
            **{f"медиана {self.f107_col}": (self.f107_col, "median"),
               f"медиана {self.f107avg_col}": (self.f107avg_col, "median")})

        info = {
            "дата": date.date(),
            "день года (невисок.)": target_doy,
            "окно Ap, сут": ap_window_days,
            "начало цикла (минимум)": prev_min.date(),
            "номер цикла": int(tph["cycle"]),
            "минимум допущен (не наблюдён)": bool(assumed),
            "фаза, лет от минимума": round(float(tph["phase"]), 3),
            "окно фазы, лет": phase_window_years,
            "циклов в выборке F10.7": int((per_cycle["n"] > 0).sum()),
            "данные": f"{self.df.index[0].date()} .. {self.df.index[-1].date()}",
        }
        if info["циклов в выборке F10.7"] < 3:
            warnings.warn(f"Фаза {tph['phase']:.2f} г. встречается лишь в "
                          f"{info['циклов в выборке F10.7']} цикле(ах): выборка мала.")
        return DateSpread(date, summary, per_cycle, info, samples)

    # --- таблицы на весь год / все фазы ---------------------------------------
    def ap_table_by_doy(self, ap_window_days: int = 15) -> pd.DataFrame:
        rows = []
        for d in range(1, 366):
            label = (pd.Timestamp("2001-01-01") + pd.Timedelta(days=d - 1)).strftime("%m-%d")
            rows.append({"день года": d, "дата (ММ-ДД)": label,
                         **describe(self.df.loc[self._ap_mask(d, ap_window_days), self.ap_col])})
        return pd.DataFrame(rows)

    def f107_table_by_phase(self, phase_window_years: float = 0.5,
                            step_years: float = 1 / 12) -> pd.DataFrame:
        rows = []
        max_phase = float(self.phases["phase"].max())
        for ph in np.arange(0.0, max_phase + 1e-9, step_years):
            mask, cyc = self._phase_mask(ph, phase_window_years)
            for p in (self.f107_col, self.f107avg_col):
                rows.append({"фаза, лет": round(ph, 4), "параметр": p,
                             "циклов": int(len(np.unique(cyc[mask]))),
                             **describe(self.df.loc[mask, p])})
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------------
def make_plots(sw: SpaceWeatherStats, res: DateSpread, outdir: Path,
               ap_window_days: int, phase_window_years: float) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files, tag = [], res.date.strftime("%Y-%m-%d")

    # 1. распределения на дату
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (p, s) in zip(axes, res.samples.items()):
        s = s.dropna()
        ax.hist(s, bins=60, color="#6a8caf", alpha=0.85)
        if p == sw.ap_col:
            ax.set_yscale("log")
        st = res.summary.loc[p]
        for key, ls in (("P5", ":"), ("P50", "-"), ("P95", ":")):
            ax.axvline(st[key], color="k", ls=ls, lw=1, label=f"{key} = {st[key]:.1f}")
        if "факт" in st and np.isfinite(st["факт"]):
            ax.axvline(st["факт"], color="red", lw=1.5, label=f"факт = {st['факт']:.1f}")
        ax.set_title(f"{p} — метод {st['метод'][0]} (n={int(st['n'])})")
        ax.legend(fontsize=8)
    fig.suptitle(f"Разброс на {tag}")
    fig.tight_layout()
    files.append(outdir / f"distribution_{tag}.png")
    fig.savefig(files[-1], dpi=120)
    plt.close(fig)

    # 2. F10.7 всех циклов по фазе
    fig, ax = plt.subplots(figsize=(12, 5))
    ph = sw.phases
    for cyc, g in sw.df.groupby(ph["cycle"]):
        x = ph.loc[g.index, "phase"]
        line, = ax.plot(x, g[sw.f107avg_col], lw=1.5, label=f"цикл {cyc}")
        ax.plot(x, g[sw.f107_col], lw=0.3, alpha=0.3, color=line.get_color())
    t = res.info["фаза, лет от минимума"]
    ax.axvspan(t - phase_window_years, t + phase_window_years, color="orange", alpha=0.25,
               label=f"окно фазы {tag}")
    ax.axvline(t, color="orange")
    ax.set_xlabel("лет от минимума цикла")
    ax.set_ylabel("sfu")
    ax.set_title(f"{sw.f107avg_col} (линии) и {sw.f107_col} (тонко) по фазе цикла")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    files.append(outdir / "f107_by_cycle_phase.png")
    fig.savefig(files[-1], dpi=120)
    plt.close(fig)

    # 3. годовой ход Ap
    tab = sw.ap_table_by_doy(ap_window_days)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.fill_between(tab["день года"], tab["P5"], tab["P95"], alpha=0.2, label="P5–P95")
    ax.fill_between(tab["день года"], tab["P25"], tab["P75"], alpha=0.35, label="P25–P75")
    ax.plot(tab["день года"], tab["P50"], lw=1.5, label="медиана")
    ax.plot(tab["день года"], tab["mean"], lw=1, ls="--", label="среднее")
    ax.axvline(res.info["день года (невисок.)"], color="red", label=tag)
    ax.set_xlabel("день года")
    ax.set_ylabel(sw.ap_col)
    ax.set_title(f"Годовой ход {sw.ap_col} (окно ±{ap_window_days} сут)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    files.append(outdir / "ap_annual.png")
    fig.savefig(files[-1], dpi=120)
    plt.close(fig)
    return files


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _fmt(df: pd.DataFrame) -> str:
    with pd.option_context("display.width", 250, "display.max_columns", 50,
                           "display.float_format", "{:.1f}".format):
        return df.to_string()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Статистика SpaceWeather (CelesTrak) и разброс на дату")
    ap.add_argument("file", help="файл SpaceWeather-All-v1.2 (.txt)")
    ap.add_argument("--date", nargs="*", default=[], help="дата(ы) ГГГГ-ММ-ДД")
    ap.add_argument("--ap-window", type=int, default=15, help="окно метода A, сут (15)")
    ap.add_argument("--phase-window", type=float, default=0.5, help="окно метода B, лет (0.5)")
    ap.add_argument("--cycle-length", type=float, default=11.0,
                    help="длина будущих циклов для достройки минимумов, лет (11.0)")
    ap.add_argument("--start-year", type=int, default=None, help="использовать данные с этого года")
    ap.add_argument("--exclude-interpolated", action="store_true",
                    help="исключить суточные F10.7 с Q=4 (интерполяция CelesTrak)")
    ap.add_argument("--minima", choices=["auto", "reference"], default="auto",
                    help="минимумы циклов: авто по ISN (со сверкой) или официальные")
    ap.add_argument("--ap-col", default="AP_AVG", choices=list(FIELDS))
    ap.add_argument("--f107-col", default="F10.7_ADJ", choices=list(F107_COLS))
    ap.add_argument("--f107avg-col", default="F10.7_ADJ_LAST81", choices=list(F107_COLS))
    ap.add_argument("--outdir", default="spacewx_output", help="папка для CSV и графиков")
    ap.add_argument("--export-tables", action="store_true",
                    help="сохранить таблицы Ap по дням года и F10.7 по фазам")
    ap.add_argument("--plots", action="store_true", help="построить графики для каждой даты")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sw = SpaceWeatherStats(args.file, start_year=args.start_year,
                           cycle_length=args.cycle_length,
                           exclude_interpolated=args.exclude_interpolated,
                           minima_source=args.minima, ap_col=args.ap_col,
                           f107_col=args.f107_col, f107avg_col=args.f107avg_col)

    print(f"Данные OBSERVED: {sw.df.index[0].date()} .. {sw.df.index[-1].date()}, "
          f"{len(sw.df)} сут")
    print(f"\nМинимумы циклов: {sw.minima_source}")
    print(_fmt(sw.minima_comparison))

    stats = sw.overall_stats()
    stats.to_csv(outdir / "stats_overall.csv", index=False, encoding="utf-8-sig")
    whole = stats[stats["группа"] == "весь период"].set_index("параметр")[STAT_COLUMNS]
    print("\nОбщая статистика за весь период:")
    print(_fmt(whole))
    cyc = stats[stats["группа"] == "цикл"].pivot(index="ключ", columns="параметр", values="P50")
    print("\nМедианы по циклам (цикл 19 и последний - неполные):")
    print(_fmt(cyc[sw.params]))
    print(f"\nПодробно (по годам, циклам): {outdir / 'stats_overall.csv'}")

    if args.export_tables:
        sw.ap_table_by_doy(args.ap_window).to_csv(
            outdir / "ap_by_day_of_year.csv", index=False, encoding="utf-8-sig")
        sw.f107_table_by_phase(args.phase_window).to_csv(
            outdir / "f107_by_cycle_phase.csv", index=False, encoding="utf-8-sig")
        print(f"Таблицы: {outdir / 'ap_by_day_of_year.csv'}, {outdir / 'f107_by_cycle_phase.csv'}")

    for d in args.date:
        try:
            pd.Timestamp(d)
        except ValueError:
            print(f"\nОшибка: некорректная дата {d!r} (нужен формат ГГГГ-ММ-ДД)")
            continue
        res = sw.spread_for_date(d, args.ap_window, args.phase_window)
        print("\n" + "=" * 100)
        print(f"РАЗБРОС НА {res.date.date()}")
        for k, v in res.info.items():
            print(f"  {k}: {v}")
        print(_fmt(res.summary))
        print("\nВклад циклов в выборку метода B:")
        print(_fmt(res.per_cycle))
        tag = res.date.strftime("%Y-%m-%d")
        res.summary.to_csv(outdir / f"spread_{tag}.csv", encoding="utf-8-sig")
        if args.plots:
            for f in make_plots(sw, res, outdir, args.ap_window, args.phase_window):
                print(f"  график: {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
