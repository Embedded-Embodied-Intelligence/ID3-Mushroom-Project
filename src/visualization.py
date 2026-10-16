"""Presentation graphics: one palette for every chapter and two ID3 tree renderers.

The tree is exported to DOT and rendered by Graphviz when the ``dot`` program
is installed; otherwise a coloured matplotlib block diagram (same information,
no system dependency) is produced automatically.
"""
from __future__ import annotations

import shutil
import subprocess
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.ticker import FuncFormatter, PercentFormatter

from .data_io import FEATURE_NAMES_PL, expand_value
from .id3 import ID3Classifier, ID3Node, ValueFormatter, entropy_from_counts

# Class pair checked with a colour-vision-deficiency validator (worst CVD ΔE 8.8,
# normal-vision ΔE 25.2, both >= 3:1 against the surface): sea green / coral.
CLASS_COLORS = {"e": "#2a9d8f", "p": "#e76f51"}
CLASS_LABELS = {"e": "jadalny (e)", "p": "trujący (p)"}
NAVY = "#1d3557"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
CONTEXT = "#d8d6cf"
ACCENT = "#2a78d6"
ACCENT_2 = "#4a3aa7"  # second series slot (validated against ACCENT: CVD ΔE 13.0)
INTERNAL_FILL = "#e8edf4"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("id3_blue", BLUE_RAMP)


def set_presentation_style() -> None:
    """Projector-friendly defaults: large type, hairline solid grid, navy titles."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
        "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlecolor": NAVY,
        "axes.titlelocation": "left", "axes.titlepad": 12,
        "axes.labelsize": 11, "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.frameon": False, "legend.fontsize": 10,
        "figure.titlesize": 15, "figure.titleweight": "bold",
        "svg.fonttype": "path",
    })


def save_figure(fig: plt.Figure, stem: str | Path, formats: Sequence[str] = ("png",), dpi: int = 200) -> list[Path]:
    """Save ``fig`` as ``stem.<fmt>`` for every format; returns the written paths."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, format=fmt, dpi=dpi)
        paths.append(path)
    return paths


def _tint(color: str, amount: float) -> str:
    """Mix ``color`` with the surface: amount=1 -> color, 0 -> surface."""
    c, s = np.array(to_rgb(color)), np.array(to_rgb(SURFACE))
    return to_hex(s + (c - s) * amount)


def _class_style(label: Any, class_colors: Mapping[Any, str] | None,
                 class_labels: Mapping[Any, str] | None) -> tuple[str, str]:
    colors = {**CLASS_COLORS, **(class_colors or {})}
    names = {**CLASS_LABELS, **(class_labels or {})}
    return colors.get(label, ACCENT), names.get(label, str(label))


def _heading(name: str) -> str:
    """'jadalny (e)' -> 'JADALNY (e)': emphasise the name, keep the class code verbatim."""
    if name.endswith(")") and " (" in name:
        text, code = name[:-1].rsplit(" (", 1)
        return f"{text.upper()} ({code})"
    return name.upper()


def _percent_axis(ax: plt.Axes, axis: str = "x", decimals: int = 0) -> None:
    formatter = PercentFormatter(1.0, decimals=decimals)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(formatter)


def _class_legend(ax_or_fig: Any, labels: Sequence[Any] = ("e", "p"), **kwargs: Any) -> None:
    handles = [Patch(facecolor=CLASS_COLORS[c], label=CLASS_LABELS[c]) for c in labels]
    ax_or_fig.legend(handles=handles, **kwargs)


# ---------------------------------------------------------------------------
# Chapter figures
# ---------------------------------------------------------------------------

def plot_entropy_curve(highlight: tuple[str, float] | None = None) -> plt.Figure:
    """Binary entropy H(p) with H(0)=0, H(0.5)=1, H(1)=0 marked."""
    p = np.linspace(0.0, 1.0, 501)
    h = np.array([entropy_from_counts([v, 1.0 - v]) for v in p])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.fill_between(p, h, color=ACCENT, alpha=0.08, linewidth=0)
    ax.plot(p, h, color=ACCENT, lw=2.2, solid_capstyle="round")
    for x, text, offset in [(0.0, "H(0) = 0\n(zbiór czysty)", (10, 14)), (0.5, "H(0.5) = 1 bit\n(maks. niepewność)", (0, 12)),
                            (1.0, "H(1) = 0\n(zbiór czysty)", (-10, 14))]:
        y = entropy_from_counts([x, 1 - x])
        ax.scatter([x], [y], s=70, color=ACCENT, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(text, (x, y), xytext=offset, textcoords="offset points", fontsize=10, color=INK,
                    ha="left" if x == 0 else ("right" if x == 1 else "center"), va="bottom")
    if highlight is not None:
        label, x = highlight
        y = entropy_from_counts([x, 1 - x])
        ax.scatter([x], [y], s=90, marker="D", color=NAVY, edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.annotate(f"{label}\np = {x:.4f} → H = {y:.4f} bit", (x, y), xytext=(-75, -150),
                    textcoords="offset points", fontsize=10, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set(xlim=(-0.03, 1.03), ylim=(0, 1.2), xlabel="p — udział jednej z klas w zbiorze S",
           ylabel="H(S) [bit]")
    ax.set_title("Entropia zbioru o dwóch klasach: H(p) = −p·log₂p − (1−p)·log₂(1−p)")
    return fig


def plot_dataset_overview(y: pd.Series, profile: pd.DataFrame) -> plt.Figure:
    """Class balance (stacked bar) and observed vs declared categories per feature."""
    fig = plt.figure(figsize=(11, 8.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 6.5], hspace=0.38)
    ax0, ax1 = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    counts = y.value_counts()
    left = 0.0
    for label in ("e", "p"):
        share = counts.get(label, 0) / len(y)
        ax0.barh([0], [share], left=left, height=0.55, color=CLASS_COLORS[label], edgecolor=SURFACE, linewidth=2)
        ax0.text(left + share / 2, 0, f"{CLASS_LABELS[label]}: {counts.get(label, 0)} ({share:.2%})",
                 ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        left += share
    ax0.set(xlim=(0, 1), yticks=[], xlabel="udział obserwacji")
    _percent_axis(ax0)
    ax0.grid(False)
    ax0.set_title(f"Rozkład klas (n = {len(y)}) — zbiór praktycznie zbalansowany")
    prof = profile.iloc[::-1]
    pos = np.arange(len(prof))
    ax1.barh(pos, prof["kategorie w słowniku"], height=0.62, color=CONTEXT, label="zadeklarowane w słowniku UCI")
    ax1.barh(pos, prof["kategorie obserwowane"], height=0.36, color=ACCENT, label="obserwowane w danych")
    for i, (_, row) in enumerate(prof.iterrows()):
        ax1.text(row["kategorie w słowniku"] + 0.15, i, f"{row['kategorie obserwowane']}/{row['kategorie w słowniku']}",
                 va="center", fontsize=9, color=INK_SECONDARY)
    ax1.set_yticks(pos, [f"{f}  ({FEATURE_NAMES_PL.get(f, '')})" for f in prof["cecha"]])
    ax1.set_xlabel("liczba kategorii")
    ax1.set_xlim(0, prof["kategorie w słowniku"].max() + 1.6)
    ax1.legend(loc="lower right")
    ax1.set_title("Profil cech: liczba kategorii (obserwowane / słownik)")
    ax1.grid(axis="y", visible=False)
    return fig


def plot_missing_overview(df: pd.DataFrame, target: str = "class", feature: str = "stalk-root") -> plt.Figure:
    """Missing-value map (rows sorted by class) and the missing share of ``feature`` per class."""
    features = [c for c in df.columns if c != target]
    order = df.assign(_m=df[feature].isna()).sort_values([target, "_m"], kind="mergesort").index
    missing = df.loc[order, features].isna().to_numpy().astype(int)
    classes = (df.loc[order, target] == "p").to_numpy().astype(int)[:, None]
    fig = plt.figure(figsize=(13.5, 6.4))
    grid = fig.add_gridspec(1, 4, width_ratios=[0.35, 6, 1.1, 3.4], wspace=0.05)
    ax_c, ax_m, ax_b = fig.add_subplot(grid[0]), fig.add_subplot(grid[1]), fig.add_subplot(grid[3])
    ax_c.imshow(classes, aspect="auto", cmap=ListedColormap([CLASS_COLORS["e"], CLASS_COLORS["p"]]),
                interpolation="nearest")
    ax_c.set(xticks=[], yticks=[])
    ax_c.set_ylabel("obserwacje (posortowane wg klasy)")
    ax_c.grid(False)
    ax_m.imshow(missing, aspect="auto", cmap=ListedColormap(["#eef0f3", NAVY]), interpolation="nearest")
    ax_m.set_xticks(range(len(features)), features, rotation=90, fontsize=9)
    ax_m.set_yticks([])
    ax_m.grid(False)
    ax_m.set_title("Mapa braków (wiersze = obserwacje)", fontsize=13)
    fig.legend(handles=[Patch(facecolor=NAVY, label="brak (NaN po zamianie '?')"),
                        Patch(facecolor="#eef0f3", edgecolor=AXIS, label="wartość obecna"),
                        Patch(facecolor=CLASS_COLORS["e"], label=CLASS_LABELS["e"]),
                        Patch(facecolor=CLASS_COLORS["p"], label=CLASS_LABELS["p"])],
               loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, fontsize=10)
    stats = df.groupby(target)[feature].apply(lambda s: (int(s.isna().sum()), len(s)))
    labels = list(stats.index)
    shares = [stats[c][0] / stats[c][1] for c in labels]
    ax_b.bar(range(len(labels)), shares, width=0.5, color=[CLASS_COLORS[c] for c in labels])
    for i, c in enumerate(labels):
        ax_b.text(i, shares[i] + 0.012, f"{stats[c][0]} / {stats[c][1]}\n({shares[i]:.2%})", ha="center",
                  va="bottom", fontsize=10, color=INK)
    overall = df[feature].isna().mean()
    ax_b.axhline(overall, color=MUTED, lw=1)
    ax_b.text(-0.42, overall + 0.01, f"ogółem {overall:.2%}", ha="left", va="bottom", fontsize=9,
              color=INK_SECONDARY)
    ax_b.set_xticks(range(len(labels)), [CLASS_LABELS[c] for c in labels])
    ax_b.set_ylim(0, max(shares) * 1.35)
    _percent_axis(ax_b, "y")
    ax_b.set_title(f"Braki '{feature}' wg klasy", fontsize=13)
    ax_b.grid(axis="x", visible=False)
    return fig


def plot_category_distributions(df: pd.DataFrame, features: Sequence[str], target: str = "class",
                                ncols: int = 3) -> plt.Figure:
    """Small multiples: class composition of every category of the selected features."""
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.5 * nrows))
    for ax, feature in zip(np.ravel(axes), features):
        values = df[feature].astype(object).where(df[feature].notna(), "__MISSING__")
        table = pd.crosstab(values, df[target]).reindex(columns=["e", "p"], fill_value=0)
        table = table.loc[table.sum(axis=1).sort_values().index]
        pos = np.arange(len(table))
        ax.barh(pos, table["e"], height=0.62, color=CLASS_COLORS["e"], edgecolor=SURFACE, linewidth=1.5)
        ax.barh(pos, table["p"], left=table["e"], height=0.62, color=CLASS_COLORS["p"], edgecolor=SURFACE,
                linewidth=1.5)
        ax.set_yticks(pos, [expand_value(feature, v) for v in table.index], fontsize=9)
        ax.set_title(f"{feature} — {FEATURE_NAMES_PL.get(feature, '')}", fontsize=12)
        ax.set_xlabel("liczba obserwacji", fontsize=9)
        ax.grid(axis="y", visible=False)
    for ax in np.ravel(axes)[len(features):]:
        ax.set_visible(False)
    _class_legend(fig, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03), fontsize=11)
    fig.tight_layout()
    return fig


def plot_cramers_v_heatmap(matrix: pd.DataFrame, title: str) -> plt.Figure:
    """Annotated Cramér's V matrix (sequential single-hue scale 0..1)."""
    size = 0.62 * len(matrix) + 2.5
    fig, ax = plt.subplots(figsize=(size + 1.2, size))
    sns.heatmap(matrix, cmap=SEQUENTIAL, vmin=0, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 8.5},
                square=True, linewidths=1.5, linecolor=SURFACE, cbar_kws={"label": "V Craméra", "shrink": 0.75},
                ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)
    return fig


def plot_split_balance(summary: pd.DataFrame) -> plt.Figure:
    """Class shares in train / validation / test (stratification check)."""
    fig, ax = plt.subplots(figsize=(10, 3.4))
    pos = np.arange(len(summary))[::-1]
    for i, (_, row) in zip(pos, summary.iterrows()):
        left = 0.0
        for label in ("e", "p"):
            share = row[f"% {label}"]
            ax.barh(i, share, left=left, height=0.58, color=CLASS_COLORS[label], edgecolor=SURFACE, linewidth=2)
            ax.text(left + share / 2, i, f"{label}: {int(row[f'n {label}'])} ({share:.2%})", ha="center",
                    va="center", color="white", fontsize=10, fontweight="bold")
            left += share
        ax.text(1.01, i, f"n = {int(row['n'])} ({row['udział']:.1%})", va="center", fontsize=10, color=INK)
    ax.set_yticks(pos, [f"zbiór {name}" for name in summary["zbiór"]])
    ax.set_xlim(0, 1)
    _percent_axis(ax)
    ax.grid(False)
    ax.set_title("Podział stratyfikowany 60 / 20 / 20 — identyczne proporcje klas")
    return fig


def plot_information_gain_ranking(table: pd.DataFrame, title: str) -> plt.Figure:
    """IG ranking (winner highlighted) next to Gain Ratio for the same attributes."""
    data = table.iloc[::-1]
    pos = np.arange(len(data))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 0.36 * len(data) + 2.2), sharey=True,
                                   gridspec_kw={"wspace": 0.08})
    best = table.iloc[0]["feature"]
    colors = [ACCENT if f == best else CONTEXT for f in data["feature"]]
    ax1.barh(pos, data["information_gain"], height=0.62, color=colors)
    ax2.barh(pos, data["gain_ratio"], height=0.62, color=[ACCENT_2 if f == best else CONTEXT for f in data["feature"]])
    for ax, column in ((ax1, "information_gain"), (ax2, "gain_ratio")):
        top = set(data.sort_values(column, ascending=False)["feature"].head(4))
        for i, (_, row) in enumerate(data.iterrows()):
            if row["feature"] in top:
                ax.text(row[column] + 0.01, i, f"{row[column]:.3f}", va="center", fontsize=9, color=INK)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(data[column].max() * 1.18, 0.05))
    ax1.set_yticks(pos, [f"{f} ({n})" for f, n in zip(data["feature"], data["n_values"])])
    ax1.set_xlabel("IG(S, A) [bit]  —  kryterium ID3")
    ax2.set_xlabel("GainRatio = IG / SplitInfo  —  korekta C4.5")
    ax1.set_title(title)
    ax2.set_title("Ta sama lista wg Gain Ratio", color=INK_SECONDARY, fontsize=12)
    return fig


def plot_root_split(X: pd.DataFrame, y: pd.Series, feature: str, title: str) -> plt.Figure:
    """Class counts in every branch of a split, with branch entropy H(S_v)."""
    table = pd.crosstab(X[feature], y).reindex(columns=["e", "p"], fill_value=0)
    table = table.loc[table.sum(axis=1).sort_values().index]
    pos = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(table) + 1.8))
    ax.barh(pos, table["e"], height=0.6, color=CLASS_COLORS["e"], edgecolor=SURFACE, linewidth=1.5)
    ax.barh(pos, table["p"], left=table["e"], height=0.6, color=CLASS_COLORS["p"], edgecolor=SURFACE, linewidth=1.5)
    for i, (value, row) in enumerate(table.iterrows()):
        h = entropy_from_counts(row.to_numpy())
        ax.text(row.sum() + table.to_numpy().sum() * 0.006, i,
                f"n = {row.sum()}  (e {row['e']} / p {row['p']})   H = {h:.3f}", va="center", fontsize=9.5)
    ax.set_yticks(pos, [expand_value(feature, v) for v in table.index])
    ax.set_xlim(0, table.sum(axis=1).max() * 1.55)
    ax.set_xlabel("liczba obserwacji w gałęzi")
    ax.grid(axis="y", visible=False)
    _class_legend(ax, loc="lower right")
    ax.set_title(title)
    return fig


def plot_experiment_heatmap(results: pd.DataFrame) -> plt.Figure:
    """Validation accuracy and p -> e errors of all 36 configurations."""
    data = results.copy()
    data["wiersz"] = data["strategy"] + "  |  max_depth=" + data["max_depth_label"]
    data["kolumna"] = ("min_samples_split=" + data["min_samples_split"].astype(str) + "\nmin_information_gain="
                       + data["min_information_gain"].map(lambda v: f"{v:g}"))
    strategy_order = {name: i for i, name in enumerate(["missing_category", "most_frequent", "drop_stalk_root"])}
    data["_s"] = data["strategy"].map(strategy_order)
    data["_d"] = [np.inf if label == "None" else float(label) for label in data["max_depth_label"]]
    rows = list(dict.fromkeys(data.sort_values(["_s", "_d"])["wiersz"]))
    cols = list(dict.fromkeys(data.sort_values(["min_samples_split", "min_information_gain"])["kolumna"]))
    acc = data.pivot(index="wiersz", columns="kolumna", values="val_accuracy").loc[rows, cols]
    ids = data.pivot(index="wiersz", columns="kolumna", values="config_id").loc[rows, cols]
    pe = data.pivot(index="wiersz", columns="kolumna", values="val_p_to_e").loc[rows, cols]
    notes = np.array([[f"{ids.iat[i, j]}: {acc.iat[i, j]:.2%}\np→e = {int(pe.iat[i, j])}" for j in range(acc.shape[1])]
                      for i in range(acc.shape[0])])
    fig, ax = plt.subplots(figsize=(12.5, 7.6))
    sns.heatmap(acc, cmap=SEQUENTIAL, vmin=acc.to_numpy().min() - 0.002, vmax=1.0, annot=notes, fmt="",
                annot_kws={"size": 9.5}, linewidths=2, linecolor=SURFACE,
                cbar_kws={"label": "accuracy (walidacja)", "format": FuncFormatter(lambda v, _: f"{v:.1%}")}, ax=ax)
    for boundary in (3, 6):
        ax.axhline(boundary, color=SURFACE, lw=6)
    ax.set(xlabel="", ylabel="")
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.set_title("36 modeli próbnych: accuracy na walidacji i liczba błędów krytycznych p → e")
    return fig


def plot_depth_curve(curve: pd.DataFrame, strategy: str, identical_across_strategies: bool) -> plt.Figure:
    """Train vs validation accuracy and validation errors as functions of max_depth."""
    data = curve[curve["strategy"] == strategy].reset_index(drop=True)
    pos = np.arange(len(data))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True, gridspec_kw={"height_ratios": [3, 2],
                                                                                         "hspace": 0.12})
    ax1.plot(pos, data["train_accuracy"], color=MUTED, lw=2, marker="o", ms=7, mec=SURFACE, mew=2,
             label="zbiór uczący")
    ax1.plot(pos, data["val_accuracy"], color=ACCENT, lw=2.4, marker="o", ms=8, mec=SURFACE, mew=2,
             label="zbiór walidacyjny")
    for i, v in enumerate(data["val_accuracy"]):
        if v < 1.0 or i == int(np.argmax(data["val_accuracy"].to_numpy() >= 1.0)):
            ax1.annotate(f"{v:.2%}", (i, v), xytext=(-6, 9) if v < 1.0 else (0, -20), textcoords="offset points",
                         ha="right" if v < 1.0 else "center", fontsize=9.5, color=INK)
    full = data.index[data["val_accuracy"] >= 1.0]
    if len(full):
        first = int(full[0])
        ax1.axvspan(first - 0.35, len(data) - 0.65, color=ACCENT, alpha=0.06, lw=0)
        ax1.text(first - 0.3, data["val_accuracy"].min() + 0.0005,
                 f"100% na walidacji od max_depth = {data.loc[first, 'max_depth_label']}", fontsize=10, color=NAVY)
    _percent_axis(ax1, "y", 1)
    ax1.set_ylabel("accuracy")
    ax1.legend(loc="lower right")
    note = " (krzywe identyczne dla 3 strategii braków)" if identical_across_strategies else f" ({strategy})"
    ax1.set_title("Krzywa walidacyjna max_depth" + note)
    width = 0.34
    bars_pe = ax2.bar(pos - width / 2, data["val_p_to_e"], width, color=CLASS_COLORS["p"],
                      label="p → e (trujący uznany za jadalny)")
    bars_ep = ax2.bar(pos + width / 2, data["val_e_to_p"], width, color=CLASS_COLORS["e"],
                      label="e → p (jadalny uznany za trujący)")
    for bars in (bars_pe, bars_ep):
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{int(bar.get_height())}",
                     ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(0, max(data["val_p_to_e"].max(), data["val_e_to_p"].max(), 1) * 1.3)
    ax2.set_ylabel("błędy na walidacji")
    ax2.legend(loc="upper right")
    ax2.grid(axis="x", visible=False)
    ax2.set_xticks(pos, [f"{d}\n{int(n)} liści" for d, n in zip(data["max_depth_label"], data["leaf_count"])])
    ax2.set_xlabel("max_depth (liczba liści drzewa)")
    return fig


def plot_train_vs_validation(results: pd.DataFrame) -> plt.Figure:
    """Train vs validation accuracy of the 36 configurations (bubble = number of configurations)."""
    groups = results.groupby(["train_accuracy", "val_accuracy"]).agg(
        n=("config_id", "size"), depths=("max_depth_label", lambda s: ", ".join(sorted(set(s)))),
        leaves=("leaf_count", "first")).reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    low = min(groups["train_accuracy"].min(), groups["val_accuracy"].min())
    lo, hi = low - 0.0025, 1.0015
    ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1, zorder=1)
    ax.text(hi - 0.0003, hi - 0.0012, "train = walidacja", rotation=45, ha="right", va="top", fontsize=9,
            color=MUTED)
    ax.scatter(groups["train_accuracy"], groups["val_accuracy"], s=groups["n"] * 55, color=ACCENT, alpha=0.85,
               edgecolor=SURFACE, linewidth=2, zorder=3)
    for _, row in groups.iterrows():
        ax.annotate(f"{row['n']} konfiguracji\nmax_depth ∈ {{{row['depths']}}}\n{row['leaves']} liści",
                    (row["train_accuracy"], row["val_accuracy"]), xytext=(-150, -12), textcoords="offset points",
                    fontsize=9.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="accuracy — zbiór uczący", ylabel="accuracy — zbiór walidacyjny")
    _percent_axis(ax, "x", 1)
    _percent_axis(ax, "y", 1)
    ax.set_title("Train vs walidacja: 36 konfiguracji w 2 punktach")
    return fig


def plot_strategy_gains(frame: pd.DataFrame) -> plt.Figure:
    """Gain of the attribute ID3 chose vs the gain of stalk-root under two imputations."""
    data = frame.iloc[::-1].reset_index(drop=True)
    pos = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(11, 0.75 * len(data) + 2.2))
    ax.barh(pos, data["winner_gain"], height=0.55, color=CONTEXT, label="atrybut wybrany przez ID3")
    for i, row in data.iterrows():
        ax.text(row["winner_gain"] + 0.01, i, f"{row['winner']} (IG = {row['winner_gain']:.3f})", va="center",
                fontsize=9.5, color=INK)
    styles = [("gain_missing_category", "rank_missing_category", ACCENT, "stalk-root: missing_category"),
              ("gain_most_frequent", "rank_most_frequent", ACCENT_2, "stalk-root: most_frequent")]
    for k, (gain_col, rank_col, color, label) in enumerate(styles):
        offset = 0.12 if k == 0 else -0.12
        ax.scatter(data[gain_col], pos + offset, s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=3,
                   label=label)
        for i, row in data.iterrows():
            near_axis = row[gain_col] < 0.06 * data["winner_gain"].max()
            ax.annotate(f"#{int(row[rank_col])}", (row[gain_col], i + offset), xytext=(8 if near_axis else -8, 0),
                        textcoords="offset points", ha="left" if near_axis else "right", va="center",
                        fontsize=8.5, color=INK_SECONDARY)
    ax.set_yticks(pos, data["node"])
    ax.set_xlim(0, data["winner_gain"].max() * 1.45)
    ax.set_xlabel("IG w węźle [bit]   (#k = pozycja stalk-root w rankingu kandydatów)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    ax.set_title("Dlaczego strategia braków nie zmienia drzewa: stalk-root nigdy nie wygrywa")
    return fig


def plot_feature_importance(importance: pd.Series, title: str) -> plt.Figure:
    """Normalised weighted information gain of the attributes used by the tree."""
    used = importance[importance > 0].sort_values()
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(used) + 1.8))
    ax.barh(np.arange(len(used)), used.to_numpy(), height=0.55, color=ACCENT)
    for i, v in enumerate(used.to_numpy()):
        ax.text(v + 0.01, i, f"{v:.2%}", va="center", fontsize=10)
    ax.set_yticks(np.arange(len(used)), [f"{f}  ({FEATURE_NAMES_PL.get(f, '')})" for f in used.index])
    ax.set_xlim(0, 1.12)
    _percent_axis(ax)
    ax.set_xlabel("udział w łącznym ważonym przyroście informacji")
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    return fig


def plot_confusion_matrix(cm: pd.DataFrame, title: str, critical: tuple[Any, Any] = ("p", "e")) -> plt.Figure:
    """Counts and row percentages; the critical cell (true p, predicted e) is outlined."""
    values = cm.to_numpy()
    rows = values.sum(axis=1, keepdims=True)
    notes = np.array([[f"{values[i, j]}\n({values[i, j] / rows[i, 0]:.2%} wiersza)" for j in range(values.shape[1])]
                      for i in range(values.shape[0])])
    fig, ax = plt.subplots(figsize=(7.4, 6))
    sns.heatmap(cm, annot=notes, fmt="", cmap=SEQUENTIAL, cbar=False, square=True, linewidths=3,
                linecolor=SURFACE, annot_kws={"size": 13}, ax=ax)
    labels = list(cm.index)
    if critical[0] in labels and critical[1] in cm.columns:
        i, j = labels.index(critical[0]), list(cm.columns).index(critical[1])
        ax.add_patch(Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, fill=False, edgecolor=CLASS_COLORS["p"], lw=3))
        ax.text(j + 0.5, i + 0.86, "błąd krytyczny p → e", ha="center", fontsize=9, color=INK)
    ax.set_xticklabels([CLASS_LABELS.get(c, c) for c in cm.columns])
    ax.set_yticklabels([CLASS_LABELS.get(c, c) for c in cm.index], rotation=0)
    ax.set_xlabel("klasa przewidziana")
    ax.set_ylabel("klasa rzeczywista")
    ax.set_title(title)
    return fig


# ---------------------------------------------------------------------------
# ID3 tree: matplotlib block diagram
# ---------------------------------------------------------------------------

@dataclass
class _Box:
    kind: str                       # "internal" | "leaf" | "group"
    nodes: list[ID3Node]
    edge_label: str | None
    children: list[_Box] = field(default_factory=list)
    breadth: float = 0.0
    level: int = 0

    @property
    def node_ids(self) -> set[int]:
        return {n.node_id for n in self.nodes}

    @property
    def counts(self) -> dict[Any, int]:
        total: dict[Any, int] = {}
        for node in self.nodes:
            for label, count in node.class_counts.items():
                total[label] = total.get(label, 0) + count
        return total


def _subtree_root(model: ID3Classifier, path: Sequence[tuple[str, str]] | None) -> ID3Node:
    node = model.root_
    for feature, value in path or ():
        if node.feature != feature or value not in node.children:
            raise ValueError(f"path step {feature}={value} does not exist in the tree")
        node = node.children[value]
    return node


def _build_boxes(node: ID3Node, fmt: ValueFormatter, group_pure_leaves: bool, classes: Sequence[Any],
                 edge_label: str | None = None, level: int = 0) -> _Box:
    if node.is_leaf:
        return _Box("leaf", [node], edge_label, level=level)
    box = _Box("internal", [node], edge_label, level=level)
    groups: dict[Any, list[tuple[str, ID3Node]]] = {}
    others = []
    for value, child in node.children.items():
        if group_pure_leaves and child.is_leaf and child.entropy == 0.0:
            groups.setdefault(child.majority_class, []).append((value, child))
        else:
            others.append((value, child))

    def group_box(label: Any) -> _Box:
        members = groups[label]
        if len(members) == 1:
            value, child = members[0]
            return _Box("leaf", [child], fmt(node.feature, value), level=level + 1)
        text = ", ".join(fmt(node.feature, value) for value, _ in members)
        return _Box("group", [child for _, child in members], textwrap.fill(text, 26), level=level + 1)

    ordered_classes = [c for c in classes if c in groups]
    if ordered_classes:
        box.children.append(group_box(ordered_classes[0]))
    box.children.extend(_build_boxes(child, fmt, group_pure_leaves, classes, fmt(node.feature, value), level + 1)
                        for value, child in others)
    box.children.extend(group_box(label) for label in ordered_classes[1:])
    return box


def _assign_breadth(box: _Box, counter: list[float]) -> None:
    if not box.children:
        box.breadth = counter[0]
        counter[0] += 1.0
        return
    for child in box.children:
        _assign_breadth(child, counter)
    box.breadth = (box.children[0].breadth + box.children[-1].breadth) / 2.0


def _iter_boxes(box: _Box) -> Iterable[_Box]:
    yield box
    for child in box.children:
        yield from _iter_boxes(child)


def plot_tree(model: ID3Classifier, *, value_formatter: ValueFormatter | None = None,
              group_pure_leaves: bool = False, subtree_path: Sequence[tuple[str, str]] | None = None,
              orientation: str = "TB", highlight_node_ids: Iterable[int] | None = None,
              class_colors: Mapping[Any, str] | None = None, class_labels: Mapping[Any, str] | None = None,
              title: str | None = None) -> plt.Figure:
    """Coloured block diagram of an ID3 tree.

    Every box shows the tested attribute (or the class of a leaf), the number of
    samples, the class distribution (numbers and a proportion strip) and the
    entropy; edges carry the category values. ``group_pure_leaves`` merges
    sibling pure leaves of the same class into one box for a compact
    presentation view (the model itself is unchanged).
    """
    fmt = value_formatter or expand_value
    start = _subtree_root(model, subtree_path)
    classes = list(model.classes_)
    root = _build_boxes(start, fmt, group_pure_leaves, classes)
    _assign_breadth(root, [0.0])
    boxes = list(_iter_boxes(root))
    n_slots = max(b.breadth for b in boxes) + 1
    n_levels = max(b.level for b in boxes) + 1
    highlight = set(highlight_node_ids or ())
    if orientation == "TB":
        slot, step, bw, bh = 2.05, 2.05, 1.78, 1.12
    elif orientation == "LR":
        slot, step, bw, bh = 1.25, 3.55, 2.15, 1.04
    else:
        raise ValueError("orientation must be 'TB' or 'LR'")

    def center(b: _Box) -> tuple[float, float]:
        if orientation == "TB":
            return b.breadth * slot, -b.level * step
        return b.level * step, -b.breadth * slot

    centers = [center(b) for b in boxes]
    xmin, xmax = min(x for x, _ in centers) - bw / 2, max(x for x, _ in centers) + bw / 2
    ymin, ymax = min(y for _, y in centers) - bh / 2, max(y for _, y in centers) + bh / 2
    margin, header = 0.3, (0.95 if title else 0.5)
    fig_w, fig_h = xmax - xmin + 2 * margin, ymax - ymin + 2 * margin + header
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin + header)
    ax.set_axis_off()
    dim = 0.33 if highlight else 1.0

    def on_path(b: _Box) -> bool:
        return bool(b.node_ids & highlight)

    # Edges: elbow connectors with the category value next to the child.
    for b in boxes:
        if not b.children:
            continue
        px, py = center(b)
        for child in b.children:
            cx, cy = center(child)
            emphasised = on_path(b) and on_path(child)
            color, lw, alpha = (NAVY, 2.6, 1.0) if emphasised else (INK_SECONDARY, 1.1, dim)
            if orientation == "TB":
                bus = py - bh / 2 - 0.28
                xs, ys = [px, px, cx, cx], [py - bh / 2, bus, bus, cy + bh / 2]
                label_xy = (cx, (bus + cy + bh / 2) / 2)
            else:
                bus = px + bw / 2 + 0.28
                xs, ys = [px + bw / 2, bus, bus, cx - bw / 2], [py, py, cy, cy]
                label_xy = ((bus + cx - bw / 2) / 2, cy)
            ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, solid_joinstyle="round", zorder=1)
            if child.edge_label:
                ax.text(*label_xy, child.edge_label, ha="center", va="center", fontsize=8.3, color=INK,
                        alpha=1.0 if (emphasised or not highlight) else 0.55, zorder=4, linespacing=1.05,
                        bbox=dict(boxstyle="round,pad=0.22", fc=SURFACE, ec=NAVY if emphasised else GRID, lw=0.8))

    # Nodes.
    for b in boxes:
        cx, cy = center(b)
        counts = b.counts
        n = sum(counts.values())
        node = b.nodes[0]
        if b.kind == "internal":
            face, edge = INTERNAL_FILL, NAVY
            heading = node.feature
            details = [f"n = {n}", " · ".join(f"{c} {counts.get(c, 0)}" for c in classes),
                       f"H = {node.entropy:.3f} · IG = {node.information_gain:.3f}"]
        else:
            label = node.majority_class
            color, name = _class_style(label, class_colors, class_labels)
            face, edge = _tint(color, 0.22), color
            heading = _heading(name) if b.kind == "leaf" else f"{_heading(name)}  ×{len(b.nodes)}"
            details = [f"n = {n}", " · ".join(f"{c} {counts.get(c, 0)}" for c in classes),
                       f"H = {entropy_from_counts(list(counts.values())):.3f}"]
        emphasised = on_path(b)
        alpha = 1.0 if (emphasised or not highlight) else dim
        ax.add_patch(FancyBboxPatch((cx - bw / 2, cy - bh / 2), bw, bh,
                                    boxstyle="round,pad=0,rounding_size=0.09", fc=face,
                                    ec=NAVY if emphasised and highlight else edge,
                                    lw=3.0 if emphasised and highlight else 1.6, alpha=alpha, zorder=2))
        ax.text(cx, cy + bh * 0.30, heading, ha="center", va="center", fontsize=9.6, fontweight="bold",
                color=INK, alpha=alpha, zorder=3)
        ax.text(cx, cy + bh * 0.02, "\n".join(details), ha="center", va="center", fontsize=8.2, color=INK,
                alpha=alpha, zorder=3, linespacing=1.25)
        strip_w, strip_h, strip_y = bw * 0.8, bh * 0.075, cy - bh / 2 + bh * 0.09
        left = cx - strip_w / 2
        for c in classes:
            share = counts.get(c, 0) / n if n else 0.0
            if share > 0:
                color, _ = _class_style(c, class_colors, class_labels)
                ax.add_patch(Rectangle((left, strip_y), strip_w * share, strip_h, fc=color, ec="none",
                                       alpha=alpha, zorder=3))
                left += strip_w * share

    handles = [Patch(facecolor=INTERNAL_FILL, edgecolor=NAVY, label="węzeł: test atrybutu (H, IG)")]
    for c in classes:
        color, name = _class_style(c, class_colors, class_labels)
        handles.append(Patch(facecolor=_tint(color, 0.22), edgecolor=color, label=f"liść: {name}"))
    if group_pure_leaves:
        handles.append(Line2D([], [], color="none", label="×k = k czystych liści tej samej klasy"))
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1 - 0.1 / fig_w, 1 - 0.12 / fig_h),
              ncol=min(len(handles), 4), fontsize=9, frameon=False, handlelength=1.6)
    if title:
        ax.text(xmin, ymax + margin + header - 0.12, title, ha="left", va="top", fontsize=14,
                fontweight="bold", color=NAVY)
    return fig


# ---------------------------------------------------------------------------
# ID3 tree: Graphviz (DOT)
# ---------------------------------------------------------------------------

def tree_to_dot(model: ID3Classifier, *, value_formatter: ValueFormatter | None = None, rankdir: str = "LR",
                class_colors: Mapping[Any, str] | None = None, class_labels: Mapping[Any, str] | None = None,
                title: str | None = None) -> str:
    """DOT source of the full tree (one box per node, one labelled edge per category)."""
    fmt = value_formatter or expand_value
    classes = list(model.classes_)
    lines = [
        "digraph ID3 {",
        f'  graph [rankdir={rankdir}, nodesep=0.22, ranksep=0.5, bgcolor="{SURFACE}", fontname="Helvetica",'
        f' labelloc="t", fontsize=16, fontcolor="{NAVY}"'
        + (f', label=<<b>{escape(title)}</b>>' if title else "") + "];",
        f'  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, penwidth=1.6,'
        f' fontcolor="{INK}"];',
        f'  edge [fontname="Helvetica", fontsize=10, color="{INK_SECONDARY}", fontcolor="{INK}", arrowsize=0.7];',
    ]
    for node in model.iter_nodes():
        dist = " · ".join(f"{c} {node.class_counts.get(c, 0)}" for c in classes)
        if node.is_leaf:
            color, name = _class_style(node.majority_class, class_colors, class_labels)
            label = (f"<<b>{escape(_heading(name))}</b><br/>n = {node.n_samples}<br/>{escape(dist)}"
                     f"<br/>H = {node.entropy:.3f}>")
            fill, border = _tint(color, 0.22), color
        else:
            label = (f"<<b>{escape(node.feature)}</b><br/>n = {node.n_samples}<br/>{escape(dist)}"
                     f"<br/>H = {node.entropy:.3f} · IG = {node.information_gain:.3f}>")
            fill, border = INTERNAL_FILL, NAVY
        lines.append(f'  n{node.node_id} [label={label}, fillcolor="{fill}", color="{border}"];')
        for value, child in node.children.items():
            lines.append(f"  n{node.node_id} -> n{child.node_id} [label=<{escape(fmt(node.feature, value))}>];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def graphviz_available() -> bool:
    return shutil.which("dot") is not None


def render_tree(model: ID3Classifier, stem: str | Path, *, value_formatter: ValueFormatter | None = None,
                formats: Sequence[str] = ("svg", "png"), title: str | None = None,
                class_colors: Mapping[Any, str] | None = None,
                class_labels: Mapping[Any, str] | None = None) -> dict[str, Any]:
    """Write ``stem.dot`` and render the full tree with Graphviz or, without ``dot``, with matplotlib."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    dot_path = stem.with_suffix(".dot")
    dot_path.write_text(tree_to_dot(model, value_formatter=value_formatter, title=title, class_colors=class_colors,
                                    class_labels=class_labels), encoding="utf-8")
    files: list[Path] = []
    if graphviz_available():
        for fmt in formats:
            out = stem.with_suffix(f".{fmt}")
            subprocess.run([shutil.which("dot"), f"-T{fmt}", str(dot_path), "-o", str(out)], check=True,
                           timeout=120, capture_output=True)
            files.append(out)
        return {"backend": "graphviz", "dot": dot_path, "files": files}
    fig = plot_tree(model, value_formatter=value_formatter, orientation="LR", title=title,
                    class_colors=class_colors, class_labels=class_labels)
    files = save_figure(fig, stem, formats=formats, dpi=150)
    plt.close(fig)
    return {"backend": "matplotlib", "dot": dot_path, "files": files}
