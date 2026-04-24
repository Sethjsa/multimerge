"""
eval_processor.py

Processes lighteval checkpoint results and generates plots using the canonical
task/language mappings in langutils.py.

Outputs: PDF plots under plots/<exp_name>/
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from utils.langutils import (
    TASK_LIST,
    TASK_PER_LANG,
    task_json_key,
    metric_for_task,
    GENERATIVE_TASKS,
)

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_ROOT = Path("results")
EXPERIMENT_DIRS = [
    # "merged-10-checkpoints",
    "mixed-10-checkpoints",
]
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

# Tasks included in the per-language overall average (excludes generative tasks)
AVG_TASKS = [t for t in TASK_LIST if t not in GENERATIVE_TASKS]

# ── IO / loading ──────────────────────────────────────────────────────────────

def checkpoint_num(p: Path) -> int:
    m = re.search(r"(\d+)$", p.name)
    return int(m.group(1)) if m else 0


def find_result_json(cp_path: Path) -> Path | None:
    hits = list(cp_path.rglob("results_*.json"))
    return hits[0] if hits else None


def load_raw_results(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)["results"]


def _save(fig, exp_name: str, filename: str):
    out = OUT_DIR / exp_name / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _line_colors(n: int):
    return cm.tab20(np.linspace(0, 1, max(n, 1)))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_experiment(exp_name: str):
    """
    Returns:
        checkpoints : sorted list[int]
        data        : dict[task -> dict[lang -> dict[cp -> {metric, stderr}]]]
    """
    exp_path = RESULTS_ROOT / exp_name
    search_root = exp_path / "checkpoints" if (exp_path / "checkpoints").exists() else exp_path
    cp_dirs = sorted(
        [d for d in search_root.iterdir() if d.is_dir() and re.search(r"\d+$", d.name)],
        key=checkpoint_num,
    )

    checkpoints: list[int] = []
    data: dict[str, dict[str, dict[int, dict]]] = {t: defaultdict(dict) for t in TASK_LIST}

    for cp_dir in cp_dirs:
        cp_num = checkpoint_num(cp_dir)
        json_path = find_result_json(cp_dir)
        if json_path is None:
            print(f"  [warn] no result JSON in {cp_dir}")
            continue
        checkpoints.append(cp_num)
        raw = load_raw_results(json_path)


        for task in TASK_LIST:
            metric_key, stderr_key = metric_for_task(task)
            for lang in TASK_PER_LANG.get(task, {}):
                key = task_json_key(task, lang)
                if key is None or key not in raw:
                    continue
                vals = raw[key]
                if metric_key not in vals:
                    continue
                data[task][lang][cp_num] = {
                    "metric": vals[metric_key],
                    "stderr": vals.get(stderr_key, 0.0),
                }

    return sorted(set(checkpoints)), data


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _avg_line(lang_data: dict, checkpoints: list) -> tuple[list, list, list]:
    """Compute mean and pooled stderr across all languages at each checkpoint."""
    cp_vals: dict[int, list] = defaultdict(list)
    cp_errs: dict[int, list] = defaultdict(list)
    for cp_data in lang_data.values():
        for cp, v in cp_data.items():
            cp_vals[cp].append(v["metric"])
            cp_errs[cp].append(v["stderr"])
    xs = [x for x in checkpoints if x in cp_vals]
    ys = [np.mean(cp_vals[x]) for x in xs]
    # TODO: do Confidence interval for each language
    conf_intervals = []
    for cp in xs:
        conf_interval = [y - e for y, e in zip(ys, cp_errs[cp])]
        conf_intervals.append(conf_interval)
    return xs, ys, conf_intervals
    # errs = [np.sqrt(np.mean([s**2 for s in cp_errs[x]])) for x in xs]
    # return xs, ys, errs


def _setup_ax(ax, title, xlabel, ylabel, checkpoints, task: str | None = None):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(checkpoints)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.4)
    if task == "flores200":
        ax.set_ylim(0, 50)
    elif task is not None and task not in GENERATIVE_TASKS:
        ax.set_ylim(0, 0.5)


def _add_legend(ax):
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.7,
              bbox_to_anchor=(1.01, 1), borderaxespad=0)


# ── Per-task plots ────────────────────────────────────────────────────────────

def plot_task(task: str, lang_data: dict, checkpoints: list, exp_name: str):
    """
    One plot per task: one line per language + bold AVERAGE line.
    """
    langs = sorted(lang_data)
    colors = _line_colors(len(langs))
    metric_key, _ = metric_for_task(task)

    fig, ax = plt.subplots(figsize=(11, 5))

    for lang, color in zip(langs, colors):
        cp_data = lang_data[lang]
        xs = sorted(cp_data)
        ys = [cp_data[x]["metric"] for x in xs]
        errs = [cp_data[x]["stderr"] for x in xs]
        ax.plot(xs, ys, marker="o", markersize=3,
                label=f"{lang} (μ={np.mean(ys):.3f})", color=color, alpha=0.85)
        ax.fill_between(xs, [y - e for y, e in zip(ys, errs)],
                        [y + e for y, e in zip(ys, errs)], alpha=0.08, color=color)

    # Average across all languages
    axs, ays, aerrs = _avg_line(lang_data, checkpoints)
    ax.plot(axs, ays, marker="D", markersize=5, linewidth=2.5,
            color="black", linestyle="--", label=f"AVG (μ={np.mean(ays):.3f})", zorder=5)
    ax.fill_between(axs, [y - e for y, e in zip(ays, aerrs)],
                    [y + e for y, e in zip(ays, aerrs)], alpha=0.15, color="black")

    _setup_ax(ax, f"{exp_name} | {task} — {metric_key} per language", "Checkpoint", metric_key, checkpoints, task)
    _add_legend(ax)
    fig.tight_layout()
    _save(fig, exp_name, f"task_{task}.pdf")


# ── Per-language plots ────────────────────────────────────────────────────────

def plot_lang_overall(lang: str, by_lang_task: dict, checkpoints: list, exp_name: str):
    """
    Per-language: thin line per task + bold AVERAGE line (non-generative tasks only).
    """
    tasks = sorted(by_lang_task)
    colors = _line_colors(len(tasks))

    fig, ax = plt.subplots(figsize=(11, 5))
    cp_sum: dict[int, list] = defaultdict(list)
    cp_sum_err: dict[int, list] = defaultdict(list)

    for task, color in zip(tasks, colors):
        cp_data = by_lang_task[task]
        xs = sorted(cp_data)
        ys = [cp_data[x]["metric"] for x in xs]
        errs = [cp_data[x]["stderr"] for x in xs]
        ax.plot(xs, ys, marker=".", markersize=3, linewidth=1,
                label=task, color=color, alpha=0.6)
        ax.fill_between(xs, [y - e for y, e in zip(ys, errs)],
                        [y + e for y, e in zip(ys, errs)], alpha=0.06, color=color)
        if task in AVG_TASKS:
            for x, y, e in zip(xs, ys, errs):
                cp_sum[x].append(y)
                cp_sum_err[x].append(e)

    avg_xs = sorted(cp_sum)
    avg_ys = [np.mean(cp_sum[x]) for x in avg_xs]
    avg_errs = [np.sqrt(np.mean([s**2 for s in cp_sum_err[x]])) for x in avg_xs]
    ax.plot(avg_xs, avg_ys, marker="o", markersize=5, linewidth=2.5,
            color="black", label="AVG (non-generative)", zorder=5)
    ax.fill_between(avg_xs, [y - e for y, e in zip(avg_ys, avg_errs)],
                    [y + e for y, e in zip(avg_ys, avg_errs)], alpha=0.15, color="black")

    _setup_ax(ax, f"{exp_name} | lang={lang} — all tasks", "Checkpoint", "metric", checkpoints, None)
    _add_legend(ax)
    fig.tight_layout()
    _save(fig, exp_name, f"lang_{lang}_overall.pdf")


# ── Global overview ───────────────────────────────────────────────────────────

def plot_overall_all_langs(by_lang: dict, checkpoints: list, exp_name: str):
    """
    One line per language: average acc_norm over AVG_TASKS at each checkpoint.
    Includes a grand mean line.
    """
    langs = sorted(by_lang)
    colors = _line_colors(len(langs))

    fig, ax = plt.subplots(figsize=(12, 6))
    grand: dict[int, list] = defaultdict(list)

    for lang, color in zip(langs, colors):
        cp_metrics: dict[int, list] = defaultdict(list)
        cp_stderrs: dict[int, list] = defaultdict(list)
        for task, cp_data in by_lang[lang].items():
            if task not in AVG_TASKS:
                continue
            for cp, vals in cp_data.items():
                cp_metrics[cp].append(vals["metric"])
                cp_stderrs[cp].append(vals["stderr"])

        if not cp_metrics:
            continue
        xs = sorted(cp_metrics)
        ys = [np.mean(cp_metrics[x]) for x in xs]
        errs = [np.sqrt(np.mean([s**2 for s in cp_stderrs[x]])) for x in xs]
        ax.plot(xs, ys, marker="o", markersize=4,
                label=f"{lang} (μ={np.mean(ys):.3f})", color=color)
        ax.fill_between(xs, [y - e for y, e in zip(ys, errs)],
                        [y + e for y, e in zip(ys, errs)], alpha=0.08, color=color)
        for x, y in zip(xs, ys):
            grand[x].append(y)

    gxs = sorted(grand)
    gys = [np.mean(grand[x]) for x in gxs]
    ax.plot(gxs, gys, marker="D", markersize=5, linewidth=2.5,
            color="black", linestyle="--", label="ALL-LANG AVG", zorder=5)

    _setup_ax(ax, f"{exp_name} — overall avg ({', '.join(AVG_TASKS)}) per language",
              "Checkpoint", "mean acc_norm (avg over tasks)", checkpoints, "overall")
    _add_legend(ax)
    fig.tight_layout()
    _save(fig, exp_name, "overall_all_languages.pdf")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for exp_name in EXPERIMENT_DIRS:
        print(f"\n=== Processing: {exp_name} ===")
        checkpoints, data = load_experiment(exp_name)
        print(f"  Checkpoints : {checkpoints}")
        print(f"  Tasks       : {list(data.keys())}")

        # Invert to by_lang[lang][task]
        by_lang: dict[str, dict[str, dict]] = defaultdict(dict)
        for task, lang_data in data.items():
            for lang, cp_data in lang_data.items():
                by_lang[lang][task] = cp_data

        # Per-task plots (one per task, all languages + avg line)
        print("  Task plots …")
        for task, lang_data in data.items():
            if lang_data:
                plot_task(task, lang_data, checkpoints, exp_name)

        # Per-language overall plots
        print("  Language plots …")
        for lang, task_map in by_lang.items():
            if task_map:
                plot_lang_overall(lang, task_map, checkpoints, exp_name)

        # Global overview
        print("  Overall plot …")
        plot_overall_all_langs(by_lang, checkpoints, exp_name)

        print(f"  → {OUT_DIR / exp_name}/")

    print("\nDone.")


if __name__ == "__main__":
    main()