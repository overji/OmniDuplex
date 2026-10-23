"""Render publication-style bar charts for the 2026-10-16 benchmark report."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np

MODELS = ['MiniCPM-o 4.5', 'LiveStar', 'ROMA', 'JoyAI']
COLORS = ['#527BA5', '#D39759', '#A8AFB6', '#539B90']
INK = '#263444'
OUT = Path(__file__).resolve().parent / 'assets'
TIMING = np.array([[21.47, 24.32, 22.81], [29.62, 6.09, 10.11],
                   [0., 0., 0.], [29.03, 10.64, 15.57]])
INTERACTION = np.array([[17.24, 30.30, 22.22], [28.97, 22.73, 43.21],
                        [0., 0., 0.], [32.41, 15.91, 38.27]])
ABILITIES = ['Action counting', 'Action perception', 'Attribute perception', 'Causal reasoning',
             'Emotion recognition', 'Event understanding', 'Grounding', 'Object counting',
             'Object perception', 'Prospective reasoning', 'Spatial understanding', 'Text-rich understanding']
WINDOWS = [22, 77, 29, 28, 14, 81, 63, 11, 54, 25, 42, 57]
ABILITY_VALUES = np.array([
    [45.45, 16.88, 20.69, 25., 28.57, 14.81, 20.63, 36.36, 20.37, 16., 28.57, 21.05],
    [13.64, 25.97, 51.72, 17.86, 21.43, 25.93, 15.87, 45.45, 50., 8., 40.48, 36.84],
    [0.] * 12,
    [9.09, 23.38, 37.93, 10.71, 7.14, 23.46, 11.11, 36.36, 48.15, 24., 38.10, 57.89],
])


def setup():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': INK,
        'axes.labelcolor': '#667382', 'xtick.color': '#788492', 'ytick.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.spines.bottom': False,
        'axes.titleweight': 'semibold', 'axes.titlesize': 12,
        'xtick.labelsize': 9, 'ytick.labelsize': 10,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'svg.fonttype': 'none', 'savefig.facecolor': 'white',
    })


def bars(ax, values, title, limit, *, decimals=1, percent=True, labels=True, step=None):
    y = np.arange(4)
    for index, (value, color) in enumerate(zip(values, COLORS)):
        if value is None:
            ax.text(limit * .018, index, 'N/A', va='center', fontsize=10, color='#89939E')
            continue
        ax.barh(index, value, height=.53, color=color, zorder=3)
        if value == 0:
            ax.plot(0, index, 'o', color=color, markersize=4, clip_on=False, zorder=4)
        text = f'{value:.{decimals}f}%' if percent else (f'{value:,.0f}' if decimals == 0 else f'{value:.{decimals}f}')
        ax.text(value + limit * .025, index, text, va='center', fontsize=10, color=INK)
    ax.set_yticks(y, MODELS if labels else [''] * 4)
    ax.set_ylim(3.65, -.65)
    ax.set_xlim(0, limit)
    ax.set_title(title, loc='left', pad=16)
    ax.tick_params(axis='both', length=0, pad=8)
    ax.grid(axis='x', color='#E8ECF0', linewidth=.8, zorder=0)
    if percent:
        ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    if step:
        ax.xaxis.set_major_locator(MultipleLocator(step))


def heading(fig, title, subtitle):
    fig.text(.06, .965, title, ha='left', va='top', fontsize=22, weight='bold', color=INK)
    fig.text(.06, .915, subtitle, ha='left', va='top', fontsize=10, color='#73808E')


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ['svg', 'png']:
        fig.savefig(OUT / f'{name}.{suffix}', dpi=220, bbox_inches='tight', pad_inches=.25)
    plt.close(fig)


def main():
    setup()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.2))
    fig.subplots_adjust(left=.12, right=.97, bottom=.07, top=.80, wspace=.7, hspace=.64)
    heading(fig, 'OmniDuplex · Benchmark results',
            '16 OCT 2026   /   4 models   /   190 evaluated QA   /   503 reference windows')
    for i, title in enumerate(['Window hit rate', 'Time precision', 'Time F1']):
        bars(axes[0, i], TIMING[:, i], title, 37, step=10)
    bars(axes[1, 0], [444, 2446, 0, 1372], 'Generated replies', 2900, percent=False, decimals=0, step=1000)
    bars(axes[1, 1], [.4421, .1762, None, .3904], 'Content score · matched replies', .54, percent=False, decimals=3, step=.2)
    bars(axes[1, 2], [.0949, .0522, 0., .1133], 'Content score · all windows', .145, percent=False, decimals=3, step=.05)
    fig.text(.12, .005, 'Content scores: normalized to 0–1. ROMA has no matched replies; its matched-reply score is N/A.',
             fontsize=9, color='#73808E')
    save(fig, 'benchmark_results_20261016')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.subplots_adjust(left=.12, right=.97, top=.72, bottom=.14, wspace=.65)
    heading(fig, 'Performance by interaction type', 'WINDOW HIT RATE   /   Aggregate Agent results; no positive/negative subtype breakdown')
    for i, (name, count) in enumerate(zip(['RTD', 'PR', 'Agent'], [290, 132, 81])):
        bars(axes[i], INTERACTION[:, i], f'{name}  ·  {count} windows', 55, step=10)
    save(fig, 'benchmark_results_20261016_interactions')

    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    fig.subplots_adjust(left=.12, right=.97, top=.86, bottom=.06, wspace=.65, hspace=.75)
    heading(fig, 'Performance across task abilities', 'WINDOW HIT RATE   /   Shared 0–70% scale   /   Categories with few windows have higher uncertainty')
    for i, ax in enumerate(axes.flat):
        bars(ax, ABILITY_VALUES[:, i], f'{ABILITIES[i]}  ·  n={WINDOWS[i]}', 70, step=20)
    save(fig, 'benchmark_results_20261016_abilities')
    print(f'Generated three PNG/SVG figure pairs in {OUT}')


if __name__ == '__main__':
    main()
