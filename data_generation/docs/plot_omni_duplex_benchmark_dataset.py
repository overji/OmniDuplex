"""Plot statistics for the complete OmniDuplex benchmark QA dataset."""

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#315A86"
ORANGE = "#E38B4F"
TEAL = "#277C73"
PURPLE = "#8068A5"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "svg.fonttype": "none",
})


def seconds(value):
    return sum(float(part) * 60 ** position
               for position, part in enumerate(reversed(str(value).split(":"))))


def video_durations(rows):
    unique = {}
    for row in rows:
        value = float(row["generation_context"]["duration_seconds"])
        previous = unique.setdefault(row["video"], value)
        if previous != value:
            raise ValueError(f"Conflicting duration for {row['video']}")
    return np.array(list(unique.values()))


def first_response(row):
    replies = [reply for reply in row["responses"] if reply.get("role") != "tool"]
    return min(replies, key=lambda reply: seconds(reply["trigger_time"]))


def first_delays(rows):
    return np.array([seconds(first_response(row)["trigger_time"])
                     - seconds(row["question_time"]) for row in rows])


def trigger_modalities(rows):
    names = {
        frozenset(("visual",)): "Visual",
        frozenset(("speech",)): "Speech",
        frozenset(("sound",)): "Sound",
        frozenset(("visual", "speech")): "Visual + speech",
        frozenset(("visual", "sound")): "Visual + sound",
        frozenset(("speech", "sound")): "Speech + sound",
        frozenset(("visual", "speech", "sound")): "Visual + speech + sound",
    }
    counts = Counter()
    for row in rows:
        raw = first_response(row).get("trigger_type")
        counts[names.get(frozenset(raw.split("+")), raw) if raw else "Unlabeled"] += 1
    return counts


def save(fig, target):
    fig.savefig(target.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(target.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def category_bars(counts, title, color, target):
    labels = sorted(counts, key=lambda label: counts[label])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, max(4.2, len(labels) * .49 + 1.3)), layout="constrained")
    ax.barh(y, [counts[label] for label in labels], color=color, height=.72)
    for i, label in enumerate(labels):
        ax.text(counts[label] + .45, i,
                f"{counts[label]} ({counts[label] / sum(counts.values()):.1%})", va="center")
    ax.set(yticks=y, yticklabels=labels, xlabel="QA records", title=title)
    ax.set_xlim(0, max(counts.values()) * 1.24 + 1)
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    save(fig, target)


def donut(counts, labels, title, center, target):
    values = [counts[label] for label in labels]
    colors = [BLUE, ORANGE, TEAL, PURPLE][:len(labels)]
    fig, ax = plt.subplots(figsize=(7.2, 5), layout="constrained")
    wedges, _, autotexts = ax.pie(
        values, startangle=90, counterclock=False, colors=colors,
        autopct=lambda pct: f"{pct:.1f}%", pctdistance=.77,
        wedgeprops={"width": .44, "edgecolor": "white", "linewidth": 2},
    )
    for item in autotexts:
        item.set_color("white")
        item.set_fontweight("bold")
    ax.text(0, 0, center, ha="center", va="center", fontsize=17, fontweight="bold")
    ax.legend(wedges, [f"{label}: {counts[label]}" for label in labels],
              loc="center left", bbox_to_anchor=(.92, .5), frameon=False)
    ax.set_title(title)
    save(fig, target)


def binned_bars(values, bins, labels, title, xlabel, annotation, color, target):
    counts, _ = np.histogram(values, bins=bins)
    fig, ax = plt.subplots(figsize=(9.5, 4.6), layout="constrained")
    ax.bar(np.arange(len(counts)), counts, color=color, width=.78)
    for i, count in enumerate(counts):
        ax.text(i, count + .25, str(count), ha="center", va="bottom")
    ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel="Count",
           xlabel=xlabel, title=title)
    ax.set_ylim(0, max(counts) * 1.2 + 1)
    ax.text(.98, .96, annotation, transform=ax.transAxes, ha="right", va="top",
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": .95})
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    save(fig, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "assets")
    args = parser.parse_args()
    rows = json.loads(args.data.read_text())
    if len(rows) != 194:
        raise ValueError("This dated report expects the complete 194-record dataset")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    category_bars(Counter(row["ability"] for row in rows), "Task ability distribution",
                  BLUE, args.output_dir / "task_abilities")
    donut(Counter(row["mode"] for row in rows), ["rtd", "pr", "agent"],
          "Interaction mode distribution", "194 QA", args.output_dir / "interaction_modes")
    agent = [row for row in rows if row["mode"] == "agent"]
    donut(Counter(row.get("sample_type", row["generation_context"].get("sample_type"))
                  for row in agent), ["positive", "missing_tool", "no_context"],
          "Agent sample types", "78 Agent", args.output_dir / "agent_subtypes")

    durations = video_durations(rows)
    binned_bars(durations, [0, 60, 120, 180, 240, 300, 420, 540, 660],
                ["0–1", "1–2", "2–3", "3–4", "4–5", "5–7", "7–9", "9–11"],
                "Video duration distribution (51 unique videos)", "Duration (minutes)",
                f"Mean {np.mean(durations):.1f}s  |  Median {np.median(durations):.1f}s",
                ORANGE, args.output_dir / "video_duration")

    delays = first_delays(rows)
    p90 = np.sort(delays)[int(np.ceil(.9 * len(delays))) - 1]
    binned_bars(delays, [0, 1, 3, 10, 30, 60, 120, 180, 240, 360],
                ["0–1", "1–3", "3–10", "10–30", "30–60", "60–120",
                 "120–180", "180–240", "240–360"],
                "Question to first assistant response (194 QA)", "Delay (seconds)",
                f"Mean {np.mean(delays):.1f}s  |  Median {np.median(delays):.1f}s  |  P90 {p90:.0f}s",
                TEAL, args.output_dir / "first_response_delay")
    category_bars(trigger_modalities(rows), "First-response trigger modality",
                  PURPLE, args.output_dir / "trigger_modality")
    print(f"Created six SVG and six PNG charts in {args.output_dir}")


if __name__ == "__main__":
    main()
