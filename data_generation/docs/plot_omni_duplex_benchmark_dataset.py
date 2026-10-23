"""Plot statistics for the complete OmniDuplex benchmark QA dataset."""

import argparse
from collections import Counter
import json
import random
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#315A86"
ORANGE = "#E38B4F"
TEAL = "#277C73"
PURPLE = "#8068A5"
GREEN = "#72AD91"
PINK = "#DFA7A7"
LIGHT_BLUE = "#AFC9D8"
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
        for response in row.get("responses", []):
            if response.get("role") == "tool":
                continue
            raw = response.get("trigger_type")
            counts[names.get(frozenset(raw.split("+")), raw) if raw else "Unlabeled"] += 1
    return counts


def event_words(rows):
    stop = set("a an the and or but of to in on at for from with by as is are was were be been being this that these those it its into after before while when where which who what how why can could would should may might must will has have had do does did their they them he she his her we you your our i not no all any more most other some such than then there here over under across during about based use used using visible heard seen shows shown appears begins continue camera scene image frame video trigger evidence one two first second new".split())
    frequencies = Counter()
    for row in rows:
        for response in row.get("responses", []):
            if response.get("role") == "tool":
                continue
            facts = [item.get("fact", "") for item in response.get("evidence", []) if isinstance(item, dict)]
            text = " ".join(facts) or response.get("trigger_type_reason", "")
            frequencies.update(word.lower() for word in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text)
                               if word.lower() not in stop)
    return frequencies


def response_trigger_times(rows):
    first, last = [], []
    for row in rows:
        times = [seconds(item["trigger_time"]) for item in row.get("responses", [])
                 if item.get("role") != "tool" and item.get("trigger_time") is not None]
        if times:
            first.append(min(times))
            last.append(max(times))
    return np.asarray(first), np.asarray(last)


def all_trigger_times(rows):
    return np.asarray([seconds(item["trigger_time"])
                       for row in rows for item in row.get("responses", [])
                       if item.get("role") != "tool" and item.get("trigger_time") is not None])


def video_genres(rows, metadata_path):
    """Use explicit source taxonomy fields and preserve missing genre labels."""
    if not metadata_path or not metadata_path.exists():
        return Counter({"Genre metadata unavailable": len({row.get("video") for row in rows})})
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = payload.get("videos", []) if isinstance(payload, dict) else payload
    by_file = {Path(item.get("file", "")).name: item for item in records if isinstance(item, dict)}
    unique = {Path(str(row.get("video", "unknown"))).name for row in rows}
    counts = Counter()
    for filename in unique:
        item = by_file.get(filename)
        genre = None
        if item and item.get("dataset") == "WorldSense":
            genre = item.get("domain")
            prefix = "WorldSense domain"
        elif item and item.get("dataset") == "OmniVideoBench":
            genre = item.get("video_type")
            prefix = "OmniVideoBench type"
        else:
            prefix = ""
        label = f"{prefix}: {genre}" if genre else "Unlabeled"
        counts[label] += 1
    return counts


def six_panel_overview(rows, metadata_path, target):
    """Plot trigger/duration/genre/task/interaction/QA-modality distributions."""
    fig, axes = plt.subplots(3, 2, figsize=(17, 15), layout="constrained")

    trigger_times = all_trigger_times(rows)
    ax = axes[0, 0]
    edges = np.arange(0, max(700, np.ceil(max(trigger_times, default=0) / 30) * 30 + 30), 30)
    ax.hist(trigger_times, bins=edges, color="#6EA5D8", edgecolor="white", linewidth=.45)
    ax.axvline(np.median(trigger_times), color=ORANGE, linestyle="--", linewidth=1.7,
               label=f"Median {np.median(trigger_times):.1f}s")
    ax.set(title=f"(a) Trigger time distribution ({len(trigger_times):,} response events)",
           xlabel="Trigger time from video start (seconds)", ylabel="Response events")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)

    durations = video_durations(rows)
    ax = axes[0, 1]
    duration_edges = [0, 30, 60, 90, 120, 180, 240, 300, 420, 540, 660, np.inf]
    duration_labels = ["0–30", "30–60", "60–90", "90–120", "120–180", "180–240",
                       "240–300", "300–420", "420–540", "540–660", ">660"]
    duration_counts, _ = np.histogram(durations, bins=duration_edges)
    ax.bar(np.arange(len(duration_counts)), duration_counts, color=ORANGE, width=.78)
    ax.set(title=f"(b) Video duration ({len(durations):,} unique videos)",
           xlabel="Duration (seconds)", ylabel="Videos", xticks=np.arange(len(duration_labels)),
           xticklabels=duration_labels)
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)

    ax = axes[1, 0]
    genre_counts = video_genres(rows, metadata_path)
    genre_items = genre_counts.most_common()
    labels = [item[0] for item in genre_items]
    values = [item[1] for item in genre_items]
    y = np.arange(len(labels))
    ax.barh(y, values, color=TEAL, height=.72)
    for index, value in enumerate(values):
        ax.text(value + max(values) * .012, index, f"{value} ({value / sum(values):.1%})", va="center", fontsize=8)
    ax.set(yticks=y, yticklabels=labels, xlabel="Unique videos", title="(c) Video genre/domain (metadata)")
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.3)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)

    ax = axes[1, 1]
    ability_counts = Counter(row.get("ability", "unknown") for row in rows)
    abilities = ability_counts.most_common()
    labels = [item[0].replace("_", "-") for item in abilities]
    values = [item[1] for item in abilities]
    y = np.arange(len(labels))
    ax.barh(y, values, color=PURPLE, height=.72)
    for index, value in enumerate(values):
        ax.text(value + max(values) * .012, index, f"{value} ({value / len(rows):.1%})", va="center", fontsize=8)
    ax.set(yticks=y, yticklabels=labels, xlabel="QA records", title="(d) Task category / ability")
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.3)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)

    def pie_panel(ax, counts, title, palette):
        items = counts.most_common()
        labels = [name for name, _ in items]
        values = [value for _, value in items]
        wedges, _, texts = ax.pie(values, startangle=90, counterclock=False,
                                  colors=palette[:len(values)],
                                  autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
                                  pctdistance=.74,
                                  wedgeprops={"edgecolor": "white", "linewidth": 1.4})
        for text in texts:
            text.set_fontsize(8)
            text.set_fontweight("bold")
        ax.legend(wedges, [f"{name} ({value})" for name, value in items],
                  loc="center left", bbox_to_anchor=(.95, .5), fontsize=8, frameon=False)
        ax.set_title(title)

    pie_panel(axes[2, 0], Counter(row.get("mode", "unknown") for row in rows),
              "(e) Interaction type", ["#6FAAD0", "#9587BF", "#79B39A", "#E27B52"])
    # Normalize modality ordering so synonymous token order does not split a slice.
    normalized = Counter()
    canonical = {
        frozenset(("visual",)): "Visual", frozenset(("speech",)): "Speech", frozenset(("sound",)): "Sound",
        frozenset(("visual", "speech")): "Visual + speech", frozenset(("visual", "sound")): "Visual + sound",
        frozenset(("speech", "sound")): "Speech + sound",
        frozenset(("visual", "speech", "sound")): "Visual + speech + sound",
    }
    for row in rows:
        raw = first_response(row).get("trigger_type")
        normalized[canonical.get(frozenset(raw.split("+")), "Unlabeled") if raw else "Unlabeled"] += 1
    pie_panel(axes[2, 1], normalized, "(f) QA modality (first response trigger_type)",
              ["#6FAAD0", "#9587BF", "#E27B52", "#79B39A", "#DFA7A7", "#7D9EB8", LIGHT_BLUE, "#B0B0B0"])
    fig.suptitle(f"Merged benchmark dataset statistics | {len(rows):,} QA | {len(durations):,} videos",
                 fontsize=17, fontweight="bold")
    save(fig, target)


def compact_wordcloud(ax, frequencies):
    """Draw a deterministic word-cloud-like panel without an extra dependency."""
    rng = random.Random(23)
    palette = [BLUE, ORANGE, TEAL, PURPLE, "#38A6B4", "#8A9A45", "#C45D63"]
    placed = []
    for word, count in frequencies.most_common(70):
        size = min(30, max(8, 8 + 4.4 * np.sqrt(count)))
        width = min(.8, len(word) * size * .00095)
        height = size * .0027
        candidates = [(rng.uniform(width / 2, 1 - width / 2),
                       rng.uniform(height / 2, 1 - height / 2)) for _ in range(250)]
        position = next((point for point in candidates if all(
            abs(point[0] - x) > (width + w) * .55 or abs(point[1] - y) > (height + h) * .6
            for x, y, w, h in placed)), None)
        if position is None:
            continue
        x, y = position
        placed.append((x, y, width, height))
        ax.text(x, y, word, ha="center", va="center", fontsize=size,
                color=palette[len(placed) % len(palette)], transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def benchmark_overview(rows, target):
    """Create the four-panel dataset overview inspired by OmniPro Figure 2."""
    fig = plt.figure(figsize=(16, 11), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1))

    # Audio dependency is a per-QA ability label, summarized as row percentages.
    ax = fig.add_subplot(grid[0, 0])
    dependencies = ("required", "helpful", "none")
    colors = ("#67A9CF", "#FDAE61", "#C9C9C9")
    ability_counts = {}
    for row in rows:
        ability_counts.setdefault(row.get("ability", "unknown"), Counter())[row.get("audio_dependency", "unknown")] += 1
    abilities = sorted(ability_counts, key=lambda key: sum(ability_counts[key].values()))
    y = np.arange(len(abilities))
    left = np.zeros(len(abilities))
    for dependency, color in zip(dependencies, colors):
        values = np.array([100 * ability_counts[name][dependency] / sum(ability_counts[name].values()) for name in abilities])
        ax.barh(y, values, left=left, color=color, label=dependency.capitalize(), height=.72)
        for index, value in enumerate(values):
            if value >= 9:
                ax.text(left[index] + value / 2, index, f"{value:.1f}", ha="center", va="center",
                        fontsize=8, fontweight="bold")
        left += values
    ax.set(yticks=y, yticklabels=[name.replace("_", "-").title() for name in abilities],
           xlim=(0, 100), xlabel="Percentage (%)", title="(a) Audio dependency per ability")
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.0), ncol=3, fontsize=8, frameon=True)
    ax.grid(axis="x", linestyle="--", alpha=.25)
    ax.set_axisbelow(True)

    # Count all non-tool response events, not just each QA's first response.
    ax = fig.add_subplot(grid[0, 1])
    modality_counts = trigger_modalities(rows)
    modality_order = ["Visual", "Speech", "Sound", "Visual + speech", "Visual + sound",
                      "Speech + sound", "Visual + speech + sound", "Unlabeled"]
    labels = [label for label in modality_order if modality_counts[label]]
    pie_colors = ["#6FAAD0", "#9587BF", "#E27B52", "#79B39A", "#DFA7A7", "#7D9EB8", LIGHT_BLUE, "#B0B0B0"]
    wedges, _, autotexts = ax.pie([modality_counts[label] for label in labels], startangle=90,
                                  counterclock=False, colors=pie_colors[:len(labels)],
                                  autopct=lambda pct: f"{pct:.1f}%" if pct >= 2.5 else "", pctdistance=.78,
                                  wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    for item in autotexts:
        item.set_fontsize(8)
        item.set_fontweight("bold")
    ax.legend(wedges, [f"{label} ({modality_counts[label]})" for label in labels],
              loc="lower center", bbox_to_anchor=(.5, -.20), ncol=2, fontsize=8)
    ax.set_title("(b) Trigger modality ratio")

    ax = fig.add_subplot(grid[1, 1])
    compact_wordcloud(ax, event_words(rows))
    ax.set_title("(c) Trigger event evidence word cloud", pad=8)

    ax = fig.add_subplot(grid[1, 0])
    first, last = response_trigger_times(rows)
    bins = np.arange(0, max(520, np.ceil(max(last, default=0) / 20) * 20 + 20) + 1, 20)
    weights_first = np.ones(len(first)) * (100 / len(first)) if len(first) else None
    weights_last = np.ones(len(last)) * (100 / len(last)) if len(last) else None
    ax.hist(first, bins=bins, weights=weights_first, color="#6EA5D8", alpha=.72, label="First trigger")
    ax.hist(last, bins=bins, weights=weights_last, color="#E99A9D", alpha=.62, label="Last trigger")
    ax.axvline(np.mean(first), color="#287CB6", linestyle="--", linewidth=1.6)
    ax.axvline(np.mean(last), color="#D9474C", linestyle="--", linewidth=1.6)
    ax.text(.06, .94, f"First mean {np.mean(first):.1f}s", color="#287CB6", weight="bold",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.text(.36, .94, f"Last mean {np.mean(last):.1f}s", color="#D9474C", weight="bold",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.set(xlabel="Trigger time from video start (seconds)", ylabel="Percentage of QA (%)",
           title="(d) First vs. last trigger time distribution")
    ax.set_xlim(0, bins[-1])
    ax.grid(axis="y", linestyle="--", alpha=.28)
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Dataset statistics: {len(rows):,} QA records | {len(ability_counts)} abilities", fontsize=16, weight="bold")
    save(fig, target)


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
    parser.add_argument("--data", type=Path, nargs="+", required=True,
                        help="one or more QA JSON files to aggregate")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "assets")
    parser.add_argument("--prefix", default="",
                        help="prefix for generated chart filenames")
    parser.add_argument("--video-metadata", type=Path,
                        default=Path(__file__).resolve().parents[2] / "videos/video_metadata.json",
                        help="optional metadata JSON with video genre/domain fields")
    args = parser.parse_args()
    rows = [row for path in args.data for row in json.loads(path.read_text())]
    if not rows:
        raise ValueError("No QA rows found in input files")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def target(name):
        return args.output_dir / f"{args.prefix}{name}"

    category_bars(Counter(row["ability"] for row in rows), "Task ability distribution",
                  BLUE, target("task_abilities"))
    donut(Counter(row["mode"] for row in rows), ["rtd", "pr", "agent"],
          "Interaction mode distribution", f"{len(rows)} QA", target("interaction_modes"))
    agent = [row for row in rows if row["mode"] == "agent"]
    agent_counts = Counter(row.get("sample_type", row["generation_context"].get("sample_type", "positive"))
                           for row in agent)
    agent_labels = [label for label in ("positive", "missing_tool", "no_context", "negative")
                    if agent_counts[label]]
    donut(agent_counts, agent_labels,
          "Agent sample types", f"{len(agent)} Agent", target("agent_subtypes"))

    durations = video_durations(rows)
    binned_bars(durations, [0, 60, 120, 180, 240, 300, 420, 540, 660],
                ["0–1", "1–2", "2–3", "3–4", "4–5", "5–7", "7–9", "9–11"],
                f"Video duration distribution ({len(durations)} unique videos)", "Duration (minutes)",
                f"Mean {np.mean(durations):.1f}s  |  Median {np.median(durations):.1f}s",
                ORANGE, target("video_duration"))

    delays = first_delays(rows)
    p90 = np.sort(delays)[int(np.ceil(.9 * len(delays))) - 1]
    binned_bars(delays, [0, 1, 3, 10, 30, 60, 120, 180, 240, 360, np.inf],
                ["0–1", "1–3", "3–10", "10–30", "30–60", "60–120",
                 "120–180", "180–240", "240–360", ">360"],
                f"Question to first assistant response ({len(rows)} QA)", "Delay (seconds)",
                f"Mean {np.mean(delays):.1f}s  |  Median {np.median(delays):.1f}s  |  P90 {p90:.0f}s",
                TEAL, target("first_response_delay"))
    category_bars(trigger_modalities(rows), "Response-event trigger modality",
                  PURPLE, target("trigger_modality"))
    benchmark_overview(rows, target("omnipro_style_overview"))
    six_panel_overview(rows, args.video_metadata, target("six_panel_overview"))
    print(f"Created eight SVG and eight PNG charts in {args.output_dir}")


if __name__ == "__main__":
    main()
