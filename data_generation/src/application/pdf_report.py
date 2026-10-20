"""Create a compact PDF summary for a merged pipeline QA bundle."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.platypus import (
        Flowable,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
except ImportError as exc:  # pragma: no cover - exercised in deployment environments
    raise RuntimeError("PDF reports require reportlab; install data_generation/requirements.txt") from exc

from ..domain.qa import is_agent_negative


NAVY = colors.HexColor("#315A86")
TEAL = colors.HexColor("#277C73")
ORANGE = colors.HexColor("#E38B4F")
PURPLE = colors.HexColor("#8068A5")
PALETTE = (NAVY, ORANGE, TEAL, PURPLE, colors.HexColor("#8E6C56"), colors.HexColor("#6E8294"))
PAGE_WIDTH, _ = A4
FONT = "Helvetica"


def _seconds(value: Any) -> float:
    parts = str(value).split(":")
    return sum(float(part) * 60**index for index, part in enumerate(reversed(parts)))


def _first_response(sample: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [item for item in sample.get("responses", [])
                  if isinstance(item, dict) and item.get("role") != "tool"
                  and item.get("trigger_time") is not None]
    return min(candidates, key=lambda item: _seconds(item["trigger_time"])) if candidates else None


def _normalize_modality(value: Any) -> str:
    if not value:
        return "unlabeled"
    modalities = frozenset(str(value).split("+"))
    labels = {
        frozenset({"visual"}): "visual only",
        frozenset({"speech"}): "speech only",
        frozenset({"sound"}): "sound only",
        frozenset({"visual", "speech"}): "visual + speech",
        frozenset({"visual", "sound"}): "visual + sound",
        frozenset({"speech", "sound"}): "speech + sound",
        frozenset({"visual", "speech", "sound"}): "visual + speech + sound",
    }
    return labels.get(modalities, "+".join(sorted(modalities)))


class DonutChart(Flowable):
    def __init__(self, title: str, counts: Counter[str], center: str):
        super().__init__()
        self.title = title
        self.items = [(key, value) for key, value in counts.most_common() if value]
        self.center = center
        self.width = PAGE_WIDTH - 34 * mm
        self.height = 66 * mm

    def draw(self) -> None:
        canvas = self.canv
        canvas.setFillColor(colors.HexColor("#26384A"))
        canvas.setFont(FONT, 12)
        canvas.drawString(0, self.height - 5 * mm, self.title)
        total = sum(value for _, value in self.items)
        if total <= 0:
            canvas.setFillColor(colors.grey)
            canvas.setFont(FONT, 10)
            canvas.drawString(4 * mm, 24 * mm, "No data available")
            return
        cx, cy, radius = 30 * mm, 27 * mm, 20 * mm
        start = 90
        for index, (_, value) in enumerate(self.items):
            extent = 360 * value / total
            canvas.setFillColor(PALETTE[index % len(PALETTE)])
            canvas.wedge(cx - radius, cy - radius, cx + radius, cy + radius,
                         start, -extent, stroke=0, fill=1)
            start -= extent
        canvas.setFillColor(colors.white)
        canvas.circle(cx, cy, radius * .57, stroke=0, fill=1)
        canvas.setFillColor(colors.HexColor("#26384A"))
        canvas.setFont(FONT, 10)
        canvas.drawCentredString(cx, cy - 1.5 * mm, self.center)
        legend_x = 63 * mm
        legend_y = self.height - 15 * mm
        canvas.setFont(FONT, 8.5)
        for index, (label, value) in enumerate(self.items):
            y = legend_y - index * 7 * mm
            canvas.setFillColor(PALETTE[index % len(PALETTE)])
            canvas.rect(legend_x, y - 1.3 * mm, 3 * mm, 3 * mm, stroke=0, fill=1)
            canvas.setFillColor(colors.HexColor("#34495E"))
            canvas.drawString(legend_x + 5 * mm, y, f"{label}: {value} ({value / total:.1%})")


class HorizontalBars(Flowable):
    def __init__(self, title: str, counts: Counter[str], *, height_per_row: float = 6 * mm):
        super().__init__()
        self.title = title
        self.items = counts.most_common()
        self.width = PAGE_WIDTH - 34 * mm
        self.height = max(20 * mm, 13 * mm + height_per_row * len(self.items))
        self.height_per_row = height_per_row

    def draw(self) -> None:
        canvas = self.canv
        canvas.setFillColor(colors.HexColor("#26384A"))
        canvas.setFont(FONT, 11)
        canvas.drawString(0, self.height - 5 * mm, self.title)
        if not self.items:
            return
        left, right = 56 * mm, self.width - 14 * mm
        max_value = max(value for _, value in self.items) or 1
        row_top = self.height - 11 * mm
        bar_h = min(3.6 * mm, self.height_per_row * .58)
        canvas.setFont(FONT, 7.2)
        for index, (label, value) in enumerate(self.items):
            y = row_top - index * self.height_per_row
            canvas.setFillColor(colors.HexColor("#4A5968"))
            canvas.drawRightString(left - 3 * mm, y + .3 * mm, str(label))
            canvas.setFillColor(colors.HexColor("#E8EDF2"))
            canvas.roundRect(left, y - 1 * mm, right - left, bar_h, 1.5 * mm, stroke=0, fill=1)
            canvas.setFillColor(TEAL if "modality" in self.title.lower() else NAVY)
            canvas.roundRect(left, y - 1 * mm, (right - left) * value / max_value,
                             bar_h, 1.5 * mm, stroke=0, fill=1)
            canvas.setFillColor(colors.HexColor("#34495E"))
            canvas.drawString(right + 2 * mm, y + .3 * mm, str(value))


class Histogram(Flowable):
    def __init__(self, title: str, values: list[float], edges: list[float], labels: list[str], color):
        super().__init__()
        self.title, self.values, self.edges, self.labels, self.color = title, values, edges, labels, color
        self.width = PAGE_WIDTH - 34 * mm
        self.height = 57 * mm

    def draw(self) -> None:
        canvas = self.canv
        canvas.setFillColor(colors.HexColor("#26384A"))
        canvas.setFont(FONT, 10)
        canvas.drawString(0, self.height - 4 * mm, self.title)
        if not self.values:
            canvas.setFillColor(colors.grey)
            canvas.drawString(0, 22 * mm, "No data available")
            return
        counts = []
        for left, right in zip(self.edges, self.edges[1:]):
            counts.append(sum(left <= value < right for value in self.values))
        max_count = max(counts) or 1
        chart_x, chart_y, chart_w, chart_h = 8 * mm, 12 * mm, self.width - 16 * mm, 32 * mm
        gap = 1.4 * mm
        bar_w = (chart_w - gap * (len(counts) - 1)) / len(counts)
        canvas.setFont(FONT, 6.5)
        for index, (count, label) in enumerate(zip(counts, self.labels)):
            x = chart_x + index * (bar_w + gap)
            height = chart_h * count / max_count
            canvas.setFillColor(self.color)
            canvas.rect(x, chart_y, bar_w, height, stroke=0, fill=1)
            canvas.setFillColor(colors.HexColor("#34495E"))
            canvas.drawCentredString(x + bar_w / 2, chart_y + height + 1.2 * mm, str(count))
            canvas.drawCentredString(x + bar_w / 2, chart_y - 4 * mm, label)


def _styles():
    global FONT
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("OmniTitle", parent=styles["Title"], fontName=FONT,
                                fontSize=22, leading=29, textColor=colors.HexColor("#20364A"),
                                alignment=TA_LEFT, spaceAfter=4 * mm),
        "section": ParagraphStyle("OmniSection", parent=styles["Heading2"], fontName=FONT,
                                   fontSize=14, leading=19, textColor=NAVY, spaceBefore=3 * mm,
                                   spaceAfter=2 * mm),
        "body": ParagraphStyle("OmniBody", parent=styles["BodyText"], fontName=FONT,
                                fontSize=9, leading=14, textColor=colors.HexColor("#334155"),
                                spaceAfter=2 * mm),
        "small": ParagraphStyle("OmniSmall", parent=styles["BodyText"], fontName=FONT,
                                 fontSize=7.5, leading=11, textColor=colors.HexColor("#64748B")),
        "cell": ParagraphStyle("OmniCell", parent=styles["BodyText"], fontName=FONT,
                                fontSize=8, leading=11, textColor=colors.HexColor("#334155")),
        "header": ParagraphStyle("OmniHeader", parent=styles["BodyText"], fontName=FONT,
                                  fontSize=8, leading=11, textColor=colors.white),
        "center": ParagraphStyle("OmniCenter", parent=styles["BodyText"], fontName=FONT,
                                  fontSize=8, leading=11, alignment=TA_CENTER,
                                  textColor=colors.HexColor("#334155")),
    }


def _table(rows, widths, *, header=True):
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#D9E1E8")),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1),
         [colors.white, colors.HexColor("#F4F7FA")]),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ])
    table.setStyle(TableStyle(commands))
    return table


def _footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E1E8"))
    canvas.line(17 * mm, 14 * mm, PAGE_WIDTH - 17 * mm, 14 * mm)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.setFont(FONT, 7)
    canvas.drawString(17 * mm, 9 * mm, "OmniDuplex Dataset Report")
    canvas.drawRightString(PAGE_WIDTH - 17 * mm, 9 * mm, f"{document.page}")
    canvas.restoreState()


def write_pdf_report(data_path: Path, output_path: Path) -> dict[str, Any]:
    """Write a PDF report, automatically excluding historical Agent negatives."""
    data_path, output_path = Path(data_path), Path(output_path)
    raw_rows = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list):
        raise ValueError(f"QA bundle must be a JSON list: {data_path}")
    rows = [row for row in raw_rows if isinstance(row, dict) and not is_agent_negative(row)]
    modes = Counter(row.get("mode", "unknown") for row in rows)
    abilities = Counter(row.get("ability", "unknown") for row in rows)
    agent_types = Counter(
        row.get("sample_type", (row.get("generation_context") or {}).get("sample_type", "positive"))
        for row in rows if row.get("mode") == "agent"
    )
    modality_counts: Counter[str] = Counter()
    delays: list[float] = []
    mode_delays: dict[str, list[float]] = defaultdict(list)
    durations: dict[str, float] = {}
    for row in rows:
        video = str(row.get("video", "unknown"))
        context = row.get("generation_context") or {}
        if context.get("duration_seconds") is not None:
            duration = float(context["duration_seconds"])
            previous = durations.setdefault(video, duration)
            if not math.isclose(previous, duration, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"conflicting video duration for {video}")
        response = _first_response(row)
        if response is None:
            continue
        modality_counts[_normalize_modality(response.get("trigger_type"))] += 1
        if row.get("question_time") is not None:
            delay = _seconds(response["trigger_time"]) - _seconds(row["question_time"])
            delays.append(delay)
            mode_delays[str(row.get("mode", "unknown"))].append(delay)

    styles = _styles()
    sample_count = len(rows)
    mode_rows = [[Paragraph("Interaction type", styles["header"]), Paragraph("Count", styles["header"]),
                  Paragraph("Share", styles["header"])]]
    for name in ("agent", "rtd", "pr"):
        count = modes.get(name, 0)
        mode_rows.append([Paragraph(name.upper() if name != "agent" else "Agent", styles["cell"]),
                          Paragraph(str(count), styles["center"]),
                          Paragraph(f"{count / sample_count:.1%}" if sample_count else "-", styles["center"])])
    for name, count in modes.items():
        if name not in {"agent", "rtd", "pr"}:
            mode_rows.append([Paragraph(name, styles["cell"]), Paragraph(str(count), styles["center"]),
                              Paragraph(f"{count / sample_count:.1%}" if sample_count else "-", styles["center"])])

    duration_values = list(durations.values())
    duration_stats = None
    if duration_values:
        duration_stats = (statistics_mean(duration_values), sorted(duration_values)[(len(duration_values) - 1) // 2]
                          if len(duration_values) % 2 else sum(sorted(duration_values)[len(duration_values)//2-1:len(duration_values)//2+1]) / 2,
                          min(duration_values), max(duration_values))
    delay_sorted = sorted(delays)
    delay_stats = None
    if delay_sorted:
        midpoint = len(delay_sorted) // 2
        median_delay = delay_sorted[midpoint] if len(delay_sorted) % 2 else (delay_sorted[midpoint - 1] + delay_sorted[midpoint]) / 2
        delay_stats = (statistics_mean(delay_sorted), median_delay,
                       delay_sorted[math.ceil(.9 * len(delay_sorted)) - 1],
                       min(delay_sorted), max(delay_sorted))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending = output_path.with_name(output_path.name + ".pending")
    doc = SimpleDocTemplate(str(pending), pagesize=A4, rightMargin=17 * mm, leftMargin=17 * mm,
                            topMargin=16 * mm, bottomMargin=20 * mm,
                            title="OmniDuplex Data Generation Report", author="OmniDuplex")
    story: list[Flowable] = [
        Paragraph("OmniDuplex Data Generation Report", styles["title"]),
        Paragraph(f"Source: {data_path.name} - Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", styles["small"]),
        Spacer(1, 5 * mm),
        _table([
            [Paragraph("Included QA", styles["header"]), Paragraph("Unique videos", styles["header"]), Paragraph("Agent negatives", styles["header"])],
            [Paragraph(str(sample_count), styles["center"]), Paragraph(str(len(durations)), styles["center"]), Paragraph("0 (filtered on merge)", styles["center"])],
        ], [doc.width / 3] * 3),
        Spacer(1, 5 * mm),
        DonutChart("Interaction type distribution", modes, f"{sample_count} QA"),
        _table(mode_rows, [doc.width * .50, doc.width * .24, doc.width * .26]),
        Spacer(1, 4 * mm),
        HorizontalBars("Task ability distribution", abilities, height_per_row=5.5 * mm),
        PageBreak(),
        Paragraph("Trigger modalities", styles["section"]),
        Paragraph("trigger_type records the audio or visual evidence that supports a response. Use a combined label when multiple modalities jointly support the same trigger.", styles["body"]),
        DonutChart("First non-tool response modality", modality_counts, f"{sum(modality_counts.values())} responses"),
        Paragraph("Modality definitions", styles["section"]),
        _table([
            [Paragraph("Label", styles["header"]), Paragraph("Evidence", styles["header"])],
            [Paragraph("visual", styles["cell"]), Paragraph("Objects, actions, states, or visible text in video frames support the trigger.", styles["cell"])],
            [Paragraph("speech", styles["cell"]), Paragraph("Spoken language content supports the trigger.", styles["cell"])],
            [Paragraph("sound", styles["cell"]), Paragraph("Non-speech audio, such as music, ambient sound, or action sounds, supports the trigger.", styles["cell"])],
            [Paragraph("combined", styles["cell"]), Paragraph("Multiple modalities jointly support one trigger; order does not indicate importance.", styles["cell"])],
        ], [doc.width * .22, doc.width * .78]),
        Spacer(1, 3 * mm),
        Paragraph("preferred_modality is a sampling preference and does not determine trigger_type. audio_dependency and vision_dependency describe whether a task requires, benefits from, or does not use audio or vision. Validation checks that required audio is labeled speech or sound, required vision is labeled visual, and every response in a vision-required ability includes visual.", styles["body"]),
        PageBreak(),
        Paragraph("Video and response timing", styles["section"]),
    ]

    if duration_stats:
        mean_duration, median_duration, min_duration, max_duration = duration_stats
        story.append(Paragraph(
            f"Video duration across unique videos: mean {mean_duration:.2f}s, median {median_duration:.2f}s, range {min_duration:.2f}-{max_duration:.2f}s.",
            styles["body"]))
        edges = [0, 60, 120, 180, 240, 300, 420, 540, 660, math.inf]
        labels = ["<1m", "1-2m", "2-3m", "3-4m", "4-5m", "5-7m", "7-9m", "9-11m", ">11m"]
        story.append(Histogram("Video duration distribution", duration_values, edges, labels, ORANGE))
    else:
        story.append(Paragraph("No video duration metadata is available.", styles["body"]))

    if delay_stats:
        mean_delay, median_delay, p90, min_delay, max_delay = delay_stats
        story.append(Paragraph(
            f"Annotated delay from question_time to the first non-tool response trigger_time: mean {mean_delay:.2f}s, median {median_delay:.2f}s, P90 {p90:.2f}s, range {min_delay:.2f}-{max_delay:.2f}s. This is not model inference latency.",
            styles["body"]))
        edges = [0, 1, 3, 10, 30, 60, 120, 180, 240, 360, math.inf]
        labels = ["<1s", "1-3s", "3-10s", "10-30s", "30-60s", "1-2m", "2-3m", "3-4m", "4-6m", ">6m"]
        story.append(Histogram("Question-to-response delay", delays, edges, labels, TEAL))
        delay_rows = [[Paragraph("Interaction type", styles["header"]), Paragraph("Count", styles["header"]),
                       Paragraph("Mean delay", styles["header"]), Paragraph("Median delay", styles["header"])]]
        for name in ("rtd", "pr", "agent"):
            values = mode_delays.get(name, [])
            if not values:
                continue
            delay_rows.append([Paragraph(name.upper() if name != "agent" else "Agent", styles["cell"]),
                               Paragraph(str(len(values)), styles["center"]),
                               Paragraph(f"{statistics_mean(values):.2f}s", styles["center"]),
                               Paragraph(f"{_median(values):.2f}s", styles["center"])])
        story.append(_table(delay_rows, [doc.width * .34, doc.width * .18, doc.width * .24, doc.width * .24]))
    else:
        story.append(Paragraph("No question and response timestamp pairs are available.", styles["body"]))

    try:
        doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
        os.replace(pending, output_path)
    finally:
        pending.unlink(missing_ok=True)
    return {
        "path": str(output_path),
        "samples": sample_count,
        "videos": len(durations),
        "modes": dict(modes),
        "modalities": dict(modality_counts),
        "filtered_agent_negatives": len(raw_rows) - len(rows),
    }


def statistics_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
