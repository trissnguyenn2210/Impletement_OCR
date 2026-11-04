# -*- coding: utf-8 -*-
"""Shared per-image visualization helpers.

The evaluator uses this module directly while it processes each image.  The
helpers are also reused by the ablation runner, but this module is not a
standalone inference command: model execution belongs to the caller.
"""
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from metrics import aggregate
from methods.base import get_method


METHODS = ("A", "B", "C", "D")
FIELD_PATHS = {
    "merchant": ("header", "merchant"),
    "address": ("header", "address"),
    "date": ("header", "date"),
    "time": ("header", "time"),
    "total": ("payment", "total"),
}
COLORS = {
    "A": (35, 105, 210),
    "B": (145, 75, 185),
    "C": (20, 145, 105),
    "D": (210, 105, 30),
}


def ensure_methods_registered():
    """Import method implementations after runtime model env is configured."""
    import methods.dummy  # noqa: F401
    import methods.real  # noqa: F401


def instantiate(method, train_on, test_on):
    """Instantiate a registered method for the ablation runner."""
    ensure_methods_registered()
    cls = get_method(method)
    selected_train = None if method == "B" or method.startswith("B_") else train_on
    try:
        return cls(train_on=selected_train, test_on=test_on)
    except TypeError:
        return cls(train_on=selected_train)


def get_font(size, bold=False):
    path = Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def value_at(doc, path):
    return (doc.get(path[0]) or {}).get(path[1])


def display_value(value, limit=80):
    if value in (None, ""):
        return "∅"
    text = str(value).replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def clean_debug(value):
    """Make method traces JSON-safe and remove private grouping fields."""
    if isinstance(value, dict):
        return {
            str(k): clean_debug(v)
            for k, v in value.items()
            if not str(k).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [clean_debug(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def fit_image(image, max_width=1450, max_height=1080):
    scale = min(max_width / image.width, max_height / image.height, 1.0)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    if size == image.size:
        return image.copy(), scale
    return image.resize(size, Image.Resampling.LANCZOS), scale


def wrapped_lines(value, width=43):
    return textwrap.wrap(
        display_value(value),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    ) or ["∅"]


def draw_debug_boxes(image, debug, scale, method):
    """Draw the debug trace exposed by A/C/D; B intentionally has no boxes."""
    if not isinstance(debug, dict):
        return
    draw = ImageDraw.Draw(image)
    color = COLORS.get(method, (30, 100, 200))
    text_font = get_font(14)

    regions = debug.get("regions") or []
    for index, region in enumerate(regions, start=1):
        box = region.get("box") if isinstance(region, dict) else None
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        x1, y1, x2, y2 = [round(float(v) * scale) for v in box[:4]]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, round(3 * scale)))
        label = str(region.get("label") or "REGION")
        text = display_value(region.get("text"), 34)
        caption = f"{index}. {label}: {text}"
        bbox = draw.textbbox((0, 0), caption, font=text_font)
        left = max(0, x1)
        top = max(0, y1 - (bbox[3] - bbox[1]) - 6)
        draw.rectangle(
            (left, top, left + bbox[2] + 8, top + bbox[3] - bbox[1] + 5),
            fill=color,
        )
        draw.text((left + 4, top + 2), caption, fill="white", font=text_font)

    lines = debug.get("ocr_lines") or []
    for index, line in enumerate(lines, start=1):
        box = line.get("box") if isinstance(line, dict) else None
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        x1, y1, x2, y2 = [round(float(v) * scale) for v in box[:4]]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, round(2 * scale)))
        caption = f"OCR {index}: {display_value(line.get('text'), 32)}"
        bbox = draw.textbbox((0, 0), caption, font=text_font)
        top = min(image.height - (bbox[3] - bbox[1]) - 5, max(0, y2 + 2))
        draw.rectangle(
            (max(0, x1), top, max(0, x1) + bbox[2] + 8,
             top + bbox[3] - bbox[1] + 5),
            fill=color,
        )
        draw.text((max(0, x1) + 4, top + 2), caption, fill="white", font=text_font)


def draw_panel(canvas, panel_x, method, image_id, pred, gt, metrics, summary, error, debug):
    panel = ImageDraw.Draw(canvas)
    left = panel_x + 28
    color = COLORS.get(method, (30, 100, 200))
    title_font = get_font(30, bold=True)
    heading_font = get_font(18, bold=True)
    body_font = get_font(17)
    small_font = get_font(14)

    panel.rectangle((panel_x, 0, canvas.width, canvas.height), fill=(247, 248, 251))
    panel.text((left, 24), f"Method {method}", fill=color, font=title_font)
    panel.text((left, 66), image_id, fill=(45, 45, 45), font=small_font)
    if error:
        panel.text((left, 94), "ERROR", fill=(190, 35, 35), font=heading_font)
        y = 126
        for line in textwrap.wrap(error, width=48):
            panel.text((left, y), line, fill=(150, 25, 25), font=small_font)
            y += 20
        return

    mfs = summary.get("MFS")
    halluc = summary.get("halluc")
    mfs_text = "N/A" if mfs is None else f"{mfs:.1f}%"
    panel.text(
        (left, 96),
        f"MFS: {mfs_text}   Halluc: {halluc:.1f}%"
        if halluc is not None
        else f"MFS: {mfs_text}",
        fill=(25, 25, 25),
        font=heading_font,
    )
    y = 142
    for field, path in FIELD_PATHS.items():
        score = metrics.get("per_field", {}).get(field)
        score_text = "N/A" if score is None else f"{score * 100:.0f}%"
        panel.text((left, y), f"{field}  [{score_text}]", fill=color, font=heading_font)
        y += 25
        for prefix, value in (("P", value_at(pred, path)), ("G", value_at(gt, path))):
            prefix_color = (25, 100, 45) if prefix == "P" and score == 1 else (55, 55, 55)
            lines = wrapped_lines(f"{prefix}: {value}", width=43)
            for line in lines[:2]:
                panel.text((left + 8, y), line, fill=prefix_color, font=body_font)
                y += 21
        y += 7
        if y > canvas.height - 190:
            break

    items = pred.get("line_items")
    item_text = "None" if items is None else str(len(items))
    panel.text((left, canvas.height - 105), f"predicted items: {item_text}", fill=(55, 55, 55), font=small_font)
    if isinstance(debug, dict) and debug.get("regions"):
        trace_text = f"layout regions: {len(debug['regions'])}"
    elif isinstance(debug, dict) and debug.get("ocr_lines"):
        trace_text = f"OCR lines: {len(debug['ocr_lines'])}"
    elif isinstance(debug, dict) and debug.get("raw_text"):
        trace_text = "VLM raw response saved in JSON"
    else:
        trace_text = "no debug trace"
    panel.text((left, canvas.height - 78), trace_text, fill=(75, 75, 75), font=small_font)
    panel.text((left, canvas.height - 49), "P = prediction   G = ground truth", fill=(90, 90, 90), font=small_font)


def render_method(method, image, image_id, pred, gt, metrics, debug, output_path, error=None):
    """Render one prediction in the same format used by the old demo."""
    base, scale = fit_image(image)
    panel_width = 670
    canvas = Image.new("RGB", (base.width + panel_width, max(base.height, 760)), "white")
    canvas.paste(base, (0, 0))
    if not error:
        draw_debug_boxes(base, debug, scale, method)
        canvas.paste(base, (0, 0))
    summary = aggregate([metrics]) if not error else {"MFS": 0.0, "halluc": None}
    draw_panel(canvas, base.width, method, image_id, pred, gt, metrics, summary, error, debug)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return canvas


def write_metadata(path, payload):
    """Write a JSON sidecar for a rendered image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
