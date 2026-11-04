# -*- coding: utf-8 -*-
"""Load layout/recognition sub-task metrics for evaluation and Excel.

The primary interface remains ``work/logs/subtask.json``.  When that file is
missing or only partially filled, this module extracts the same metrics from
the training logs produced by PaddleDetection and PaddleOCR.  Explicit values
in ``subtask.json`` always win over inferred values.
"""
import argparse
import json
import os
import re
from pathlib import Path

import cfg


SUBTASK_FIELDS = (
    "layout_mAP50",
    "layout_mAP50_95",
    "rec_acc",
    "rec_ned",
)

_LAYOUT_MAP95_RE = re.compile(
    r"Average Precision.*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|.*=\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+))"
)
_LAYOUT_MAP50_RE = re.compile(
    r"Average Precision.*IoU=0\.50\s*\|\s*area=\s*all\s*\|.*=\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+))"
)
_BEST_AP_RE = re.compile(
    r"Best test bbox ap is\s+(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+))"
)
_REC_BEST_RE = re.compile(
    r"best metric,\s*acc:\s*(?P<acc>[-+]?(?:\d+(?:\.\d*)?|\.\d+))"
    r".*?norm_edit_dis:\s*(?P<ned>[-+]?(?:\d+(?:\.\d*)?|\.\d+))",
    re.IGNORECASE,
)
_REC_CUR_RE = re.compile(
    r"cur metric,\s*acc:\s*(?P<acc>[-+]?(?:\d+(?:\.\d*)?|\.\d+))"
    r".*?norm_edit_dis:\s*(?P<ned>[-+]?(?:\d+(?:\.\d*)?|\.\d+))",
    re.IGNORECASE,
)


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_layout(value):
    value = _number(value)
    if value is not None and value > 1.0:
        value /= 100.0
    return value


def _normalise_acc(value):
    value = _number(value)
    # Excel's existing sub-task contract stores recognition accuracy as a
    # percentage (for example 71.8), while PaddleOCR logs it in [0, 1].
    if value is not None and 0.0 <= value <= 1.0:
        value *= 100.0
    return value


def _normalise_ned(value):
    value = _number(value)
    if value is not None and value > 1.0:
        value /= 100.0
    return value


def _subtask_path():
    return cfg.LOG_DIR / "subtask.json"


def _read_subtask_file():
    path = _subtask_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _env_log(kind, train_on):
    return os.environ.get(f"VIE_{kind.upper()}_LOG_{train_on.upper()}")


def _candidate_logs(kind, train_on):
    """Return deterministic default locations for D1 and D2 training logs."""
    explicit = _env_log(kind, train_on)
    if explicit:
        return [Path(explicit).expanduser()]

    root = cfg.ROOT
    work = cfg.WORK
    if kind == "layout":
        if train_on == "d1":
            paths = [
                root / "outputs" / "layout" / "train.log",
                work / "models" / "layout_d1" / "train.log",
                work / "logs" / "layout_d1.log",
                root / "logs" / "layout_d1.log",
            ]
        else:
            paths = [
                work / "models" / "layout_d2" / "train.log",
                root / "work" / "models" / "layout_d2" / "train.log",
                root / "outputs" / "layout_d2" / "train.log",
                work / "logs" / "layout_d2.log",
                root / "logs" / "layout_d2.log",
            ]
    else:
        if train_on == "d1":
            paths = [
                root / "outputs" / "recognition" / "train.log",
                work / "models" / "rec_d1" / "train.log",
                work / "logs" / "rec_d1.log",
                root / "logs" / "rec_d1.log",
            ]
        else:
            paths = [
                work / "models" / "rec_d2" / "train.log",
                root / "work" / "models" / "rec_d2" / "train.log",
                root / "outputs" / "recognition_d2" / "train.log",
                work / "logs" / "rec_d2.log",
                root / "logs" / "rec_d2.log",
            ]
    return paths


def _first_log(kind, train_on):
    for path in _candidate_logs(kind, train_on):
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def parse_layout_log(path):
    """Return best COCO AP50/AP50:95 from one PaddleDetection log."""
    records = []
    current = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _LAYOUT_MAP95_RE.search(line)
        if match:
            if "mAP50_95" in current:
                records.append(current)
                current = {}
            current["mAP50_95"] = float(match.group("value"))
            continue
        match = _LAYOUT_MAP50_RE.search(line)
        if match:
            current["mAP50"] = float(match.group("value"))
            continue
        match = _BEST_AP_RE.search(line)
        if match and current:
            current["best_score"] = float(match.group("value"))
            records.append(current)
            current = {}
    if current:
        records.append(current)

    complete = [row for row in records if "mAP50_95" in row]
    if not complete:
        return {}
    # PaddleDetection selects the best checkpoint using bbox AP@[.50:.95].
    # Use the highest complete COCO block, retaining its paired AP50 value.
    best = max(enumerate(complete), key=lambda item: (item[1]["mAP50_95"], item[0]))[1]
    return {
        "mAP50": _normalise_layout(best.get("mAP50")),
        "mAP50_95": _normalise_layout(best.get("mAP50_95")),
        "source": str(path),
    }


def parse_recognition_log(path):
    """Return best accuracy/NED from a PaddleOCR recognition log."""
    best_rows = []
    current_rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _REC_BEST_RE.search(line)
        if match:
            best_rows.append(
                {
                    "acc": float(match.group("acc")),
                    "ned": float(match.group("ned")),
                }
            )
            continue
        match = _REC_CUR_RE.search(line)
        if match:
            current_rows.append(
                {
                    "acc": float(match.group("acc")),
                    "ned": float(match.group("ned")),
                }
            )
    rows = best_rows or current_rows
    if not rows:
        return {}
    best = max(enumerate(rows), key=lambda item: (item[1]["acc"], item[0]))[1]
    return {
        "acc": _normalise_acc(best.get("acc")),
        "ned": _normalise_ned(best.get("ned")),
        "source": str(path),
    }


def infer_subtask(train_on):
    """Infer all available sub-task values for ``d1`` or ``d2``."""
    if train_on not in {"d1", "d2"}:
        return {}
    out = {}
    layout_log = _first_log("layout", train_on)
    if layout_log:
        out["layout"] = parse_layout_log(layout_log)
    rec_log = _first_log("rec", train_on)
    if rec_log:
        out["rec"] = parse_recognition_log(rec_log)
    return out


def load_subtask_metrics(train_on):
    """Return flattened metrics used by ``Method.subtask()`` and Excel."""
    out = {key: None for key in SUBTASK_FIELDS}
    if train_on not in {"d1", "d2"}:
        return out

    stored = _read_subtask_file()
    inferred = infer_subtask(train_on)
    stored_layout = stored.get(f"layout_{train_on}") or {}
    stored_rec = stored.get(f"rec_{train_on}") or {}
    if not isinstance(stored_layout, dict):
        stored_layout = {}
    if not isinstance(stored_rec, dict):
        stored_rec = {}
    inferred_layout = inferred.get("layout") or {}
    inferred_rec = inferred.get("rec") or {}

    out["layout_mAP50"] = _normalise_layout(
        stored_layout.get("mAP50")
        if stored_layout.get("mAP50") is not None
        else inferred_layout.get("mAP50")
    )
    out["layout_mAP50_95"] = _normalise_layout(
        stored_layout.get("mAP50_95")
        if stored_layout.get("mAP50_95") is not None
        else inferred_layout.get("mAP50_95")
    )
    out["rec_acc"] = _normalise_acc(
        stored_rec.get("acc")
        if stored_rec.get("acc") is not None
        else inferred_rec.get("acc")
    )
    out["rec_ned"] = _normalise_ned(
        stored_rec.get("ned")
        if stored_rec.get("ned") is not None
        else inferred_rec.get("ned")
    )
    return out


def refresh_subtask_log():
    """Fill missing ``subtask.json`` entries from discovered training logs."""
    data = _read_subtask_file()
    changed = False
    for train_on in ("d1", "d2"):
        inferred = infer_subtask(train_on)
        if not inferred:
            continue
        layout = inferred.get("layout") or {}
        rec = inferred.get("rec") or {}
        layout_key = f"layout_{train_on}"
        rec_key = f"rec_{train_on}"
        if not isinstance(data.get(layout_key), dict):
            data[layout_key] = {}
        if not isinstance(data.get(rec_key), dict):
            data[rec_key] = {}
        target_layout = data[layout_key]
        target_rec = data[rec_key]
        for key in ("mAP50", "mAP50_95"):
            if target_layout.get(key) is None and layout.get(key) is not None:
                target_layout[key] = layout[key]
                changed = True
        for key in ("acc", "ned"):
            if target_rec.get(key) is None and rec.get(key) is not None:
                target_rec[key] = rec[key]
                changed = True
        if layout.get("source") and target_layout.get("source") is None:
            target_layout["source"] = layout["source"]
            changed = True
        if rec.get("source") and target_rec.get("source") is None:
            target_rec["source"] = rec["source"]
            changed = True

    if changed:
        path = _subtask_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cfg.add_path_args(parser)
    args = parser.parse_args()
    cfg.configure_paths(args)
    refresh_subtask_log()
    for train_on in ("d1", "d2"):
        print(f"{train_on}: {load_subtask_metrics(train_on)}")
    print(f"-> {_subtask_path()}")


if __name__ == "__main__":
    main()
