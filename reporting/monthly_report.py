import os
import re
import json
import time
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta

OUTPUT_DIR = Path("/output")
REPORTS_DIR = OUTPUT_DIR / "reports"

ENTRY_RE = re.compile(
    r"^## (\d{2}:\d{2}) - (.+?)(?: \(((?:\d+m \d+s|\d+s))\))?\n\n"
    r"(?:\*(.+?)\*\n\n)?"
    r"(.*?)"
    r"(?:\n\n---|\n---)",
    re.DOTALL | re.MULTILINE,
)


def parse_monthly_files(month_prefix):
    entries = []
    year, month = month_prefix.split("-")
    paths = sorted(OUTPUT_DIR.glob(f"{year}/{month}/*/transcription-*.md"))
    if not paths:
        paths = sorted(OUTPUT_DIR.glob(f"{month_prefix}-*.md"))
    for path in paths:
        content = path.read_text()
        for m in ENTRY_RE.finditer(content):
            hour = int(m.group(1).split(":")[0])
            caller = m.group(2).strip()
            duration_raw = m.group(3)
            meta_raw = m.group(4)
            transcript = m.group(5).strip()

            duration = 0
            if duration_raw:
                if "m" in duration_raw:
                    parts = duration_raw.split()
                    mins = int(parts[0].rstrip("m"))
                    secs = int(parts[1].rstrip("s")) if len(parts) > 1 else 0
                    duration = mins * 60 + secs
                else:
                    duration = int(duration_raw.rstrip("s"))

            date_from_stem = path.stem.replace("transcription-", "")
            entries.append({
                "date": date_from_stem,
                "hour": hour,
                "caller": caller,
                "duration": duration,
                "transcript": transcript,
            })
    return entries


def busiest_hours(entries):
    hour_counts = Counter(e["hour"] for e in entries)
    return sorted(hour_counts.items())


def most_common_callers(entries, top=15):
    caller_counts = Counter(e["caller"] for e in entries)
    return caller_counts.most_common(top)


def query_ollama_for_report(transcript):
    from ollama_helpers import query_ollama

    text = transcript[:2000].strip()
    if not text:
        return {"car_brands": [], "car_models": [], "topics": []}

    prompt_template = (
        "Extract car brands, car models, and main topics from this call transcript. "
        "Return ONLY valid JSON with no explanation. "
        'Format: {{"car_brands":["..."],"car_models":["..."],"topics":["..."]}}\n\n'
        "Transcript:\n{text}"
    )
    result = query_ollama(text, prompt_template, format_json=True)
    if not result:
        return {"car_brands": [], "car_models": [], "topics": []}
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {"car_brands": [], "car_models": [], "topics": []}


def generate_report(entries, month_prefix):
    hours = busiest_hours(entries)
    callers = most_common_callers(entries)

    max_hour_count = max((c for _, c in hours), default=1)
    max_bar = 30

    all_brands = Counter()
    all_models = Counter()
    all_topics = Counter()

    for i, entry in enumerate(entries):
        print(f"  Analyzing call {i+1}/{len(entries)} ...")
        result = query_ollama_for_report(entry["transcript"])
        for brand in result.get("car_brands", []):
            all_brands[brand.strip().title()] += 1
        for model in result.get("car_models", []):
            all_models[model.strip().title()] += 1
        for topic in result.get("topics", []):
            all_topics[topic.strip().title()] += 1

    lines = []
    lines.append(f"# Monthly Report — {month_prefix}\n")
    lines.append(f"**{len(entries)} calls processed**\n")

    lines.append("## Busiest Hours\n")
    lines.append("| Hour | Calls |")
    lines.append("|------|-------|")
    for hour, count in hours:
        bar_len = max(1, round(count / max_hour_count * max_bar))
        bar = "█" * bar_len
        lines.append(f"| {hour:02d}:00 | {count} {bar}")
    lines.append("")

    lines.append("## Most Common Callers\n")
    lines.append("| Caller | Calls |")
    lines.append("|--------|-------|")
    for caller, count in callers:
        lines.append(f"| {caller} | {count} |")
    lines.append("")

    lines.append("## Top Car Brands\n")
    if all_brands:
        for brand, count in all_brands.most_common(10):
            lines.append(f"- {brand}: {count}")
    else:
        lines.append("*(no car brands detected)*")
    lines.append("")

    lines.append("## Top Car Models\n")
    if all_models:
        for model, count in all_models.most_common(10):
            lines.append(f"- {model}: {count}")
    else:
        lines.append("*(no car models detected)*")
    lines.append("")

    lines.append("## Top Topics\n")
    if all_topics:
        for topic, count in all_topics.most_common(15):
            lines.append(f"- {topic}: {count}")
    else:
        lines.append("*(no topics detected)*")
    lines.append("")

    return "\n".join(lines)


def resolve_month(month):
    if month:
        return month
    today = datetime.now()
    first_of_month = today.replace(day=1)
    last_month = first_of_month - timedelta(days=1)
    return last_month.strftime("%Y-%m")


def main():
    parser = argparse.ArgumentParser(description="Generate monthly call report")
    parser.add_argument("--month", help="Month prefix like 2026-06 (default: previous month)")
    args = parser.parse_args()

    month = resolve_month(args.month)
    print(f"Parsing files for {month} ...")
    entries = parse_monthly_files(month)
    print(f"Found {len(entries)} entries")

    if not entries:
        print("No entries found, nothing to report.")
        return

    print("Generating report ...")
    report = generate_report(entries, month)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{month}-report.md"
    report_path.write_text(report)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
