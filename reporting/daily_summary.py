import re
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta

OUTPUT_DIR = Path("/output")

ENTRY_RE = re.compile(
    r"^## (\d{2}:\d{2}) - (.+?)(?: \(((?:\d+m \d+s|\d+s))\))?\n\n"
    r"(?:\*(.+?)\*\n\n)?"
    r"(.*?)"
    r"(?:\n\n---|\n---)",
    re.DOTALL | re.MULTILINE,
)


def resolve_date(raw):
    """Parse date argument: literal date, "yesterday", or default to yesterday."""
    if raw == "yesterday":
        d = datetime.now() - timedelta(days=1)
        return d.strftime("%Y-%m-%d")
    if raw:
        return raw
    d = datetime.now() - timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def parse_date_entries(date_str):
    """Read transcription Markdown for a given date and extract call entries."""
    year, month, day = date_str.split("-")
    path = OUTPUT_DIR / year / month / day / f"transcription-{date_str}.md"
    if not path.exists():
        print(f"No transcription file found at {path}")
        return []

    content = path.read_text(encoding="utf-8")
    entries = []
    for m in ENTRY_RE.finditer(content):
        hour = int(m.group(1).split(":")[0])
        caller = m.group(2).strip()
        duration_raw = m.group(3)
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

        entries.append({
            "hour": hour,
            "caller": caller,
            "duration": duration,
            "transcript": transcript,
        })
    return entries


def daily_stats(entries):
    """Compute aggregate stats (total calls, duration, busiest hour, unique callers)."""
    total_calls = len(entries)
    total_duration = sum(e["duration"] for e in entries)
    hours = Counter(e["hour"] for e in entries)
    busiest_hour = max(hours, key=hours.get) if hours else None
    unique_callers = len(set(e["caller"] for e in entries if e["caller"] != "Unknown"))

    lines = []
    lines.append(f"- **Total calls**: {total_calls}")
    lines.append(f"- **Total talk time**: {total_duration // 60}m {total_duration % 60}s")
    if busiest_hour is not None:
        lines.append(f"- **Busiest hour**: {busiest_hour:02d}:00 ({hours[busiest_hour]} calls)")
    lines.append(f"- **Unique callers**: {unique_callers}")
    return "\n".join(lines)


def build_full_transcript(entries):
    """Concatenate all entries into a single text blob for Ollama ingestion."""
    parts = []
    for e in entries:
        parts.append(f"[{e['hour']:02d}:00 - {e['caller']}]\n{e['transcript']}")
    return "\n\n".join(parts)


def build_daily_summary(date_str, entries):
    """Generate a daily-summary.md file with stats and an Ollama narrative retell."""
    year, month, day = date_str.split("-")
    day_dir = OUTPUT_DIR / year / month / day
    day_dir.mkdir(parents=True, exist_ok=True)

    stats = daily_stats(entries)
    full_text = build_full_transcript(entries)

    from ollama_helpers import query_ollama

    retell_prompt = (
        "You are a business analyst reviewing a day's call transcripts. "
        "Write a narrative retelling of the day covering: what happened, "
        "key topics discussed, notable callers, and any patterns. "
        "Write 2-4 paragraphs in a natural, professional tone.\n\n"
        "Transcripts:\n{text}"
    )

    print("  Generating narrative retell via Ollama ...")
    retell = query_ollama(full_text, retell_prompt, format_json=False)
    if not retell:
        retell = "*(no summary generated)*"

    lines = []
    lines.append(f"# Daily Summary — {date_str}\n")
    lines.append("## Stats\n")
    lines.append(stats)
    lines.append("")
    lines.append("## Retell\n")
    lines.append(retell)
    lines.append("")
    lines.append("---\n")
    lines.append(f"*Auto-generated at {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")

    output_path = day_dir / "daily-summary.md"
    output_path.write_text("\n".join(lines))
    print(f"Daily summary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate daily call summary")
    parser.add_argument("--date", help="Date YYYY-MM-DD or 'yesterday' (default: yesterday)")
    args = parser.parse_args()

    date_str = resolve_date(args.date)
    print(f"Generating daily summary for {date_str} ...")
    entries = parse_date_entries(date_str)

    if not entries:
        print("No entries found for this day.")
        return

    print(f"Found {len(entries)} calls")
    build_daily_summary(date_str, entries)


if __name__ == "__main__":
    main()
