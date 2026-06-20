import os
import json
import re
import time
import uuid
import threading
import subprocess
import yaml
import httpx
import torch
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

CONFIG_PATH = Path("/config/config.yaml")
QUEUE_DIR = Path("/data/queue")
OUTPUT_DIR = Path("/output")

config = {}
diarization_pipeline = None
MODEL_LOCK = threading.Lock()

app = FastAPI(title="Call Pipeline")
httpx_client = httpx.Client(timeout=300.0)


def load_config():
    global config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        config["hf_token"] = env_token


def format_timestamp(ts):
    t = time.strptime(ts.split("T")[1][:8], "%H:%M:%S")
    return time.strftime("%H:%M", t)


def format_date(ts):
    return ts.split("T")[0]


def resample_to_16k(audio_path):
    resampled = audio_path.with_name(audio_path.stem + ".16k.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path),
         "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
         str(resampled)],
        capture_output=True, text=True
    )
    return resampled


def get_diarization_pipeline():
    global diarization_pipeline
    with MODEL_LOCK:
        if diarization_pipeline is None:
            import torch
            _orig = torch.load
            torch.load = lambda *a, **kw: _orig(*a, **{**kw, 'weights_only': False})
            from pyannote.audio import Pipeline
            hf_token = config.get("hf_token", "")
            if not hf_token:
                raise RuntimeError("hf_token required for diarization")
            print("Loading pyannote speaker-diarization-3.1 pipeline ...")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            if torch.cuda.is_available():
                print("Moving diarization pipeline to GPU")
                pipeline.to(torch.device("cuda"))
            diarization_pipeline = pipeline
        return diarization_pipeline


def transcribe_with_speaches(resampled_path):
    speaches_url = config.get("speaches_url", "http://10.112.200.5:8002")
    model = config.get("speaches_model", "Systran/faster-whisper-small")

    with open(resampled_path, "rb") as f:
        response = httpx_client.post(
            f"{speaches_url}/v1/audio/transcriptions",
            data={
                "model": model,
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",
                "language": "en",
            },
            files={"file": ("audio.wav", f, "audio/wav")},
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"speaches API error {response.status_code}: {response.text}"
        )

    data = response.json()
    words = data.get("words")
    segments = data.get("segments", [])

    if words:
        return words, segments

    if not segments:
        full_text = data.get("text", "")
        return [], [{"start": 0, "end": 0, "text": full_text}]

    return [], segments


def diarize_with_pyannote(resampled_path):
    pipeline = get_diarization_pipeline()
    diarization = pipeline({"audio": str(resampled_path)})

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })
    return segments


def get_speaker(t, diarization_segments, tolerance=0.3):
    best = None
    best_dist = tolerance
    for seg in diarization_segments:
        if seg["start"] <= t < seg["end"]:
            return seg["speaker"]
        d = min(abs(t - seg["start"]), abs(t - seg["end"]))
        if d < best_dist:
            best_dist = d
            best = seg
    return best["speaker"] if best else None


NAME_PATTERNS = [
    re.compile(r"my name is ([A-Z][a-z]+)", re.IGNORECASE),
    re.compile(r"i'?m ([A-Z][a-z]+)", re.IGNORECASE),
    re.compile(r"i am ([A-Z][a-z]+)", re.IGNORECASE),
    re.compile(r"this is ([A-Z][a-z]+)", re.IGNORECASE),
    re.compile(r"call me ([A-Z][a-z]+)", re.IGNORECASE),
]

def assign_speaker_names(transcript):
    mapping = {}
    for line in transcript.split("\n"):
        m = re.match(r"\*\*(\w+)\*\*:\s*(.*)", line)
        if not m:
            continue
        label, text = m.group(1), m.group(2)
        if label in mapping:
            continue
        for pat in NAME_PATTERNS:
            m2 = pat.search(text)
            if m2 and m2.group(1)[0].isupper():
                mapping[label] = m2.group(1)
                break
    if not mapping:
        return transcript
    result = transcript
    for old, new in mapping.items():
        result = result.replace(f"**{old}**", f"**{new}**")
    return result


def cleanup_unknowns(transcript):
    lines = transcript.split("\n")
    result = []
    buf = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        m = re.match(r"\*\*SPEAKER_UNKNOWN\*\*:\s*(.*)", line)
        if m:
            buf.append(m.group(1).strip())
            i += 1
            continue
        if buf and re.match(r"\*\*(?!SPEAKER_UNKNOWN)\w+\*\*:\s*", line):
            text = " ".join(buf)
            rest = re.sub(r"^\*\*\w+\*\*:\s*", "", line)
            result.append(f"{line.split(':')[0]}: {text} {rest}")
            buf = []
            i += 1
            continue
        if buf:
            result.extend(f"**SPEAKER_UNKNOWN**: {w}" for w in buf)
            buf = []
        result.append(line)
        i += 1
    if buf:
        result.extend(f"**SPEAKER_UNKNOWN**: {w}" for w in buf)
    return "\n".join(result)


def merge_transcript_diarization(words, diarization_segments):
    if not words and not diarization_segments:
        return "[transcription failed]"

    # Fallback: no word-level timestamps, return speaker timeline
    if not words:
        parts = []
        for seg in diarization_segments:
            label = seg.get("speaker", "SPEAKER_UNKNOWN")
            parts.append(f"**{label}**: [{seg['start']:.1f}s-{seg['end']:.1f}s]")
        return "\n\n".join(parts) if parts else "[transcription failed]"

    output_parts = []
    current_spk = None
    current_text = []

    for word in words:
        spk = get_speaker(word["start"], diarization_segments)
        label = spk if spk else "SPEAKER_UNKNOWN"
        if label != current_spk:
            if current_text:
                output_parts.append(f"**{current_spk}**: {' '.join(current_text)}")
            current_spk = label
            current_text = [word["word"].strip()]
        else:
            current_text.append(word["word"].strip())

    if current_text:
        output_parts.append(f"**{current_spk}**: {' '.join(current_text)}")

    return "\n\n".join(output_parts) if output_parts else "[transcription failed]"


def format_entry(caller, duration, transcript, ts, audio_relpath=None):
    time_str = format_timestamp(ts)
    duration_str = ""
    if duration:
        mins, secs = divmod(int(duration), 60)
        duration_str = f" ({mins}m {secs}s)" if mins else f" ({secs}s)"
    audio_link = ""
    if audio_relpath:
        audio_link = f"\n🔊 ![[{audio_relpath}]]\n"
    return (
        f"\n## {time_str} - {caller or 'Unknown'}{duration_str}\n\n"
        f"{transcript}\n\n{audio_link}---\n"
    )


def worker_loop():
    while True:
        try:
            files = [
                f for f in QUEUE_DIR.iterdir()
                if f.suffix not in (".meta",)
                and ".processing" not in f.stem
                and ".16k" not in f.stem
                and not f.name.startswith(".")
            ]
        except FileNotFoundError:
            time.sleep(5)
            continue

        for af in sorted(files, key=lambda p: p.stat().st_mtime):
            meta_path = af.with_suffix(".meta")
            proc_path = af.with_name(af.stem + ".processing.wav")

            try:
                af.rename(proc_path)
                meta = {}
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())

                resampled = resample_to_16k(proc_path)
                words, segments = transcribe_with_speaches(resampled)

                enable_diarization = config.get("enable_diarization", False)
                hf_token = config.get("hf_token", "")

                if enable_diarization and hf_token:
                    diarization_segments = diarize_with_pyannote(resampled)
                    transcript = merge_transcript_diarization(words, diarization_segments)
                    transcript = cleanup_unknowns(transcript)
                    transcript = assign_speaker_names(transcript)
                else:
                    if words:
                        transcript = " ".join(w["word"].strip() for w in words)
                    else:
                        transcript = " ".join(s.get("text", "").strip() for s in segments)
                    if not transcript.strip():
                        transcript = "[transcription failed]"

                resampled.unlink(missing_ok=True)

                call_id = meta.get("call_id", af.stem)
                ts = meta.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
                date = format_date(ts)
                output_file = OUTPUT_DIR / f"{date}.md"

                audio_relpath = None
                try:
                    audio_dir = OUTPUT_DIR / "audio" / date
                    audio_dir.mkdir(parents=True, exist_ok=True)
                    mp3_path = audio_dir / f"{call_id}.mp3"
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", str(proc_path),
                         "-codec:a", "libmp3lame", "-b:a", "64k", str(mp3_path)],
                        capture_output=True
                    )
                    audio_relpath = f"audio/{date}/{call_id}.mp3"
                except Exception:
                    pass

                entry = format_entry(
                    meta.get("caller", ""),
                    meta.get("duration", 0),
                    transcript,
                    ts,
                    audio_relpath=audio_relpath,
                )

                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "a") as f:
                    f.write(entry)

            except Exception as e:
                print(f"Error processing {af.name}: {e}")
            finally:
                for p in [proc_path, meta_path]:
                    p.unlink(missing_ok=True)

        time.sleep(5)


@app.on_event("startup")
async def startup():
    load_config()
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Recover orphaned processing files from previous crashes
    for f in sorted(QUEUE_DIR.iterdir(), key=lambda p: p.stat().st_mtime):
        if ".processing" in f.stem:
            original_id = f.stem.split(".")[0]
            recovered = f.with_name(f"{original_id}{f.suffix}")
            print(f"Recovering stale file: {f.name} -> {recovered.name}")
            f.rename(recovered)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: enable_diarization={config.get('enable_diarization', False)}, hf_token={'set' if config.get('hf_token') else 'not set'}")
    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/recording/{token}")
async def receive_recording(
    token: str,
    audio: UploadFile = File(...),
    caller: str = Form(default=""),
    duration: int = Form(default=0),
    timestamp: str = Form(default=""),
):
    if token != config.get("token"):
        raise HTTPException(401, "invalid token")

    contents = await audio.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > config.get("max_file_size_mb", 200):
        raise HTTPException(413, f"file too large: {size_mb:.1f} MB")

    call_id = str(uuid.uuid4())
    ext = Path(audio.filename).suffix or ".wav"
    filename = f"{call_id}{ext}"
    meta = {
        "call_id": call_id,
        "caller": caller,
        "duration": duration,
        "timestamp": timestamp or time.strftime("%Y-%m-%dT%H:%M:%S"),
        "original_filename": audio.filename,
    }

    filepath = QUEUE_DIR / filename
    filepath.write_bytes(contents)

    metapath = QUEUE_DIR / f"{call_id}.meta"
    metapath.write_text(json.dumps(meta))

    return {"status": "ok", "call_id": call_id}
