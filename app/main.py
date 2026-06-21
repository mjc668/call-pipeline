import os
import json
import re
import time
import uuid
import threading
import subprocess
import traceback
import yaml
import httpx
import torch
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Query

CONFIG_PATH = Path("/config/config.yaml")
QUEUE_DIR = Path("/data/queue")
OUTPUT_DIR = Path("/output")

SAMPLE_RATE = 16000
SPEAKER_TOLERANCE = 0.3
DEFAULT_HTTPX_TIMEOUT = 300.0

config = {}
diarization_pipeline = None
MODEL_LOCK = threading.Lock()

app = FastAPI(title="Call Pipeline")
httpx_client = httpx.Client(timeout=DEFAULT_HTTPX_TIMEOUT)


def load_config():
    global config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    if not config.get("hf_token"):
        config["hf_token"] = os.environ.get("HF_TOKEN", "")


def format_timestamp(ts):
    t = time.strptime(ts.split("T")[1][:8], "%H:%M:%S")
    return time.strftime("%H:%M", t)


def format_date(ts):
    return ts.split("T")[0]


def resample_to_16k(audio_path):
    """Resample audio to 16kHz mono WAV for transcription/diarization.
    
    Returns path to the resampled file (audio_path.stem + ".16k.wav").
    """
    resampled = audio_path.with_name(audio_path.stem + ".16k.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path),
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-sample_fmt", "s16",
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
    """Transcribe resampled audio via the speaches (Whisper) API.
    
    Returns (words, segments) where words is a list of {start, end, word}
    dicts and segments is a list of {start, end, text} dicts.
    Raises RuntimeError on non-200 response.
    """
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
    """Run pyannote speaker diarization on resampled audio.
    
    Returns a list of {start, end, speaker} segments.
    Requires hf_token configured and GPU recommended.
    """
    pipeline = get_diarization_pipeline()
    diarization = pipeline({"audio": str(resampled_path)}, min_speakers=2)

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


def get_speaker(t, diarization_segments, tolerance=SPEAKER_TOLERANCE):
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


def format_entry(caller, duration, transcript, ts, audio_relpath=None, meta=None):
    time_str = format_timestamp(ts)
    duration_str = ""
    if duration:
        mins, secs = divmod(int(duration), 60)
        duration_str = f" ({mins}m {secs}s)" if mins else f" ({secs}s)"
    audio_link = ""
    if audio_relpath:
        audio_link = f"\n🔊 ![[{audio_relpath}]]\n"
    meta_line = ""
    if meta:
        parts = []
        for k in ("from", "to", "caller_id_name", "caller_id_number", "call_id", "cdr_id", "interaction_id", "account_id"):
            v = meta.get(k)
            if v:
                label = k.replace("_", " ").title()
                parts.append(f"{label}: {v}")
        if parts:
            meta_line = f"*{' | '.join(parts)}*\n\n"
    return (
        f"\n## {time_str} - {caller or 'Unknown'}{duration_str}\n\n"
        f"{meta_line}{transcript}\n\n{audio_link}---\n"
    )


def warmup_speaches():
    speaches_url = config.get("speaches_url", "http://10.112.200.5:8002")
    model = config.get("speaches_model", "Systran/faster-whisper-small")
    import io, struct, wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(struct.pack(f"<{SAMPLE_RATE}h", *[0]*SAMPLE_RATE))
    buf.seek(0)
    try:
        r = httpx_client.post(
            f"{speaches_url}/v1/audio/transcriptions",
            data={"model": model, "language": "en"},
            files={"file": ("silence.wav", buf, "audio/wav")},
        )
        print(f"Warm-up: speaches ready ({r.status_code})")
    except Exception as e:
        print(f"Warm-up: speaches not ready yet ({e})")


def worker_loop():
    """Background loop monitoring QUEUE_DIR for new audio files.
    
    Picks up files in order of modification time, resamples, transcribes
    via speaches, optionally diarizes via pyannote, and writes formatted
    Markdown entries to OUTPUT_DIR.
    """
    warmup_speaches()
    while True:
        print("Worker: scanning queue ...")
        try:
            files = [
                f for f in QUEUE_DIR.iterdir()
                if f.suffix not in (".meta",)
                and ".processing" not in f.stem
                and ".16k" not in f.stem
                and not f.name.startswith(".")
            ]
        except FileNotFoundError:
            print("Worker: queue directory not found, retrying in 5s ...")
            time.sleep(5)
            continue

        if not files:
            print("Worker: queue empty, sleeping 5s ...")
            time.sleep(5)
            continue

        print(f"Worker: found {len(files)} file(s) to process")
        for af in sorted(files, key=lambda p: p.stat().st_mtime):
            meta_path = af.with_suffix(".meta")
            proc_path = af.with_name(af.stem + ".processing.wav")
            resampled = None

            try:
                af.rename(proc_path)
                print(f"Processing {af.name} ...")
                meta = {}
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                print(f"  Meta: caller={meta.get('caller','')}")

                print(f"  Resampling ...")
                resampled = resample_to_16k(proc_path)
                print(f"  Transcribing ...")
                words, segments = transcribe_with_speaches(resampled)
                print(f"  Transcription: {len(words)} words, {len(segments)} segments")

                enable_diarization = config.get("enable_diarization", False)
                hf_token = config.get("hf_token", "")

                if enable_diarization and hf_token:
                    print(f"  Diarizing ...")
                    diarization_segments = diarize_with_pyannote(resampled)
                    print(f"  Merging diarization ({len(diarization_segments)} segs) ...")
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

                print(f"  Transcript length: {len(transcript)} chars")
                resampled.unlink(missing_ok=True)

                call_id = meta.get("call_id", af.stem)
                ts = meta.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
                date = format_date(ts)
                year, month, day = date.split("-")
                day_dir = OUTPUT_DIR / year / month / day
                output_file = day_dir / f"transcription-{date}.md"

                audio_relpath = None
                try:
                    audio_dir = day_dir / "audio"
                    audio_dir.mkdir(parents=True, exist_ok=True)
                    mp3_path = audio_dir / f"{call_id}.mp3"
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", str(proc_path),
                         "-codec:a", "libmp3lame", "-b:a", "64k", str(mp3_path)],
                        capture_output=True
                    )
                    audio_relpath = f"audio/{call_id}.mp3"
                    print(f"  MP3 saved to {mp3_path}")
                except Exception as e:
                    print(f"  MP3 conversion failed: {e}")

                entry = format_entry(
                    meta.get("caller", ""),
                    meta.get("duration", 0),
                    transcript,
                    ts,
                    audio_relpath=audio_relpath,
                    meta=meta,
                )

                print(f"  Writing entry to {output_file}")
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(entry)
                print(f"  Done: {call_id}")

            except httpx.RequestError as e:
                print(f"HTTP request error processing {af.name}: {e}")
                traceback.print_exc()
            except subprocess.CalledProcessError as e:
                print(f"Subprocess error processing {af.name}: {e}")
                traceback.print_exc()
            except (json.JSONDecodeError, RuntimeError) as e:
                print(f"Data error processing {af.name}: {e}")
                traceback.print_exc()
            except Exception as e:
                print(f"Unexpected error processing {af.name}: {e}")
                traceback.print_exc()
            finally:
                for p in [proc_path, meta_path]:
                    p.unlink(missing_ok=True)
                if resampled:
                    resampled.unlink(missing_ok=True)

        print("Worker: batch complete, sleeping 5s ...")
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
    if config.get("enable_diarization", False) and not config.get("hf_token"):
        print("WARNING: diarization enabled but hf_token not set — diarization will be skipped")
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
    """Receive a call recording with basic metadata.
    
    Validates token, checks file size, writes audio and .meta
    to QUEUE_DIR for the worker to process.
    """
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


@app.put("/recording/{token}/{rest:path}")
async def receive_spintel_recording(
    token: str,
    rest: str,
    request: Request,
    from_: str = Query(default="", alias="from"),
    to: str = Query(default=""),
    caller_id_name: str = Query(default=""),
    caller_id_number: str = Query(default=""),
    call_id: str = Query(default=""),
    cdr_id: str = Query(default=""),
    interaction_id: str = Query(default=""),
    account_id: str = Query(default=""),
):
    """Receive a call recording from Spintel with enhanced metadata.
    
    Supports query parameters for caller ID, CDR, and account info.
    Audio is written as MP3 to QUEUE_DIR with a .meta sidecar.
    """
    if token != config.get("token"):
        raise HTTPException(401, "invalid token")

    contents = await request.body()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > config.get("max_file_size_mb", 200):
        raise HTTPException(413, f"file too large: {size_mb:.1f} MB")

    uid = str(uuid.uuid4())
    filename = f"{uid}.mp3"
    meta = {
        "call_id": call_id or uid,
        "caller": caller_id_number or from_ or "",
        "duration": 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "original_filename": rest,
        "from": from_,
        "to": to,
        "caller_id_name": caller_id_name,
        "caller_id_number": caller_id_number,
        "cdr_id": cdr_id,
        "interaction_id": interaction_id,
        "account_id": account_id,
    }

    filepath = QUEUE_DIR / filename
    filepath.write_bytes(contents)

    metapath = QUEUE_DIR / f"{uid}.meta"
    metapath.write_text(json.dumps(meta))

    return {"status": "ok", "call_id": uid}
