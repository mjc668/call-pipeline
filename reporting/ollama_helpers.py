import os
import json
import time
import traceback
import yaml
import httpx
from pathlib import Path

CONFIG_PATH = Path("/config/config.yaml")

DEFAULT_OLLAMA_URL = "http://10.112.200.5:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_TIMEOUT = 120


def load_ollama_config():
    """Load Ollama connection config from config.yaml, env vars, or defaults.
    
    Precedence: environment variables > config.yaml > DEFAULT_* constants.
    Returns dict with url, model, timeout keys.
    """
    cfg = {}
    config_path = os.environ.get("CONFIG_PATH", str(CONFIG_PATH))
    p = Path(config_path)
    if p.exists():
        with open(p) as f:
            raw = yaml.safe_load(f) or {}
            o = raw.get("ollama", {})
            if o:
                cfg["url"] = o.get("url")
                cfg["model"] = o.get("model")
                cfg["timeout"] = o.get("timeout")

    cfg["url"] = os.environ.get("OLLAMA_URL") or cfg.get("url") or DEFAULT_OLLAMA_URL
    cfg["model"] = os.environ.get("OLLAMA_MODEL") or cfg.get("model") or DEFAULT_OLLAMA_MODEL
    cfg["timeout"] = int(os.environ.get("OLLAMA_TIMEOUT") or cfg.get("timeout") or DEFAULT_TIMEOUT)

    return cfg


def query_ollama(text, prompt_template, retries=3, format_json=False):
    """Query Ollama with a prompt built from the given text and template.
    
    The prompt_template should contain a {text} placeholder. If format_json
    is True, the "format": "json" parameter is sent to Ollama.
    Returns the response string, or "" on failure after retries.
    """
    if not text.strip():
        return ""

    cfg = load_ollama_config()
    client = httpx.Client(timeout=cfg["timeout"])

    prompt = prompt_template.format(text=text[:3000].strip())
    payload = {
        "model": cfg["model"],
        "prompt": prompt,
        "stream": False,
    }
    if format_json:
        payload["format"] = "json"

    for attempt in range(retries):
        try:
            resp = client.post(cfg["url"], json=payload)
            if resp.status_code != 200:
                print(f"  Ollama API error: status={resp.status_code}, body={resp.text[:500]}")
                continue
            data = resp.json()
            return data.get("response", "")
        except Exception as e:
            if attempt == retries - 1:
                print(f"  Ollama error (attempt {attempt+1}/{retries}): {e}")
                traceback.print_exc()
                return ""
            time.sleep(2)

    return ""
