# Run locally (not in Colab) having downloaded the model/installed ollama
import json, time, requests, pandas as pd

BASE  = "http://localhost:11434"
MODEL = "llama3.2:3b"  # or "mistral:7b-instruct", "qwen2.5:7b-instruct"

SYSTEM = (
  "You are a careful social science annotator. "
  "Return ONLY valid JSON with fields {label, rationale, confidence}."
)

def annotate_one(text: str, retries: int = 1):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                'Label one of ["protest","discrimination","solidarity","uncertain"]. '
                f'Text: "{text}"\n'
                'Return ONLY JSON: {"label": "", "rationale": "", "confidence": 0.0}'
            },
        ],
        "stream": False,
        # Ask Ollama to ensure JSON output
        "format": "json",
        # Decoding options (deterministic, reproducible)
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 7,
            "num_predict": 200,
            # You can also tune: "top_k", "stop", "num_ctx"
        },
    }
    for attempt in range(retries + 1):
        r = requests.post(f"{BASE}/api/chat", json=body, timeout=120)
        r.raise_for_status()
        content = r.json()["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            if attempt < retries:
                # one gentle retry with a stricter reminder
                body["messages"][-1]["content"] += "\nReturn only VALID MINIFIED JSON. No prose."
                time.sleep(0.2)
                continue
            # last resort: surface raw text for debugging
            return {"label": None, "rationale": f"JSON parse error. Raw: {content}", "confidence": None}

def annotate_batch(texts):
    rows = []
    for t in texts:
        out = annotate_one(t)
        rows.append({"text": t, **out})
        time.sleep(0.05)  # polite pacing
    return pd.DataFrame(rows, columns=["text", "label", "rationale", "confidence"])

# --- Your inputs ---
texts = [
    "Thousands gathered in front of parliament.",
    "Volunteers cleaned the park and cooked for neighbors.",
    "He yelled slurs at a woman on the tram.",
    "Police blocked the march after clashes."
]

df = annotate_batch(texts)

print(df)