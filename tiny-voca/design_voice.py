"""Design new voices for tiny-voca, as its two voices were made (dataset n4ze3m/tiny-voca, voices.json):
Qwen3-TTS 1.7B VoiceDesign speaks one fixed reference text from a voice description, for a few seeds per design.
Listen to the clips and pick one; that clip is the voice prompt for the rendering step.

Input: a designs JSON file, a list of {"id", "label", "prompt"}.
Output, under --out: <id>-s<seed>.wav for each design and seed, and voices.json in the dataset's format (one entry per clip).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
# the reference text of the released voices F2 and M2
TEXT = ("Hey, it's good to hear from you. I was thinking we could grab coffee this Saturday, around ten thirty. Does that work for "
        "you? Oh, and I finally finished that project. I'm really happy with how it turned out!")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[31416, 31417])
    a = ap.parse_args()
    from qwen_tts import Qwen3TTSModel
    a.out.mkdir(parents=True, exist_ok=True)
    model = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    voices = {}
    for d in json.loads(a.designs.read_text()):
        for seed in a.seeds:
            torch.manual_seed(seed)
            t = time.time()
            wavs, sr = model.generate_voice_design(text=TEXT, language="English", instruct=d["prompt"])
            w = np.asarray(wavs[0], dtype=np.float32)
            w = 0.95 * w / max(float(np.max(np.abs(w))), 1e-5)   # peak normalisation, as the released references
            name = f"{d['id']}-s{seed}"
            sf.write(a.out / f"{name}.wav", w, sr)
            voices[name] = {"id": d["id"], "label": d["label"], "prompt": d["prompt"], "text": TEXT, "seed": seed, "model": MODEL,
                            "seconds": round(len(w) / sr, 2), "sample_rate": sr, "generation_seconds": round(time.time() - t, 1),
                            "audio": f"{name}.wav", "peak_normalized": True}
            print(json.dumps({k: voices[name][k] for k in ("audio", "seconds", "generation_seconds")}), flush=True)
    (a.out / "voices.json").write_text(json.dumps(voices, indent=2))


if __name__ == "__main__":
    main()
