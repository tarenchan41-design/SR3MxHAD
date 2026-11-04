"""Zero-shot VLM STOP scores for protocol section 18 (W4). Runs on the 5090; standalone (no imports from this repository).

Input:  manifest.jsonl written by `python -m experiments.v3_stop_probe2 vlm-manifest` (one line per down-view frame of the
        pilot training maps: split, row, image path relative to the dataset root, target phrase, question).
Output: <out>/train.npy and <out>/val_seen.npy, P(yes) per prepared row, and <out>/meta.json. Copy them back to the
        --vlm-dir of v3_stop_probe2 and run its probe stage.
P(yes) is read from the next-token logits after the chat prompt: softmax mass on the "yes" tokens over "yes" + "no" tokens.
Needs torch, transformers >= 4.57 (Qwen3-VL), accelerate, pillow. Resumes from <out>/progress.npz.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def p_yes(yes_logits, no_logits):
    yes, no = np.atleast_2d(yes_logits).astype(np.float64), np.atleast_2d(no_logits).astype(np.float64)
    m = np.maximum(yes.max(1, keepdims=True), no.max(1, keepdims=True))
    a, b = np.exp(yes - m).sum(1), np.exp(no - m).sum(1)
    return a / (a + b)


def first_tokens(tokenizer, words):
    return sorted({tokenizer.encode(w, add_special_tokens=False)[0] for w in words})


def main():
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True); p.add_argument("--root", required=True, help="dataset root the manifest paths are relative to")
    p.add_argument("--out", required=True); p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--smoke", type=int, default=None, help="smoke test: N frames within 10 m of the target and N beyond 60 m; prints both means, writes nothing")
    args = p.parse_args()

    lines = [json.loads(s) for s in open(args.manifest, encoding="utf-8")]
    if args.smoke:
        rng = np.random.default_rng(0)
        near = [i for i, line in enumerate(lines) if line["distance_m"] <= 10]
        far = [i for i, line in enumerate(lines) if line["distance_m"] >= 60]
        picked = list(rng.choice(near, args.smoke, replace=False)) + list(rng.choice(far, args.smoke, replace=False))
        lines = [lines[i] for i in picked]
    args.limit = args.smoke
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    yes_ids = first_tokens(processor.tokenizer, ["yes", "Yes", " yes", " Yes", "YES"])
    no_ids = first_tokens(processor.tokenizer, ["no", "No", " no", " No", "NO"])
    if set(yes_ids) & set(no_ids):
        raise SystemExit("yes and no share a first token")

    scores = np.full(len(lines), np.nan)
    progress = out / "progress.npz"
    start = 0
    if progress.exists() and not args.limit:
        saved = np.load(progress)
        if len(saved["scores"]) == len(lines):
            scores, start = saved["scores"].copy(), int(saved["cursor"])
            print("resuming at", start, flush=True)
    started = time.monotonic()
    for i in range(start, len(lines), args.batch_size):
        batch = lines[i:i + args.batch_size]
        messages = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": line["question"]}]}] for line in batch]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        images = [Image.open(Path(args.root) / line["image"]).convert("RGB") for line in batch]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            logits = model(**inputs).logits[:, -1, :].float().cpu().numpy()
        scores[i:i + len(batch)] = p_yes(logits[:, yes_ids], logits[:, no_ids])
        done = i + len(batch)
        if (done // args.batch_size) % 50 == 0 or done == len(lines):
            if not args.limit:
                np.savez(progress, scores=scores, cursor=done)
            rate = (done - start) / (time.monotonic() - started)
            print(f"{done}/{len(lines)}, {rate:.1f} frames/s, mean P(yes) so far {np.nanmean(scores[:done]):.3f}", flush=True)
    if not np.isfinite(scores).all():
        raise SystemExit("Some frames have no score")
    if args.smoke:
        n = args.smoke
        print(f"smoke test: mean P(yes) within 10 m {scores[:n].mean():.3f}, beyond 60 m {scores[n:].mean():.3f}; "
              f"share > 0.5: {np.mean(scores[:n] > .5):.2f} vs {np.mean(scores[n:] > .5):.2f}; "
              f"{2 * n / (time.monotonic() - started):.1f} frames/s", flush=True)
        for line, s in list(zip(lines, scores))[:3] + list(zip(lines, scores))[n:n + 3]:
            print(f"  {line['distance_m']:7.1f} m  P(yes) {s:.3f}  {line['question']}", flush=True)
        return
    for split in ("train", "val_seen"):
        rows = [(line["row"], s) for line, s in zip(lines, scores) if line["split"] == split]
        if not rows:
            continue
        array = np.full(max(r for r, _ in rows) + 1, np.nan, dtype=np.float32)
        for r, s in rows:
            array[r] = s
        np.save(out / f"{split}.npy", array)
    meta = {"model": args.model, "revision": getattr(model.config, "_commit_hash", None), "transformers": transformers.__version__,
            "torch": torch.__version__, "yes_ids": yes_ids, "no_ids": no_ids, "frames": len(lines), "limit": args.limit,
            "seconds": time.monotonic() - started, "question_example": lines[0]["question"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", out, meta, flush=True)


if __name__ == "__main__":
    main()
