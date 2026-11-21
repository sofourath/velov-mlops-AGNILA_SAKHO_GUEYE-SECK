"""TP5a : DistilBERT en float32 contre int8. Taille, latence, précision : le triplet (EX-17).

    pip install -r requirements-quantization.txt
    python scripts/bench_quantization.py                      # SST-2, 872 phrases de validation
    python scripts/bench_quantization.py --n-eval 200         # plus court
    python scripts/bench_quantization.py --random-weights     # sans accès à Hugging Face

Variantes comparées (quantization post-entraînement, couches Linear) :
    float32           le modèle de référence
    int8 torch.ao     torch.ao.quantization.quantize_dynamic : déprécié, encore le plus rapide sur CPU
    int8 torchao      Int8DynamicActivationInt8WeightConfig, si torchao est installé

Protocole (S5 · 12) : un seul thread, échauffement non compté, une phrase de --seq-len tokens,
médiane et p95. La précision se mesure sur les mêmes phrases pour toutes les variantes, avec
l'accord avec float32 : deux modèles à 91 % ne se trompent pas forcément sur les mêmes phrases.
--random-weights garde l'architecture mais tire les poids au hasard : taille et latence restent
valables, la précision non (elle n'est pas affichée).
"""

from __future__ import annotations

import argparse
import copy
import io
import platform
import time
import warnings

import numpy as np
import torch

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


def load(random_weights: bool):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DistilBertConfig

    if random_weights:
        torch.manual_seed(0)
        return None, AutoModelForSequenceClassification.from_config(DistilBertConfig(num_labels=2)).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return tokenizer, AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()


def variants(model) -> dict:
    out = {"float32": model}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from torch.ao.quantization import quantize_dynamic

        out["int8 torch.ao (déprécié)"] = quantize_dynamic(copy.deepcopy(model), {torch.nn.Linear}, dtype=torch.qint8)
    try:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

        m = copy.deepcopy(model)
        quantize_(m, Int8DynamicActivationInt8WeightConfig())
        out["int8 torchao"] = m
    except ImportError:
        print("torchao absent : variante ignorée (pip install torchao)")
    return out


def size_mb(model) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


@torch.inference_mode()
def latency_ms(model, inputs: dict, n: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        model(**inputs)
    t = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        model(**inputs)
        t[i] = (time.perf_counter() - t0) * 1000
    return float(np.percentile(t, 50)), float(np.percentile(t, 95))


@torch.inference_mode()
def predict(model, batches: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    logits = torch.cat([model(**b).logits for b in batches]).float()
    return logits.argmax(-1).numpy(), logits.numpy()


def sst2_validation(n: int) -> tuple[list[str], np.ndarray]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("nyu-mll/glue", "sst2/validation-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path).head(n)
    return df["sentence"].tolist(), df["label"].to_numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-eval", type=int, default=872, help="phrases de validation SST-2 (872 au total)")
    parser.add_argument("--n-latency", type=int, default=200, help="mesures de latence par variante")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=128, help="longueur de la phrase chronométrée")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--random-weights", action="store_true", help="sans Hugging Face : précision non mesurée")
    args = parser.parse_args()
    if args.n_latency < 20 or args.n_eval < 1 or args.seq_len < 8:
        parser.error("--n-latency >= 20, --n-eval >= 1, --seq-len >= 8")

    torch.set_num_threads(args.threads)
    tokenizer, model = load(args.random_weights)

    if tokenizer is None:
        gen = torch.Generator().manual_seed(0)
        ids = torch.randint(1000, 30000, (1, args.seq_len), generator=gen)
        timed = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        eval_ids = torch.randint(1000, 30000, (64, 32), generator=gen)
        batches, labels = [{"input_ids": eval_ids, "attention_mask": torch.ones_like(eval_ids)}], None
    else:
        sentences, labels = sst2_validation(args.n_eval)
        enc = tokenizer(["word " * args.seq_len], truncation=True, max_length=args.seq_len, return_tensors="pt")
        timed = dict(enc)
        batches = [
            dict(tokenizer(sentences[i : i + 32], padding=True, truncation=True, max_length=128, return_tensors="pt"))
            for i in range(0, len(sentences), 32)
        ]

    print(f"torch {torch.__version__}, {args.threads} thread(s), {platform.processor() or platform.machine()}")
    rows, ref_pred, ref_logits = [], None, None
    for name, m in variants(model).items():
        p50, p95 = latency_ms(m, timed, args.n_latency, args.warmup)
        pred, logits = predict(m, batches)
        if ref_pred is None:
            ref_pred, ref_logits = pred, logits
        rows.append(
            {
                "variante": name,
                "taille Mo": f"{size_mb(m):.0f}",
                "p50 ms": f"{p50:.0f}",
                "p95 ms": f"{p95:.0f}",
                "précision": "n/a" if labels is None else f"{(pred == labels).mean():.3f}",
                "accord fp32": f"{(pred == ref_pred).mean():.3f}",
                "écart max logits": f"{np.abs(logits - ref_logits).max():.3f}",
            }
        )

    head = list(rows[0])
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        print("| " + " | ".join(r.values()) + " |")
    if labels is None:
        print("\nPoids aléatoires : taille et latence valables, précision non mesurée.")
    else:
        print(f"\nPrécision sur {len(labels)} phrases SST-2 ; latence sur une phrase de {args.seq_len} tokens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
