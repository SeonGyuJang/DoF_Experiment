import argparse
import json
import math
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Optional, Dict
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def word_tokens(text: str) -> List[str]:
    tokens = []
    cur = []
    for ch in text:
        if ch.isalpha():
            cur.append(ch.lower())
        else:
            if cur:
                tokens.append(''.join(cur))
                cur = []
    if cur:
        tokens.append(''.join(cur))
    return tokens

_SENT_SPLIT_RE = re.compile(r'(?<=[\.!?。！？])\s+|[\r\n]+')

def sentence_split(text: str) -> List[str]:
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    if parts:
        return parts
    parts = []
    cur = []
    for ch in text:
        cur.append(ch)
        if ch in '.!?':
            parts.append(''.join(cur).strip())
            cur = []
    if cur:
        parts.append(''.join(cur).strip())
    return [s for s in parts if s]

def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def mtld(tokens: List[str], ttr_threshold: float = 0.72, min_seg: int = 10) -> float:
    def _mtld_one(seq: List[str]) -> float:
        types = set()
        token_count = 0
        factor_count = 0.0
        types_count = 0
        for tok in seq:
            token_count += 1
            if tok not in types:
                types.add(tok)
                types_count += 1
            ttr = types_count / token_count if token_count > 0 else 0.0
            if ttr <= ttr_threshold and token_count >= min_seg:
                factor_count += 1.0
                types.clear()
                token_count = 0
                types_count = 0
        if token_count > 0:
            ttr = types_count / token_count
            if (1 - ttr_threshold) > 0:
                factor_count += (1 - ttr) / (1 - ttr_threshold)
        return len(seq) / factor_count if factor_count > 0 else float(len(seq))
    if len(tokens) == 0:
        return 0.0
    return ( _mtld_one(tokens) + _mtld_one(list(reversed(tokens))) ) / 2.0

VOWELS = "aeiouy"

def syllable_count(word: str) -> int:
    w = word.lower()
    if not w:
        return 0
    count = 0
    prev_vowel = False
    for ch in w:
        is_v = ch in VOWELS
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if w.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)

_ALPHA_RE = re.compile(r'[A-Za-z]')

def is_mostly_english(text: str, thresh: float = 0.6) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    eng = sum(1 for ch in letters if 'a' <= ch.lower() <= 'z')
    return (eng / len(letters)) >= thresh

def fkgl_safe(text: str) -> float:
    if not is_mostly_english(text):
        return np.nan

    sents = sentence_split(text)
    words = word_tokens(text)
    W = len(words)
    S = len(sents)

    if W == 0 or S == 0:
        return np.nan

    if (W / S) > 100:  
        return np.nan

    syls = sum(syllable_count(w) for w in words)
    syl_per_word = syls / W

    if not (0.8 <= syl_per_word <= 3.5):
        return np.nan

    return float(0.39 * (W / S) + 11.8 * syl_per_word - 15.59)

def mean_sentence_length(text: str) -> float:
    sents = sentence_split(text)
    if not sents:
        return float(len(word_tokens(text)))
    lengths = [len(word_tokens(s)) for s in sents]
    return float(np.mean(lengths)) if lengths else 0.0

TT_METRICS = ["MTLD", "MSL", "FKGL"]

def load_jsonl_rows(input_dir: Path) -> pd.DataFrame:
    files = sorted([p for p in input_dir.glob("*.jsonl") if p.is_file()])
    rows = []
    for p in files:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("error") not in (None, "",):
                    continue
                cont = (obj.get("continuation") or "").strip()
                if not cont:
                    continue
                rows.append({
                    "index": int(obj.get("index", -1)),
                    "dof_value": float(obj["dof_value"]),
                    "sample_id": int(obj.get("sample_id", obj.get("index", -1))),
                    "continuation": cont
                })
    if not rows:
        raise RuntimeError("No valid rows found.")
    return pd.DataFrame(rows)

def compute_metrics_per_sample(
    df: pd.DataFrame,
    show_progress: bool = True
) -> pd.DataFrame:
    metrics: Dict[str, list] = {k: [] for k in TT_METRICS}

    iterator = tqdm(df.iterrows(), total=len(df), desc="Computing metrics (per sample)") if show_progress else df.iterrows()

    for _, row in iterator:
        txt = row["continuation"]
        toks = word_tokens(txt)

        # MTLD
        try: metrics["MTLD"].append(mtld(toks))
        except Exception: metrics["MTLD"].append(np.nan)

        # MSL
        try: metrics["MSL"].append(mean_sentence_length(txt))
        except Exception: metrics["MSL"].append(np.nan)

        # FKGL
        try: metrics["FKGL"].append(fkgl_safe(txt))
        except Exception: metrics["FKGL"].append(np.nan)

    for k, v in metrics.items():
        df[k] = v
    return df

def plot_ecdf(df: pd.DataFrame, out_dir: Path, palette: Optional[List[str]] = None):
    sns.set(style="whitegrid")
    dof_order = sorted(df["dof_value"].unique())
    df["dof_value"] = pd.Categorical(df["dof_value"], categories=dof_order, ordered=True)
    metric_cols = list(TT_METRICS)

    if palette is None:
        palette = sns.color_palette(n_colors=len(dof_order))

    for metric in metric_cols:
        plot_df = df[["dof_value", metric]].dropna().copy()
        if len(plot_df) == 0:
            continue
        plt.figure(figsize=(7.6, 5.4))
        for color, d in zip(palette, dof_order):
            sub = plot_df.loc[plot_df["dof_value"] == d, metric].values
            if len(sub) == 0:
                continue
            sns.ecdfplot(x=sub, label=f"DoF {d}", color=color)
        plt.legend(title="DoF")
        plt.xlabel(metric)
        plt.ylabel("ECDF")
        plt.title(f"{metric} — ECDF by DoF")
        plt.tight_layout()
        out_path = out_dir / f"ecdf_{metric}.png"
        plt.savefig(out_path, dpi=220)
        plt.close()

def main():
    ap = argparse.ArgumentParser(description="Metric evaluation (MTLD, MSL, FKGL) with ECDF plots by DoF.")
    ap.add_argument("--input-dir", required=True, help="Directory with .jsonl files")
    ap.add_argument("--out-dir", default=None, help="Output directory for plots/CSVs")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (input_dir / "eval_TT_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading jsonl rows...")
    df = load_jsonl_rows(input_dir)
    raw_csv = out_dir / "raw_success_rows.csv"
    df.to_csv(raw_csv, index=False, encoding="utf-8")
    print(f"Saved raw rows: {raw_csv}")

    print("Computing metrics per sample...")
    df_metrics = compute_metrics_per_sample(df, show_progress=True)
    per_sample_csv = out_dir / "per_sample_metrics_TT.csv"
    df_metrics.to_csv(per_sample_csv, index=False, encoding="utf-8")
    print(f"Saved per-sample metrics: {per_sample_csv}")

    print("Plotting ECDFs (DoF-wise)...")
    plot_ecdf(df_metrics, out_dir)

    print(f"All done. Outputs saved to: {out_dir}")

if __name__ == "__main__":
    main()
