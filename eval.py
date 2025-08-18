import argparse
import json
import math
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

_HF_AVAILABLE = True
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except Exception:
    _HF_AVAILABLE = False

STOPWORDS = set("""
a an the and or but if while of in on at by for to from with without within about across after before during between
is am are was were be been being do does did doing have has had having can could may might must shall should will would
i you he she it we they me him her us them my your his hers its our their mine yours ours theirs this that these those
as not no nor so than too very just only also into over under again further then once here there when where why how
""".split())

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

def sentence_split(text: str) -> List[str]:
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

def hdd(tokens: List[str], sample_size: int = 42) -> float:
    N = len(tokens)
    if N == 0:
        return 0.0
    s = min(sample_size, N)
    freqs = Counter(tokens)
    denom = math.comb(N, s) if N >= s else 0
    contribs = []
    for f in freqs.values():
        p0 = math.comb(N - f, s) / denom if denom > 0 else 0.0
        contribs.append(1.0 - p0)
    return float(np.mean(contribs)) if contribs else 0.0

def shannon_entropy_normalized(tokens: List[str]) -> float:
    N = len(tokens)
    if N == 0:
        return 0.0
    freqs = Counter(tokens)
    V = len(freqs)
    if V <= 1:
        return 0.0
    ps = [c / N for c in freqs.values()]
    H = -sum(p * math.log(p + 1e-12) for p in ps)
    return float(H / math.log(V))

def mean_sentence_length(text: str) -> float:
    sents = sentence_split(text)
    if not sents:
        return float(len(word_tokens(text)))
    lengths = [len(word_tokens(s)) for s in sents]
    return float(np.mean(lengths)) if lengths else 0.0

def distinct_n(tokens: List[str], n: int) -> float:
    ng = ngrams(tokens, n)
    total = len(ng)
    if total == 0:
        return 0.0
    return float(len(set(ng)) / total)

class HFPerplexity:
    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None, max_len: int = 512):
        if not _HF_AVAILABLE:
            raise RuntimeError("transformers/torch not available")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.max_len = max_len

    @torch.no_grad()
    def perplexity(self, text: str) -> float:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_len)
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
        loss = outputs.loss.item()
        ppl = float(math.exp(loss))
        return ppl

def build_unigram_model(corpus_tokens: List[List[str]], alpha: float = 1.0) -> Dict[str, float]:
    counts = Counter()
    for toks in corpus_tokens:
        counts.update(toks)
    V = len(counts)
    N = sum(counts.values())
    base = alpha / (N + alpha * V) if (N + alpha * V) > 0 else 0.0
    probs = {w: (counts[w] + alpha) / (N + alpha * V) for w in counts}
    probs["<UNK>"] = base
    return probs

def unigram_perplexity(tokens: List[str], probs: Dict[str, float]) -> float:
    if len(tokens) == 0:
        return 0.0
    ll = 0.0
    for w in tokens:
        p = probs.get(w, probs.get("<UNK>", 1e-12))
        ll += -math.log(p + 1e-12)
    xent = ll / len(tokens)
    return float(math.exp(xent))

TT_METRICS = ["MTLD", "HDD", "NormEntropy", "Perplexity", "Distinct1", "Distinct2", "MSL"]

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
                    "index": int(obj["index"]),
                    "dof_value": float(obj["dof_value"]),
                    "sample_id": int(obj.get("sample_id", obj["index"])),
                    "continuation": cont
                })
    if not rows:
        raise RuntimeError("No valid rows found.")
    return pd.DataFrame(rows)

def compute_metrics_per_sample(
    df: pd.DataFrame,
    ppl_model_name: Optional[str] = None,
    ppl_max_len: int = 512,
    unigram_alpha: float = 1.0,
    show_progress: bool = True
) -> pd.DataFrame:
    hf_ppl = None
    if ppl_model_name:
        if not _HF_AVAILABLE:
            print("[WARN] transformers not available; falling back to unigram perplexity.")
        else:
            try:
                hf_ppl = HFPerplexity(ppl_model_name, max_len=ppl_max_len)
                print(f"[INFO] Using HF model for perplexity: {ppl_model_name} (device={hf_ppl.device})")
            except Exception as e:
                print(f"[WARN] Failed to init HF model ({e}); fallback to unigram perplexity.")

    all_tokens = [word_tokens(t) for t in df["continuation"].tolist()]
    unigram_probs = build_unigram_model(all_tokens, alpha=unigram_alpha)

    metrics = {k: [] for k in TT_METRICS}

    iterator = tqdm(df.iterrows(), total=len(df), desc="Computing metrics (per sample)") if show_progress else df.iterrows()

    for _, row in iterator:
        txt = row["continuation"]
        toks = word_tokens(txt)

        try: metrics["MTLD"].append(mtld(toks))
        except Exception: metrics["MTLD"].append(np.nan)

        try: metrics["HDD"].append(hdd(toks))
        except Exception: metrics["HDD"].append(np.nan)

        try: metrics["NormEntropy"].append(shannon_entropy_normalized(toks))
        except Exception: metrics["NormEntropy"].append(np.nan)

        try: metrics["Distinct1"].append(distinct_n(toks, 1))
        except Exception: metrics["Distinct1"].append(np.nan)

        try: metrics["Distinct2"].append(distinct_n(toks, 2))
        except Exception: metrics["Distinct2"].append(np.nan)

        try: metrics["MSL"].append(mean_sentence_length(txt))
        except Exception: metrics["MSL"].append(np.nan)

        # Perplexity
        try:
            if hf_ppl is not None:
                val = hf_ppl.perplexity(txt)
            else:
                val = unigram_perplexity(toks, unigram_probs)
        except Exception:
            val = np.nan
        metrics["Perplexity"].append(val)

    for k, v in metrics.items():
        df[k] = v
    return df

def plot_violin_box_points(df: pd.DataFrame, out_dir: Path, downsample: Optional[int] = 1000):
    sns.set(style="whitegrid")
    dof_order = sorted(df["dof_value"].unique())
    df["dof_value"] = pd.Categorical(df["dof_value"], categories=dof_order, ordered=True)
    metric_cols = [c for c in TT_METRICS]

    for metric in metric_cols:
        plot_df = df[["dof_value", metric]].dropna().copy()

        if downsample is not None and downsample > 0:
            sampled = []
            for d in dof_order:
                sub = plot_df[plot_df["dof_value"] == d]
                if len(sub) > downsample:
                    sampled.append(sub.sample(n=downsample, random_state=2025))
                else:
                    sampled.append(sub)
            plot_df = pd.concat(sampled, ignore_index=True)

        plt.figure(figsize=(7.6, 5.4))
        ax = sns.violinplot(
            data=plot_df,
            x="dof_value",
            y=metric,
            inner=None,
            cut=0
        )
        sns.boxplot(
            data=plot_df,
            x="dof_value",
            y=metric,
            showfliers=False,
            whis=1.5,
            width=0.25
        )
        sns.stripplot(
            data=plot_df,
            x="dof_value",
            y=metric,
            size=2.0,
            alpha=0.35,
            jitter=0.25
        )
        ax.set_xlabel("DoF")
        ax.set_title(f"{metric} by DoF — Violin + Box + Points")
        plt.tight_layout()
        out_path = out_dir / f"violin_{metric}.png"
        plt.savefig(out_path, dpi=220)
        plt.close()

def plot_ecdf(df: pd.DataFrame, out_dir: Path):
    sns.set(style="whitegrid")
    dof_order = sorted(df["dof_value"].unique())
    df["dof_value"] = pd.Categorical(df["dof_value"], categories=dof_order, ordered=True)
    metric_cols = [c for c in TT_METRICS]

    for metric in metric_cols:
        plot_df = df[["dof_value", metric]].dropna().copy()
        plt.figure(figsize=(7.6, 5.4))
        for d in dof_order:
            sub = plot_df[plot_df["dof_value"] == d][metric].values
            if len(sub) == 0:
                continue
            sns.ecdfplot(x=sub, label=f"DoF {d}")
        plt.legend(title="DoF")
        plt.xlabel(metric)
        plt.ylabel("ECDF")
        plt.title(f"{metric} — ECDF by DoF")
        plt.tight_layout()
        out_path = out_dir / f"ecdf_{metric}.png"
        plt.savefig(out_path, dpi=220)
        plt.close()

def plot_mean_ci(df: pd.DataFrame, out_dir: Path, ci: int = 95):
    """DoF별 평균과 신뢰구간(부트스트랩) 라인 플롯."""
    sns.set(style="whitegrid")
    dof_order = sorted(df["dof_value"].unique())
    df["dof_value"] = pd.Categorical(df["dof_value"], categories=dof_order, ordered=True)
    metric_cols = [c for c in TT_METRICS]

    for metric in metric_cols:
        plot_df = df[["dof_value", metric]].dropna().copy()
        plt.figure(figsize=(7.6, 5.4))
        ax = sns.pointplot(
            data=plot_df,
            x="dof_value",
            y=metric,
            errorbar=("ci", ci),
            capsize=.15,
            dodge=True
        )
        ax.set_xlabel("DoF")
        ax.set_title(f"{metric} — Mean ± {ci}% CI by DoF")
        plt.tight_layout()
        out_path = out_dir / f"mean_ci_{metric}.png"
        plt.savefig(out_path, dpi=220)
        plt.close()

def main():
    ap = argparse.ArgumentParser(description="Top-tier metric evaluation (MTLD, HDD, Entropy, Perplexity, Distinct-n, MSL) with plots.")
    ap.add_argument("--input-dir", required=True, help="Directory with .jsonl files")
    ap.add_argument("--out-dir", default=None, help="Output directory for plots/CSVs")
    ap.add_argument("--plots", choices=["violin", "ecdf", "meanci", "all"], default="all",
                    help="Which plots to generate")
    ap.add_argument("--downsample", type=int, default=1000, help="Max points per DoF for scatter overlay")
    ap.add_argument("--ppl-model", default=None,
                    help="HF model name for perplexity (e.g., gpt2). If omitted or unavailable, uses unigram PPL.")
    ap.add_argument("--ppl-max-len", type=int, default=512, help="Max tokens for HF model perplexity")
    ap.add_argument("--unigram-alpha", type=float, default=1.0, help="Laplace smoothing alpha for unigram PPL fallback")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (input_dir / "eval_TT_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading jsonl rows...")
    df = load_jsonl_rows(input_dir)
    raw_csv = out_dir / "raw_success_rows.csv"
    df.to_csv(raw_csv, index=False, encoding="utf-8")
    print(f"Saved raw rows: {raw_csv}")

    print("Computing top-tier metrics per sample...")
    df_metrics = compute_metrics_per_sample(
        df,
        ppl_model_name=args.ppl_model,
        ppl_max_len=args.ppl_max_len,
        unigram_alpha=args.unigram_alpha,
        show_progress=True
    )
    per_sample_csv = out_dir / "per_sample_metrics_TT.csv"
    df_metrics.to_csv(per_sample_csv, index=False, encoding="utf-8")
    print(f"Saved per-sample metrics: {per_sample_csv}")

    print("Plotting...")
    if args.plots in ("violin", "all"):
        plot_violin_box_points(df_metrics, out_dir, downsample=args.downsample)
    if args.plots in ("ecdf", "all"):
        plot_ecdf(df_metrics, out_dir)
    if args.plots in ("meanci", "all"):
        plot_mean_ci(df_metrics, out_dir)

    print(f"All done. Outputs saved to: {out_dir}")

if __name__ == "__main__":
    main()
