import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

BASE = Path(__file__).resolve().parent
RESULTS_BASE = Path(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\results\gemini\gemini-2.0-flash\essay")
JUDGE_RESULTS_BASE = Path(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\results\judge")

SUPPORTED_METHODS = ["dof", "zero_shot", "few_shot", "simple_instruction"]
METHOD_FRIENDLY = {
    "dof": "DoF",
    "zero_shot": "Zero-shot",
    "few_shot": "Few-shot",
    "simple_instruction": "Simple-Instruction",
}

JUDGE_PROMPT_2 = """You are an expert evaluator of creative writing. Evaluate and compare TWO essays (A and B) on THREE SEPARATE criteria: creativity, lexical diversity, and coherence.

Criteria definitions:
1. Creativity: Novel ideas, unique perspectives, imaginative approaches, innovative use of language and concepts
2. Lexical Diversity: Varied vocabulary, richness of word choice, avoidance of repetition, sophisticated and diverse use of language
3. Coherence: Logical flow, clear organization, consistent argument development, overall structural integrity

Essays:
Essay A:
{essay_a}

Essay B:
{essay_b}

Instructions:
- For EACH criterion (creativity, lexical_diversity, coherence), choose a ranking (1st, 2nd) between A and B.
- Provide scores (1-10) for each essay on each criterion.
- Provide brief reasoning for each criterion's ranking.

Output (JSON):
{{
  "creativity": {{
    "ranking": {{"1st": "A" or "B", "2nd": "A" or "B"}},
    "scores": {{"A": 1-10, "B": 1-10}},
    "reasoning": "Brief explanation"
  }},
  "lexical_diversity": {{
    "ranking": {{"1st": "A" or "B", "2nd": "A" or "B"}},
    "scores": {{"A": 1-10, "B": 1-10}},
    "reasoning": "Brief explanation"
  }},
  "coherence": {{
    "ranking": {{"1st": "A" or "B", "2nd": "A" or "B"}},
    "scores": {{"A": 1-10, "B": 1-10}},
    "reasoning": "Brief explanation"
  }}
}}
"""

SINGLE_JUDGE_PROMPT = """You are an expert evaluator of creative writing. Evaluate ONE essay on THREE criteria: creativity, lexical diversity, and coherence.

Criteria definitions:
1. Creativity: Novel ideas, unique perspectives, imaginative approaches, innovative use of language and concepts
2. Lexical Diversity: Varied vocabulary, richness of word choice, avoidance of repetition, sophisticated and diverse use of language
3. Coherence: Logical flow, clear organization, consistent argument development, overall structural integrity

Essay:
{essay}

Instructions:
- For EACH criterion, provide a score from 1 to 10.
- Provide one or two sentences of reasoning per criterion.

Output (JSON):
{{
  "creativity": {{"score": 1-10, "reasoning": "brief"}} ,
  "lexical_diversity": {{"score": 1-10, "reasoning": "brief"}} ,
  "coherence": {{"score": 1-10, "reasoning": "brief"}}
}}
"""

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def init_env():
    loaded = load_dotenv(dotenv_path=find_dotenv(usecwd=True))
    if not loaded:
        load_dotenv(BASE / ".env")
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            f"GOOGLE_API_KEY not found. cwd={os.getcwd()} base={BASE}"
        )

def initialize_judge_llm(model_name: str = "gemini-2.0-flash"):
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3,
        max_output_tokens=3072
    )

def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def load_jsonl_results(file_path: Path) -> List[Dict[str, Any]]:
    results = []
    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return results
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if data.get('continuation') and str(data.get('continuation')).strip():
                        results.append(data)
                except json.JSONDecodeError:
                    continue
    return results

def _has_prompt_tag(s: str) -> bool:
    return any(tag in s for tag in ("prompt1", "prompt7", "prompt8"))

def find_result_files(results_dir: Path) -> Dict[str, Path]:
    buckets = {k: [] for k in SUPPORTED_METHODS}
    for p in results_dir.glob("*.jsonl"):
        name = p.name.lower()
        if not _has_prompt_tag(name):
            continue
        if "dof" in name:
            buckets["dof"].append(p)
        if "zero_shot" in name:
            buckets["zero_shot"].append(p)
        if "few_shot" in name:
            buckets["few_shot"].append(p)
        if ("simple" in name and "instruction" in name):
            buckets["simple_instruction"].append(p)
    files = {}
    for k, arr in buckets.items():
        if arr:
            files[k] = max(arr, key=lambda x: x.stat().st_mtime)
    return files

def create_chain(llm, prompt_template: str):
    prompt = PromptTemplate.from_template(prompt_template)
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    return chain

def validate_pairwise_result(result: Dict[str, Any]) -> bool:
    try:
        for criterion in ['creativity', 'lexical_diversity', 'coherence']:
            if criterion not in result:
                return False
            cd = result[criterion]
            ranking = cd.get("ranking", {})
            if not all(k in ranking for k in ["1st", "2nd"]):
                return False
            ranked = {ranking["1st"], ranking["2nd"]}
            if ranked != {"A", "B"}:
                return False
            scores = cd.get("scores", {})
            if not all(k in scores for k in ["A", "B"]):
                return False
            if not (isinstance(scores["A"], (int, float)) and isinstance(scores["B"], (int, float))):
                return False
        return True
    except Exception:
        return False

def validate_single_result(result: Dict[str, Any]) -> bool:
    try:
        for criterion in ['creativity', 'lexical_diversity', 'coherence']:
            if criterion not in result:
                return False
            cd = result[criterion]
            if "score" not in cd:
                return False
            if not isinstance(cd["score"], (int, float)):
                return False
        return True
    except Exception:
        return False

def judge_pair_single(idx, essay_a, essay_b, model_name: str, max_retries=3):
    try:
        llm = initialize_judge_llm(model_name)
        chain = create_chain(llm, JUDGE_PROMPT_2)
        for attempt in range(max_retries):
            try:
                out = chain.invoke({"essay_a": essay_a, "essay_b": essay_b})
                if validate_pairwise_result(out):
                    return {"idx": idx, "result": out, "success": True}
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"idx": idx, "success": False, "error": str(e)}
                time.sleep(0.5)
        return {"idx": idx, "success": False, "error": "Max retries exceeded"}
    except Exception as e:
        return {"idx": idx, "success": False, "error": str(e)}

def judge_pair_batch(tasks: List[Tuple[int, str, str]], model_name: str, batch_concurrency: int, max_retries=1):
    """
    tasks: list of (idx, essay_a, essay_b)
    returns: list of dicts like judge_pair_single
    """
    llm = initialize_judge_llm(model_name)
    chain = create_chain(llm, JUDGE_PROMPT_2)

    inputs = [{"essay_a": a, "essay_b": b} for _, a, b in tasks]
    idxs = [i for i, _, _ in tasks]

    results: List[Dict[str, Any]] = []
    for attempt in range(max_retries):
        try:
            outs = chain.batch(inputs, config={"max_concurrency": batch_concurrency})
            for i, out in zip(idxs, outs):
                if validate_pairwise_result(out):
                    results.append({"idx": i, "result": out, "success": True})
                else:
                    results.append({"idx": i, "success": False, "error": "Validation failed"})
            return results
        except Exception as e:
            if attempt == max_retries - 1:
                for i in idxs:
                    results.append({"idx": i, "success": False, "error": f"Batch error: {e}"})
                return results
            time.sleep(0.5)

def judge_single_absolute(method: str, idx: int, essay: str, model_name: str, max_retries=3):
    try:
        llm = initialize_judge_llm(model_name)
        chain = create_chain(llm, SINGLE_JUDGE_PROMPT)
        for attempt in range(max_retries):
            try:
                out = chain.invoke({"essay": essay})
                if validate_single_result(out):
                    return {"method": method, "idx": idx, "result": out, "success": True}
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"method": method, "idx": idx, "success": False, "error": str(e)}
                time.sleep(0.4)
        return {"method": method, "idx": idx, "success": False, "error": "Max retries exceeded"}
    except Exception as e:
        return {"method": method, "idx": idx, "success": False, "error": str(e)}

def judge_single_batch(tasks: List[Tuple[str, int, str]], model_name: str, batch_concurrency: int, max_retries=1):
    """
    tasks: list of (method, idx, essay)
    returns: list of dicts like judge_single_absolute
    """
    llm = initialize_judge_llm(model_name)
    chain = create_chain(llm, SINGLE_JUDGE_PROMPT)

    inputs = [{"essay": e} for _, _, e in tasks]
    meta = [(m, i) for m, i, _ in tasks]

    results: List[Dict[str, Any]] = []
    for attempt in range(max_retries):
        try:
            outs = chain.batch(inputs, config={"max_concurrency": batch_concurrency})
            for (m, i), out in zip(meta, outs):
                if validate_single_result(out):
                    results.append({"method": m, "idx": i, "result": out, "success": True})
                else:
                    results.append({"method": m, "idx": i, "success": False, "error": "Validation failed"})
            return results
        except Exception as e:
            if attempt == max_retries - 1:
                for (m, i) in meta:
                    results.append({"method": m, "idx": i, "success": False, "error": f"Batch error: {e}"})
                return results
            time.sleep(0.5)

def run_pairwise(
    results_dir: Path,
    n_comparisons: int,
    seed: int,
    model_name: str,
    threads: int,
    output_dir: Path,
    use_batch: bool = False,
    batch_size: int = 20,
    batch_concurrency: int = 8,
) -> Dict[str, Any]:

    files = find_result_files(results_dir)
    print("Found files:")
    for k, v in files.items():
        print(f"  {k}: {v}")

    required = {"dof", "simple_instruction"}
    missing = required - set(files.keys())
    if missing:
        raise ValueError(f"Missing result files for methods: {missing}")

    print("\nLoading essays...")
    dof = load_jsonl_results(files["dof"])
    simp = load_jsonl_results(files["simple_instruction"])
    min_len = min(len(dof), len(simp))
    if min_len < n_comparisons:
        print(f"Only {min_len} paired essays available. Reducing comparisons to {min_len}.")
        n_comparisons = min_len

    random.seed(seed)
    indices = random.sample(range(min_len), n_comparisons)

    tasks = []
    for i, idx in enumerate(indices):
        a = dof[idx]["continuation"]
        b = simp[idx]["continuation"]
        tasks.append((i, a, b))

    print(f"Running pairwise DoF(A) vs Simple(B): {n_comparisons} comparisons ...")
    start = time.time()
    results = []

    if use_batch:
        for chunk in tqdm(list(chunk_list(tasks, batch_size)), total=int(np.ceil(len(tasks)/batch_size)), desc="Judging(Batch)"):
            batch_outs = judge_pair_batch(chunk, model_name=model_name, batch_concurrency=batch_concurrency, max_retries=1)
            results.extend(batch_outs)
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            fut2task = {ex.submit(judge_pair_single, i, a, b, model_name, 3): (i, a, b) for (i, a, b) in tasks}
            for fut in tqdm(as_completed(fut2task), total=len(tasks), desc="Judging"):
                try:
                    results.append(fut.result(timeout=180))
                except Exception as e:
                    i, _, _ = fut2task[fut]
                    results.append({"idx": i, "success": False, "error": f"Thread timeout/error: {e}"})

    end = time.time()

    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]
    print(f"Success: {len(successes)}, Fail: {len(failures)}, Time: {end-start:.2f}s")

    criteria = ['creativity', 'lexical_diversity', 'coherence']
    stats = {}
    for c in criteria:
        stats[c] = {
            "rankings_count": {"A": {"1st": 0, "2nd": 0}, "B": {"1st": 0, "2nd": 0}},
            "avg_rank_score": {"A": 0.0, "B": 0.0}, 
            "detail_scores": {"A": [], "B": []},
            "avg_detail_scores": {"A": 0.0, "B": 0.0},
        }

    for r in successes:
        out = r["result"]
        for c in criteria:
            cd = out[c]
            first = cd["ranking"]["1st"]
            second = cd["ranking"]["2nd"]
            stats[c]["rankings_count"][first]["1st"] += 1
            stats[c]["rankings_count"][second]["2nd"] += 1
            stats[c]["avg_rank_score"][first] += 2
            stats[c]["avg_rank_score"][second] += 1
            stats[c]["detail_scores"]["A"].append(cd["scores"]["A"])
            stats[c]["detail_scores"]["B"].append(cd["scores"]["B"])

    n = len(successes)
    for c in criteria:
        if n > 0:
            stats[c]["avg_rank_score"]["A"] /= n
            stats[c]["avg_rank_score"]["B"] /= n
        stats[c]["avg_detail_scores"]["A"] = float(np.mean(stats[c]["detail_scores"]["A"])) if stats[c]["detail_scores"]["A"] else 0.0
        stats[c]["avg_detail_scores"]["B"] = float(np.mean(stats[c]["detail_scores"]["B"])) if stats[c]["detail_scores"]["B"] else 0.0

    summary = {
        "mode": "pairwise",
        "total_comparisons": n,
        "failed": len(failures),
        "performance": {
            "total_time": end - start,
            "comparisons_per_second": (len(successes) / (end - start)) if (end - start) > 0 else 0.0,
            "success_rate": (len(successes) / len(results) * 100) if results else 0.0
        },
        "criteria_stats": stats
    }

    ensure_dir(output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"judge_pairwise_dof_vs_simple_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")
    return summary

def viz_pairwise(summary: Dict[str, Any], out_dir: Path):
    ensure_dir(out_dir)
    criteria = ['creativity', 'lexical_diversity', 'coherence']
    labels = ['Creativity', 'Lexical Diversity', 'Coherence']

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    for i, (c, lab) in enumerate(zip(criteria, labels)):
        s = summary["criteria_stats"][c]
        ax1 = axes[i, 0]
        methods = ["A (DoF)", "B (Simple)"]
        firsts = [s["rankings_count"]["A"]["1st"], s["rankings_count"]["B"]["1st"]]
        seconds = [s["rankings_count"]["A"]["2nd"], s["rankings_count"]["B"]["2nd"]]
        x = np.arange(len(methods))
        width = 0.6
        ax1.bar(x, firsts, width, label='1st (2 pts)', alpha=0.85)
        ax1.bar(x, seconds, width, bottom=firsts, label='2nd (1 pt)', alpha=0.85)
        ax1.set_title(f"{lab}: Ranking Distribution")
        ax1.set_xticks(x); ax1.set_xticklabels(methods); ax1.grid(True, alpha=0.3)
        if i == 0: ax1.legend()

        ax2 = axes[i, 1]
        avg_rank = [s["avg_rank_score"]["A"], s["avg_rank_score"]["B"]]
        bars = ax2.bar(methods, avg_rank, alpha=0.85)
        ax2.set_ylim(0, 2.0)
        ax2.set_title(f"{lab}: Avg Ranking Score (max 2)")
        ax2.grid(True, alpha=0.3)
        for b, score in zip(bars, avg_rank):
            ax2.text(b.get_x() + b.get_width()/2, b.get_height()+0.03, f"{score:.2f}", ha="center")

        ax3 = axes[i, 2]
        avg_det = [s["avg_detail_scores"]["A"], s["avg_detail_scores"]["B"]]
        bars = ax3.bar(methods, avg_det, alpha=0.85)
        ax3.set_ylim(0, 10)
        ax3.set_title(f"{lab}: Avg Detail Scores (1-10)")
        ax3.grid(True, alpha=0.3)
        for b, score in zip(bars, avg_det):
            ax3.text(b.get_x() + b.get_width()/2, b.get_height()+0.1, f"{score:.1f}", ha="center")

    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"pairwise_dof_simple_{ts}.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Figure saved: {path}")

def print_pairwise_summary(summary: Dict[str, Any]):
    print("\n" + "="*80)
    print("PAIRWISE SUMMARY: DoF (A) vs Simple-Instruction (B)")
    print("="*80)
    perf = summary["performance"]
    print(f"Total comparisons: {summary['total_comparisons']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success rate: {perf['success_rate']:.1f}%")
    print(f"Speed: {perf['comparisons_per_second']:.2f} comps/s")
    print(f"Time: {perf['total_time']:.2f}s")

    print("\nPer-Criterion:")
    for c in ['creativity', 'lexical_diversity', 'coherence']:
        s = summary["criteria_stats"][c]
        print("-"*70)
        print(c.upper())
        print(f"1st counts - A(DoF): {s['rankings_count']['A']['1st']} | B(Simple): {s['rankings_count']['B']['1st']}")
        print(f"Avg Rank Score (max 2) - A: {s['avg_rank_score']['A']:.2f} | B: {s['avg_rank_score']['B']:.2f}")
        print(f"Avg Detail Score (1-10) - A: {s['avg_detail_scores']['A']:.1f} | B: {s['avg_detail_scores']['B']:.1f}")

def run_independent(
    results_dir: Path,
    methods: List[str],
    n_per_method: int,
    seed: int,
    model_name: str,
    threads: int,
    output_dir: Path,
    use_batch: bool = False,
    batch_size: int = 20,
    batch_concurrency: int = 8,
) -> Dict[str, Any]:

    files = find_result_files(results_dir)
    print("Found files:")
    for k, v in files.items():
        print(f"  {k}: {v}")

    unknown = [m for m in methods if m not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(f"Unsupported methods in --methods: {unknown}")

    missing = [m for m in methods if m not in files]
    if missing:
        raise ValueError(f"Missing result files for methods: {missing}")

    print("\nLoading essays...")
    pool: Dict[str, List[Dict[str, Any]]] = {}
    for m in methods:
        pool[m] = load_jsonl_results(files[m])
        print(f"  {m}: {len(pool[m])} essays")

    random.seed(seed)
    tasks = []
    for m in methods:
        total = len(pool[m])
        take = min(n_per_method, total)
        idxs = random.sample(range(total), take)
        for i in idxs:
            essay = pool[m][i]["continuation"]
            tasks.append((m, i, essay))

    print(f"Running independent absolute judging for methods={methods} | total tasks={len(tasks)}")
    start = time.time()
    results = []

    if use_batch:
        for chunk in tqdm(list(chunk_list(tasks, batch_size)), total=int(np.ceil(len(tasks)/batch_size)), desc="Judging(Batch)"):
            batch_outs = judge_single_batch(chunk, model_name=model_name, batch_concurrency=batch_concurrency, max_retries=1)
            results.extend(batch_outs)
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            fut2task = {ex.submit(judge_single_absolute, m, i, e, model_name, 3): (m, i, e) for (m, i, e) in tasks}
            for fut in tqdm(as_completed(fut2task), total=len(tasks), desc="Judging"):
                try:
                    results.append(fut.result(timeout=180))
                except Exception as e:
                    m, i, _ = fut2task[fut]
                    results.append({"method": m, "idx": i, "success": False, "error": f"Thread timeout/error: {e}"})

    end = time.time()

    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]
    print(f"Success: {len(successes)}, Fail: {len(failures)}, Time: {end-start:.2f}s")

    criteria = ['creativity', 'lexical_diversity', 'coherence']
    stats: Dict[str, Any] = {}
    raw_scores: Dict[str, Dict[str, List[float]]] = {}
    for m in methods:
        stats[m] = {
            "counts": {c: 0 for c in criteria},
            "scores": {c: [] for c in criteria},
            "avg_scores": {c: 0.0 for c in criteria},
        }
        raw_scores[m] = {c: [] for c in criteria}

    for r in successes:
        m = r["method"]
        out = r["result"]
        for c in criteria:
            sc = float(out[c]["score"])
            stats[m]["scores"][c].append(sc)
            raw_scores[m][c].append(sc)
            stats[m]["counts"][c] += 1

    for m in methods:
        for c in criteria:
            scores = stats[m]["scores"][c]
            stats[m]["avg_scores"][c] = float(np.mean(scores)) if scores else 0.0

    summary = {
        "mode": "independent",
        "methods": methods,
        "per_method_requested": n_per_method,
        "total_success": len(successes),
        "total_fail": len(failures),
        "performance": {
            "total_time": end - start,
            "items_per_second": (len(successes) / (end - start)) if (end - start) > 0 else 0.0,
            "success_rate": (len(successes) / len(results) * 100) if results else 0.0
        },
        "stats": stats,
        "raw_scores": raw_scores  
    }

    ensure_dir(output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"judge_independent_{'-'.join(methods)}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")
    return summary

def viz_independent(summary: Dict[str, Any], out_dir: Path):
    ensure_dir(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    criteria = ['creativity', 'lexical_diversity', 'coherence']
    labels = ['Creativity', 'Lexical Diversity', 'Coherence']

    methods = summary["methods"]
    raw_scores: Dict[str, Dict[str, List[float]]] = summary.get("raw_scores", {})

    for m in methods:
        scores_by_c = raw_scores.get(m, {})
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        fig.suptitle(f"{METHOD_FRIENDLY.get(m, m)}", fontsize=16)

        for i, (c, lab) in enumerate(zip(criteria, labels)):
            ax = axes[i]
            vals = scores_by_c.get(c, [])
            if len(vals) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(lab)
                ax.set_xlim(0, 10)
                continue

            ax.hist(vals, bins=20, range=(0, 10), alpha=0.85)
            mu = float(np.mean(vals))
            ax.axvline(mu, linestyle="--", linewidth=2)
            ax.set_xlim(0, 10)
            ax.set_title(f"{lab} (mean={mu:.2f})")
            ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        path = out_dir / f"independent_{m}_{ts}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Figure saved: {path}")

def print_independent_summary(summary: Dict[str, Any]):
    print("\n" + "="*80)
    print("INDEPENDENT (ABSOLUTE) SUMMARY")
    print("="*80)
    perf = summary["performance"]
    print(f"Methods: {summary['methods']}")
    print(f"Success: {summary['total_success']}, Fail: {summary['total_fail']}")
    print(f"Success rate: {perf['success_rate']:.1f}% | Speed: {perf['items_per_second']:.2f} items/s | Time: {perf['total_time']:.2f}s")
    print("\nAverage Scores per Method:")
    for m in summary["methods"]:
        avg = summary["stats"][m]["avg_scores"]
        print(f"- {METHOD_FRIENDLY[m]}: "
              f"Creativity {avg['creativity']:.1f}, "
              f"LexDiv {avg['lexical_diversity']:.1f}, "
              f"Coherence {avg['coherence']:.1f}")

def load_results_summary(summary_path: Path) -> Dict[str, Any]:
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge: Pairwise(DoF vs Simple) or Independent(absolute) evaluation.")
    parser.add_argument("--results-dir", type=str,
                        default=str(RESULTS_BASE),
                        help="Directory containing the essay results")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results/figures")

    parser.add_argument("--model-name", type=str, default="gemini-2.0-flash",
                        help="Judge model name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--threads", type=int, default=8, help="Thread pool size")

    parser.add_argument("--use-batch", action="store_true",
                        help="Use LCEL batch() for request batching")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Number of items per batch() call")
    parser.add_argument("--batch-concurrency", type=int, default=8,
                        help="max_concurrency for chain.batch")

    parser.add_argument("--mode", type=str, choices=["pairwise", "independent"], default="pairwise",
                        help="Evaluation mode")

    parser.add_argument("--n-comparisons", type=int, default=100,
                        help="[pairwise] Number of paired comparisons (DoF vs Simple)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization")

    parser.add_argument("--methods", type=str, default=None,
                        help="[independent] Comma-separated methods to evaluate independently (e.g., dof,zero_shot)")
    parser.add_argument("--n-per-method", type=int, default=100,
                        help="[independent] Number of essays per method to evaluate")

    parser.add_argument("--viz-from-summary", type=str, default=None,
                        help="Path to a saved summary JSON for visualization only")

    args = parser.parse_args()
    init_env()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else JUDGE_RESULTS_BASE
    ensure_dir(output_dir)

    if args.viz_from_summary:
        summary_path = Path(args.viz_from_summary)
        if not summary_path.exists():
            raise FileNotFoundError(f"--viz-from-summary not found: {summary_path}")
        summary = load_results_summary(summary_path)
        mode = summary.get("mode", "pairwise")
        if mode == "pairwise":
            print_pairwise_summary(summary)
            if not args.no_viz:
                viz_pairwise(summary, output_dir)
        else:
            print_independent_summary(summary)
            if not args.no_viz:
                viz_independent(summary, output_dir)
        return

    if args.mode == "pairwise":
        print("\n" + "="*60)
        print("PAIRWISE MODE: DoF (A) vs Simple-Instruction (B)")
        print("="*60)
        summary = run_pairwise(
            results_dir=results_dir,
            n_comparisons=args.n_comparisons,
            seed=args.seed,
            model_name=args.model_name,
            threads=args.threads,
            output_dir=output_dir,
            use_batch=args.use_batch,
            batch_size=args.batch_size,
            batch_concurrency=args.batch_concurrency,
        )
        print_pairwise_summary(summary)
        if not args.no_viz:
            viz_pairwise(summary, output_dir)

    else:  
        if not args.methods:
            raise ValueError("--methods is required for independent mode (e.g., --methods dof,zero_shot)")
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        print("\n" + "="*60)
        print(f"INDEPENDENT MODE: methods={methods}")
        print("="*60)
        summary = run_independent(
            results_dir=results_dir,
            methods=methods,
            n_per_method=args.n_per_method,
            seed=args.seed,
            model_name=args.model_name,
            threads=args.threads,
            output_dir=output_dir,
            use_batch=args.use_batch,
            batch_size=args.batch_size,
            batch_concurrency=args.batch_concurrency,
        )
        print_independent_summary(summary)
        if not args.no_viz:
            viz_independent(summary, output_dir)

if __name__ == "__main__":
    main()
