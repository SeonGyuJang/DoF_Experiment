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

# BERTScore 관련 import
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "bert-score"])
        from bert_score import score as bert_score
        BERTSCORE_AVAILABLE = True
        print("Successfully installed and imported bert-score")
    except:
        BERTSCORE_AVAILABLE = False
        print("Warning: BERTScore not available. Installing bert-score...")

def word_tokens(text: str) -> List[str]:
    """텍스트를 단어 토큰으로 분리"""
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
    """텍스트를 문장으로 분리"""
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
    """n-gram 생성"""
    if n <= 0:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def mtld(tokens: List[str], ttr_threshold: float = 0.72, min_seg: int = 10) -> float:
    """MTLD (Measure of Textual Lexical Diversity) 계산"""
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
    return (_mtld_one(tokens) + _mtld_one(list(reversed(tokens)))) / 2.0

def mean_sentence_length(text: str) -> float:
    """평균 문장 길이 계산"""
    sents = sentence_split(text)
    if not sents:
        return float(len(word_tokens(text)))
    lengths = [len(word_tokens(s)) for s in sents]
    return float(np.mean(lengths)) if lengths else 0.0

def load_data_from_document(doc_content: str) -> pd.DataFrame:
    """문서에서 데이터를 로드하여 DataFrame으로 변환"""
    lines = doc_content.strip().split('\n')
    rows = []
    
    current_section = None
    for line in lines:
        line = line.strip()
        if line.startswith('< ') and line.endswith(' >'):
            current_section = line[2:-2].strip()
            continue
        
        if line.startswith('{') and line.endswith('}'):
            try:
                obj = json.loads(line)
                if obj.get("error") not in (None, ""):
                    continue
                cont = (obj.get("continuation") or "").strip()
                if not cont:
                    continue
                
                # 섹션별로 method 결정
                if current_section == "dof":
                    method = "DoF"
                elif current_section == "zero-shot":
                    method = "Zero-shot"
                elif current_section == "few-shot":
                    method = "Few-shot"
                else:
                    continue
                
                rows.append({
                    "sample_id": int(obj.get("sample_id", -1)),
                    "method": method,
                    "continuation": cont,
                    "prompt_name": obj.get("prompt_name", "")
                })
            except Exception as e:
                print(f"Error parsing line: {line[:100]}... Error: {e}")
                continue
    
    if not rows:
        raise RuntimeError("No valid rows found in document.")
    
    return pd.DataFrame(rows)

def load_jsonl_from_directory(input_dir: Path) -> pd.DataFrame:
    """디렉토리에서 JSONL 파일들을 로드"""
    files = sorted([p for p in input_dir.glob("*.jsonl") if p.is_file()])
    rows = []
    
    for file_path in files:
        print(f"Processing file: {file_path.name}")
        
        # 파일명에서 method 추출
        filename = file_path.stem.lower()
        if "dof" in filename:
            method = "DoF"
        elif "zero" in filename or "0shot" in filename:
            method = "Zero-shot"  
        elif "few" in filename or "3shot" in filename:
            method = "Few-shot"
        else:
            print(f"Unknown method for file: {file_path.name}")
            continue
            
        with file_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print(f"JSON parse error in {file_path.name}:{line_num}: {e}")
                    continue
                    
                if obj.get("error") not in (None, ""):
                    continue
                    
                cont = (obj.get("continuation") or "").strip()
                if not cont:
                    continue
                    
                rows.append({
                    "sample_id": int(obj.get("sample_id", obj.get("index", -1))),
                    "method": method,
                    "continuation": cont,
                    "prompt_name": obj.get("prompt_name", ""),
                    "dof_value": obj.get("dof_value", None)
                })
    
    if not rows:
        raise RuntimeError("No valid rows found in directory.")
    
    return pd.DataFrame(rows)

def compute_bertscore(texts: List[str], references: List[str] = None, batch_size: int = 64) -> List[float]:
    """BERTScore F1 점수 계산 (배치 처리)"""
    if not BERTSCORE_AVAILABLE:
        print("BERTScore not available, returning NaN values")
        return [np.nan] * len(texts)
    
    try:
        # 참조 텍스트가 없으면 첫 번째 텍스트를 참조로 사용
        if references is None:
            references = [texts[0]] * len(texts)
        
        print(f"  Computing BERTScore for {len(texts)} samples...")
        
        # 배치로 처리
        all_scores = []
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(texts), batch_size), 
                     desc="  BERTScore batches", 
                     total=num_batches,
                     leave=False):
            batch_texts = texts[i:i+batch_size]
            batch_refs = references[i:i+batch_size]
            
            # BERTScore 계산 (F1 점수 사용)
            P, R, F1 = bert_score(batch_texts, batch_refs, lang="en", verbose=False)
            all_scores.extend(F1.tolist())
        
        print(f"  ✓ BERTScore computation completed")
        return all_scores
        
    except Exception as e:
        print(f"  ✗ Error computing BERTScore: {e}")
        return [np.nan] * len(texts)

def compute_metrics_per_sample(
    df: pd.DataFrame,
    show_progress: bool = True,
    use_bertscore: bool = True
) -> pd.DataFrame:
    """각 샘플별로 메트릭 계산"""
    df = df.copy()
    
    # 메트릭 초기화
    metrics = {"MTLD": [], "MSL": []}
    if use_bertscore and BERTSCORE_AVAILABLE:
        metrics["BERTScore"] = []
    
    iterator = tqdm(df.iterrows(), total=len(df), desc="Computing metrics") if show_progress else df.iterrows()
    
    # 각 샘플별 메트릭 계산
    for _, row in iterator:
        txt = row["continuation"]
        toks = word_tokens(txt)
        
        # MTLD
        try:
            metrics["MTLD"].append(mtld(toks))
        except Exception:
            metrics["MTLD"].append(np.nan)
        
        # MSL
        try:
            metrics["MSL"].append(mean_sentence_length(txt))
        except Exception:
            metrics["MSL"].append(np.nan)
    
    # BERTScore 계산 (배치로 처리)
    if use_bertscore and BERTSCORE_AVAILABLE:
        print("\n📊 Computing BERTScore...")
        print("ℹ️  Note: Using each method's first sample as reference")
        try:
            # 각 method별로 첫 번째 텍스트를 참조로 사용
            method_groups = df.groupby("method")
            bertscore_values = []
            
            total_methods = len(method_groups)
            for method_idx, (method, group) in enumerate(method_groups, 1):
                group_texts = group["continuation"].tolist()
                if len(group_texts) > 0:
                    print(f"\n[{method_idx}/{total_methods}] Processing {method} method:")
                    print(f"  📝 {len(group_texts):,} samples to process")
                    
                    # 해당 method의 첫 번째 텍스트를 참조로 사용
                    reference = group_texts[0]
                    references = [reference] * len(group_texts)
                    
                    scores = compute_bertscore(group_texts, references)
                    bertscore_values.extend(scores)
                    
                    # 간단한 통계 출력
                    valid_scores = [s for s in scores if not np.isnan(s)]
                    if valid_scores:
                        print(f"  📈 Mean BERTScore: {np.mean(valid_scores):.4f}")
                        print(f"  📊 Score range: {np.min(valid_scores):.4f} - {np.max(valid_scores):.4f}")
                    
            metrics["BERTScore"] = bertscore_values
            print(f"\n✅ BERTScore computation completed for all methods!")
            
        except Exception as e:
            print(f"\n❌ Error computing BERTScore: {e}")
            metrics["BERTScore"] = [np.nan] * len(df)
    elif use_bertscore and not BERTSCORE_AVAILABLE:
        print("\n⚠️  BERTScore requested but not available")
        print("   Try installing with: pip install bert-score")
    else:
        print("\n⏭️  BERTScore computation skipped (--no-bertscore flag)")
    
    # DataFrame에 메트릭 추가
    for metric_name, values in metrics.items():
        df[metric_name] = values
    
    return df

def plot_ecdf_comparison(df: pd.DataFrame, out_dir: Path, palette: Optional[List[str]] = None):
    """Method별 ECDF 비교 플롯 생성"""
    sns.set_style("whitegrid")
    
    methods = sorted(df["method"].unique())
    if palette is None:
        palette = sns.color_palette("Set2", n_colors=len(methods))
    
    # 사용할 메트릭들
    metrics = ["MTLD", "MSL"]
    if "BERTScore" in df.columns:
        metrics.append("BERTScore")
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        plot_df = df[["method", metric]].dropna().copy()
        if len(plot_df) == 0:
            continue
            
        for color, method in zip(palette, methods):
            method_data = plot_df[plot_df["method"] == method][metric].values
            if len(method_data) == 0:
                continue
            
            # ECDF 플롯
            sns.ecdfplot(data=method_data, label=method, color=color, linewidth=2)
        
        plt.legend(title="Method", fontsize=12)
        plt.xlabel(metric, fontsize=12)
        plt.ylabel("Cumulative Probability", fontsize=12)
        plt.title(f"{metric} Distribution Comparison", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 저장
        out_path = out_dir / f"ecdf_comparison_{metric}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

def generate_summary_stats(df: pd.DataFrame, out_dir: Path):
    """요약 통계 생성 및 저장"""
    metrics = ["MTLD", "MSL"]
    if "BERTScore" in df.columns:
        metrics.append("BERTScore")
    
    summary_stats = []
    
    for metric in metrics:
        for method in df["method"].unique():
            method_data = df[df["method"] == method][metric].dropna()
            if len(method_data) == 0:
                continue
                
            stats = {
                "Method": method,
                "Metric": metric,
                "Count": len(method_data),
                "Mean": method_data.mean(),
                "Std": method_data.std(),
                "Min": method_data.min(),
                "25%": method_data.quantile(0.25),
                "50%": method_data.quantile(0.50),
                "75%": method_data.quantile(0.75),
                "Max": method_data.max()
            }
            summary_stats.append(stats)
    
    stats_df = pd.DataFrame(summary_stats)
    
    # CSV 저장
    stats_path = out_dir / "summary_statistics.csv"
    stats_df.to_csv(stats_path, index=False, encoding="utf-8")
    print(f"Summary statistics saved: {stats_path}")
    
    # 콘솔에 출력
    print("\n=== Summary Statistics ===")
    for metric in metrics:
        print(f"\n{metric}:")
        metric_stats = stats_df[stats_df["Metric"] == metric]
        for _, row in metric_stats.iterrows():
            print(f"  {row['Method']}: Mean={row['Mean']:.3f}, Std={row['Std']:.3f}")

def create_combined_plot(df: pd.DataFrame, out_dir: Path):
    """모든 메트릭을 한 번에 보여주는 조합 플롯"""
    metrics = ["MTLD", "MSL"]
    if "BERTScore" in df.columns:
        metrics.append("BERTScore")
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    
    methods = sorted(df["method"].unique())
    palette = sns.color_palette("Set2", n_colors=len(methods))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        plot_df = df[["method", metric]].dropna().copy()
        
        for color, method in zip(palette, methods):
            method_data = plot_df[plot_df["method"] == method][metric].values
            if len(method_data) == 0:
                continue
            
            sns.ecdfplot(data=method_data, label=method, color=color, linewidth=2, ax=ax)
        
        ax.set_xlabel(metric, fontsize=12)
        ax.set_ylabel("Cumulative Probability", fontsize=12)
        ax.set_title(f"{metric}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(title="Method", fontsize=10)
    
    plt.tight_layout()
    combined_path = out_dir / "combined_ecdf_comparison.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined plot saved: {combined_path}")

def main():
    ap = argparse.ArgumentParser(description="Compare DoF, Zero-shot, and Few-shot performance using MTLD, MSL, and BERTScore metrics")
    ap.add_argument("--input-dir", required=True, help="Directory containing JSONL files")
    ap.add_argument("--out-dir", default=None, help="Output directory for plots and results")
    ap.add_argument("--no-bertscore", action="store_true", help="Skip BERTScore computation")
    args = ap.parse_args()
    
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (input_dir / "comparison_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    try:
        df = load_jsonl_from_directory(input_dir)
        print(f"Loaded {len(df)} samples")
        print(f"Methods found: {df['method'].unique().tolist()}")
        print(f"Sample counts per method: {df['method'].value_counts().to_dict()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # 원본 데이터 저장
    raw_csv = out_dir / "raw_data.csv"
    df.to_csv(raw_csv, index=False, encoding="utf-8")
    print(f"Raw data saved: {raw_csv}")
    
    # 메트릭 계산
    print("\n🔄 Computing metrics...")
    print("📊 Calculating MTLD (lexical diversity) and MSL (sentence length)...")
    use_bertscore = not args.no_bertscore
    df_metrics = compute_metrics_per_sample(df, show_progress=True, use_bertscore=use_bertscore)
    
    # 메트릭 결과 저장
    metrics_csv = out_dir / "metrics_results.csv"
    df_metrics.to_csv(metrics_csv, index=False, encoding="utf-8")
    print(f"Metrics results saved: {metrics_csv}")
    
    # ECDF 플롯 생성
    print("\n📈 Generating ECDF plots...")
    plot_ecdf_comparison(df_metrics, out_dir)
    
    # 조합 플롯 생성
    print("\n🎨 Generating combined plot...")
    create_combined_plot(df_metrics, out_dir)
    
    # 요약 통계 생성
    print("\n📋 Generating summary statistics...")
    generate_summary_stats(df_metrics, out_dir)
    
    print(f"\n🎉 All results saved to: {out_dir}")
    print("\n📁 Generated files:")
    print(f"   📄 raw_data.csv: Original data")
    print(f"   📊 metrics_results.csv: Data with computed metrics")
    print(f"   📈 summary_statistics.csv: Statistical summary")
    print(f"   🖼️  ecdf_comparison_*.png: Individual metric comparisons")
    print(f"   🎨 combined_ecdf_comparison.png: All metrics in one plot")
    
    if not args.no_bertscore and BERTSCORE_AVAILABLE:
        print(f"\n✅ BERTScore metric included in analysis")
    elif args.no_bertscore:
        print(f"\n⏭️  BERTScore computation skipped (--no-bertscore flag)")
    else:
        print(f"\n❌ BERTScore not available (install with: pip install bert-score)")

if __name__ == "__main__":
    main()