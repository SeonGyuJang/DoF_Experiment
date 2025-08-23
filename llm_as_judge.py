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
from multiprocessing import Pool, set_start_method, current_process
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

BASE = Path(__file__).resolve().parent
RESULTS_BASE = Path(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\results\gemini\gemini-2.0-flash\essay")
JUDGE_RESULTS_BASE = Path(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\results\judge")

JUDGE_PROMPT_4 = """You are an expert evaluator of creative writing. Your task is to evaluate and rank four essay responses on THREE SEPARATE criteria: creativity, lexical diversity, and coherence.

**Evaluation Criteria:**
1. **Creativity**: Novel ideas, unique perspectives, imaginative approaches, innovative use of language and concepts
2. **Lexical Diversity**: Varied vocabulary, richness of word choice, avoidance of repetition, sophisticated and diverse use of language
3. **Coherence**: Logical flow, clear organization, consistent argument development, overall structural integrity

**Essays to Evaluate:**

**Essay A**
{essay_dof}

**Essay B**
{essay_zero_shot}

**Essay C:**
{essay_few_shot}

**Essay D:**
{essay_simple}

**Instructions:**
- For EACH criterion (creativity, lexical_diversity, coherence), rank the four essays separately from 1st to 4th place.
- Each criterion should have its own independent ranking.
- Provide scores (1-10) for each essay on each criterion.
- Provide brief reasoning for each criterion's ranking.

**Output Format (JSON):**
{{
    "creativity": {{
        "ranking": {{
            "1st": "A" or "B" or "C" or "D",
            "2nd": "A" or "B" or "C" or "D",
            "3rd": "A" or "B" or "C" or "D",
            "4th": "A" or "B" or "C" or "D"
        }},
        "scores": {{
            "A": 1-10,
            "B": 1-10,
            "C": 1-10,
            "D": 1-10
        }},
        "reasoning": "Brief explanation for creativity ranking"
    }},
    "lexical_diversity": {{
        "ranking": {{
            "1st": "A" or "B" or "C" or "D",
            "2nd": "A" or "B" or "C" or "D",
            "3rd": "A" or "B" or "C" or "D",
            "4th": "A" or "B" or "C" or "D"
        }},
        "scores": {{
            "A": 1-10,
            "B": 1-10,
            "C": 1-10,
            "D": 1-10
        }},
        "reasoning": "Brief explanation for lexical diversity ranking"
    }},
    "coherence": {{
        "ranking": {{
            "1st": "A" or "B" or "C" or "D",
            "2nd": "A" or "B" or "C" or "D",
            "3rd": "A" or "B" or "C" or "D",
            "4th": "A" or "B" or "C" or "D"
        }},
        "scores": {{
            "A": 1-10,
            "B": 1-10,
            "C": 1-10,
            "D": 1-10
        }},
        "reasoning": "Brief explanation for coherence ranking"
    }}
}}"""

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def init_env():
    loaded = load_dotenv(dotenv_path=find_dotenv(usecwd=True))
    if not loaded:
        load_dotenv(BASE / ".env")
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            f"cwd={os.getcwd()} base={BASE}"
        )

def initialize_judge_llm(model_name: str = "gemini-2.0-flash"):
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3,
        max_output_tokens=3072
    )

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

def _name_has_any(s: str, keywords: List[str]) -> bool:
    return all(k in s for k in keywords)

def find_result_files(results_dir: Path) -> Dict[str, Path]:
    buckets = {"dof": [], "zero_shot": [], "few_shot": [], "simple_instruction": []}

    def has_prompt_tag(s: str) -> bool:
        return any(tag in s for tag in ("prompt1", "prompt7", "prompt8",))

    for p in results_dir.glob("*.jsonl"):
        name = p.name.lower()
        if "dof" in name and has_prompt_tag(name):
            buckets["dof"].append(p)
        elif "zero_shot" in name and has_prompt_tag(name):
            buckets["zero_shot"].append(p)
        elif "few_shot" in name and has_prompt_tag(name):
            buckets["few_shot"].append(p)
        elif ("simple" in name and "instruction" in name) and has_prompt_tag(name):
            buckets["simple_instruction"].append(p)

    files = {}
    for k, arr in buckets.items():
        if arr:
            files[k] = max(arr, key=lambda x: x.stat().st_mtime)  # 가장 최근 파일
    return files


def create_judge_chain(llm, prompt_template: str):
    prompt = PromptTemplate.from_template(prompt_template)
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    return chain, prompt

def judge_single_essay_quad_threaded(comparison_id, essay_idx, essay_dof, essay_zero_shot, essay_few_shot, essay_simple, model_name, max_retries=3):
    try:
        judge_llm = initialize_judge_llm(model_name)
        judge_chain, _ = create_judge_chain(judge_llm, JUDGE_PROMPT_4)
        for attempt in range(max_retries):
            try:
                result = judge_chain.invoke({
                    "essay_dof": essay_dof,
                    "essay_zero_shot": essay_zero_shot,
                    "essay_few_shot": essay_few_shot,
                    "essay_simple": essay_simple
                })
                if validate_judge_result(result):
                    return {
                        'comparison_id': comparison_id,
                        'essay_idx': essay_idx,
                        'result': result,
                        'success': True
                    }
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(0.5)
        return {
            'comparison_id': comparison_id,
            'essay_idx': essay_idx,
            'success': False,
            'error': 'Max retries exceeded'
        }
    except Exception as e:
        return {
            'comparison_id': comparison_id,
            'essay_idx': essay_idx,
            'success': False,
            'error': str(e)
        }

def validate_judge_result(result: Dict[str, Any]) -> bool:
    try:
        criteria = ['creativity', 'lexical_diversity', 'coherence']
        for criterion in criteria:
            if criterion not in result:
                return False
            criterion_data = result[criterion]
            ranking = criterion_data.get('ranking', {})
            if not all(pos in ranking for pos in ['1st', '2nd', '3rd', '4th']):
                return False
            essays = set(['A', 'B', 'C', 'D'])
            ranked_essays = set([ranking['1st'], ranking['2nd'], ranking['3rd'], ranking['4th']])
            if essays != ranked_essays:
                return False
            scores = criterion_data.get('scores', {})
            if not all(essay in scores for essay in ['A', 'B', 'C', 'D']):
                return False
        return True
    except:
        return False

def convert_rankings_to_scores(result: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    method_map = {'A': 'dof', 'B': 'zero_shot', 'C': 'few_shot', 'D': 'simple_instruction'}
    criteria = ['creativity', 'lexical_diversity', 'coherence']
    scores: Dict[str, Dict[str, int]] = {}
    for criterion in criteria:
        scores[criterion] = {'dof': 0, 'zero_shot': 0, 'few_shot': 0, 'simple_instruction': 0}
        ranking = result[criterion]['ranking']
        scores[criterion][method_map[ranking['1st']]] = 4
        scores[criterion][method_map[ranking['2nd']]] = 3
        scores[criterion][method_map[ranking['3rd']]] = 2
        scores[criterion][method_map[ranking['4th']]] = 1
    return scores

def run_judge_evaluation(
    results_dir: Path,
    n_comparisons: int = 100,
    model_name: str = "gemini-2.0-flash",
    seed: int = 42,
    output_dir: Optional[Path] = None,
    num_processes: int = 4,
    batch_size: int = 5,
    use_threading: bool = True,
    threads_per_process: int = 3
) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"LLM-as-a-Judge: Separate Evaluation of Creativity, Lexical Diversity & Coherence (4-way)")
    print(f"Judge Model: {model_name}")
    print(f"Number of comparisons: {n_comparisons}")
    print(f"Results directory: {results_dir}")
    print(f"Processes: {num_processes}")
    print(f"Batch size: {batch_size}")
    print(f"Use threading: {use_threading}")
    if use_threading:
        print(f"Threads per process: {threads_per_process}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")

    init_env()

    result_files = find_result_files(results_dir)
    print("Found result files:")
    for method, path in result_files.items():
        print(f"  {method}: {path}")

    required = {'dof', 'zero_shot', 'few_shot', 'simple_instruction'}
    missing = required - set(result_files.keys())
    if missing:
        raise ValueError(f"Missing result files for methods: {missing}")

    print("\nLoading results...")
    results: Dict[str, List[Dict[str, Any]]] = {}
    for method, path in result_files.items():
        results[method] = load_jsonl_results(path)
        print(f"  {method}: {len(results[method])} essays loaded")

    min_essays = min(len(results[m]) for m in required)
    if min_essays < n_comparisons:
        print(f"Warning: Only {min_essays} essays available across all methods, reducing comparisons to {min_essays}")
        n_comparisons = min_essays

    random.seed(seed)
    indices = random.sample(range(min_essays), n_comparisons)

    print("Preparing evaluation tasks...")
    tasks = []
    for i, idx in enumerate(indices):
        essay_dof = results['dof'][idx]['continuation']
        essay_zero_shot = results['zero_shot'][idx]['continuation']
        essay_few_shot = results['few_shot'][idx]['continuation']
        essay_simple = results['simple_instruction'][idx]['continuation']
        tasks.append((i, idx, essay_dof, essay_zero_shot, essay_few_shot, essay_simple, model_name, 3))

    print(f"Running {n_comparisons} judge evaluations with threading...")
    start_time = time.time()
    evaluation_results = []

    with ThreadPoolExecutor(max_workers=num_processes * threads_per_process) as executor:
        future_to_task = {
            executor.submit(judge_single_essay_quad_threaded, *task): task
            for task in tasks
        }
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Judging essays"):
            try:
                result = future.result(timeout=180)
                evaluation_results.append(result)
            except Exception as e:
                task = future_to_task[future]
                evaluation_results.append({
                    'comparison_id': task[0],
                    'essay_idx': task[1],
                    'success': False,
                    'error': f'Thread timeout or error: {str(e)}'
                })

    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds")

    successful_results = [r for r in evaluation_results if r.get('success', False)]
    failed_results = [r for r in evaluation_results if not r.get('success', False)]

    print(f"Successful evaluations: {len(successful_results)}")
    print(f"Failed evaluations: {len(failed_results)}")

    if len(failed_results) > 0:
        print("Failed evaluation errors:")
        error_counts: Dict[str, int] = {}
        for result in failed_results[:10]:
            error = result.get('error', 'Unknown error')
            error_counts[error] = error_counts.get(error, 0) + 1
        for error, count in error_counts.items():
            print(f"  {error}: {count} times")

    criteria = ['creativity', 'lexical_diversity', 'coherence']
    method_keys = ['dof', 'zero_shot', 'few_shot', 'simple_instruction']
    criteria_stats: Dict[str, Any] = {}

    for criterion in criteria:
        criteria_stats[criterion] = {
            'rankings_count': {
                'dof': {'1st': 0, '2nd': 0, '3rd': 0, '4th': 0},
                'zero_shot': {'1st': 0, '2nd': 0, '3rd': 0, '4th': 0},
                'few_shot': {'1st': 0, '2nd': 0, '3rd': 0, '4th': 0},
                'simple_instruction': {'1st': 0, '2nd': 0, '3rd': 0, '4th': 0},
            },
            'total_scores': {'dof': 0, 'zero_shot': 0, 'few_shot': 0, 'simple_instruction': 0},
            'detailed_scores': {'dof': [], 'zero_shot': [], 'few_shot': [], 'simple_instruction': []}
        }

    processed_results = []
    method_map = {'A': 'dof', 'B': 'zero_shot', 'C': 'few_shot', 'D': 'simple_instruction'}

    for result in successful_results:
        judge_result = result['result']
        criterion_scores = convert_rankings_to_scores(judge_result)

        for criterion in criteria:
            ranking = judge_result[criterion]['ranking']
            scores = judge_result[criterion]['scores']

            criteria_stats[criterion]['rankings_count'][method_map[ranking['1st']]]['1st'] += 1
            criteria_stats[criterion]['rankings_count'][method_map[ranking['2nd']]]['2nd'] += 1
            criteria_stats[criterion]['rankings_count'][method_map[ranking['3rd']]]['3rd'] += 1
            criteria_stats[criterion]['rankings_count'][method_map[ranking['4th']]]['4th'] += 1

            for method, score in criterion_scores[criterion].items():
                criteria_stats[criterion]['total_scores'][method] += score

            for essay_id, score in scores.items():
                if essay_id in method_map:
                    method = method_map[essay_id]
                    criteria_stats[criterion]['detailed_scores'][method].append(score)

        processed_results.append({
            'comparison_id': result['comparison_id'],
            'essay_idx': result['essay_idx'],
            'results_by_criterion': judge_result,
            'scores_by_criterion': criterion_scores
        })

    n_successful = len(successful_results)
    for criterion in criteria:
        criteria_stats[criterion]['average_scores'] = {
            method: score / max(n_successful, 1)
            for method, score in criteria_stats[criterion]['total_scores'].items()
        }
        criteria_stats[criterion]['average_detailed_scores'] = {
            method: float(np.mean(scores)) if scores else 0.0
            for method, scores in criteria_stats[criterion]['detailed_scores'].items()
        }

    results_summary = {
        'total_comparisons': n_successful,
        'failed_comparisons': len(failed_results),
        'criteria_stats': criteria_stats,
        'evaluation_details': processed_results,
        'performance': {
            'total_time': end_time - start_time,
            'comparisons_per_second': (n_successful / (end_time - start_time)) if (end_time - start_time) > 0 else 0.0,
            'success_rate': (n_successful / len(evaluation_results) * 100) if evaluation_results else 0.0
        }
    }

    if output_dir is None:
        output_dir = JUDGE_RESULTS_BASE
    ensure_dir(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"judge_results_separate_criteria_4way_{timestamp}.json"

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"Performance: {results_summary['performance']['comparisons_per_second']:.2f} comparisons/second")
    print(f"Success rate: {results_summary['performance']['success_rate']:.1f}%")

    return results_summary

def create_visualization(results_summary: Dict[str, Any], output_dir: Path):
    ensure_dir(output_dir)

    criteria = ['creativity', 'lexical_diversity', 'coherence']
    criteria_labels = ['Creativity', 'Lexical Diversity', 'Coherence']
    methods = ['DoF', 'Zero-shot', 'Few-shot', 'Simple-Instruction']
    method_keys = ['dof', 'zero_shot', 'few_shot', 'simple_instruction']

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    for i, (criterion, criterion_label) in enumerate(zip(criteria, criteria_labels)):
        criterion_data = results_summary['criteria_stats'][criterion]
        rankings_count = criterion_data['rankings_count']
        avg_scores = criterion_data['average_scores']
        avg_detailed_scores = criterion_data['average_detailed_scores']

        first_place = [rankings_count[key]['1st'] for key in method_keys]
        second_place = [rankings_count[key]['2nd'] for key in method_keys]
        third_place = [rankings_count[key]['3rd'] for key in method_keys]
        fourth_place = [rankings_count[key]['4th'] for key in method_keys]
        avg_ranking_scores = [avg_scores[key] for key in method_keys]
        avg_detail_scores = [avg_detailed_scores[key] for key in method_keys]

        x = np.arange(len(methods))
        width = 0.6

        ax1 = axes[i, 0]
        ax1.bar(x, first_place, width, label='1st (4 pts)', color='gold', alpha=0.85)
        ax1.bar(x, second_place, width, bottom=first_place, label='2nd (3 pts)', color='silver', alpha=0.85)
        ax1.bar(x, third_place, width, bottom=np.array(first_place) + np.array(second_place), 
                label='3rd (2 pts)', color='#CD7F32', alpha=0.85)
        ax1.bar(x, fourth_place, width, bottom=np.array(first_place) + np.array(second_place) + np.array(third_place),
                label='4th (1 pt)', color='#8c8c8c', alpha=0.85)

        ax1.set_xlabel('Methods')
        ax1.set_ylabel('Number of Rankings')
        ax1.set_title(f'{criterion_label}: Ranking Distribution')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=0)
        if i == 0:
            ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[i, 1]
        bars = ax2.bar(methods, avg_ranking_scores, alpha=0.85)
        ax2.set_xlabel('Methods')
        ax2.set_ylabel('Average Ranking Score')
        ax2.set_title(f'{criterion_label}: Avg Ranking Scores (max 4)')
        ax2.set_ylim(0, 4)
        ax2.grid(True, alpha=0.3)
        for bar, score in zip(bars, avg_ranking_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                     f'{score:.2f}', ha='center', va='bottom')

        ax3 = axes[i, 2]
        bars = ax3.bar(methods, avg_detail_scores, alpha=0.85)
        ax3.set_xlabel('Methods')
        ax3.set_ylabel('Average Detail Score')
        ax3.set_title(f'{criterion_label}: Avg Detail Scores (1-10)')
        ax3.set_ylim(0, 10)
        ax3.grid(True, alpha=0.3)
        for bar, score in zip(bars, avg_detail_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                     f'{score:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = output_dir / f"separate_criteria_evaluation_4way_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Visualization saved to: {fig_path}")

def print_summary_stats(results_summary: Dict[str, Any]):
    print(f"\n{'='*80}")
    print(f"SEPARATE EVALUATION SUMMARY (4-way): CREATIVITY, LEXICAL DIVERSITY & COHERENCE")
    print(f"{'='*80}")

    total_comparisons = results_summary['total_comparisons']
    failed_comparisons = results_summary.get('failed_comparisons', 0)
    performance = results_summary.get('performance', {})
    criteria_stats = results_summary['criteria_stats']

    print(f"Total successful comparisons: {total_comparisons}")
    print(f"Failed comparisons: {failed_comparisons}")
    print(f"Success rate: {performance.get('success_rate', 0):.1f}%")
    print(f"Processing speed: {performance.get('comparisons_per_second', 0):.2f} comparisons/second")
    print(f"Total time: {performance.get('total_time', 0):.2f} seconds")

    criteria = ['creativity', 'lexical_diversity', 'coherence']
    criteria_labels = ['CREATIVITY', 'LEXICAL DIVERSITY', 'COHERENCE']
    methods = ['DoF', 'Zero-shot', 'Few-shot', 'Simple-Instruction']
    method_keys = ['dof', 'zero_shot', 'few_shot', 'simple_instruction']

    for criterion, criterion_label in zip(criteria, criteria_labels):
        print(f"\n{'='*60}")
        print(f"{criterion_label} EVALUATION RESULTS")
        print(f"{'='*60}")

        criterion_data = criteria_stats[criterion]
        rankings_count = criterion_data['rankings_count']
        avg_scores = criterion_data['average_scores']
        avg_detailed_scores = criterion_data['average_detailed_scores']

        print("Ranking Statistics:")
        print("-" * 85)
        print(f"{'Method':<20} {'1st':<8} {'2nd':<8} {'3rd':<8} {'4th':<8} {'Avg Score':<12} {'Avg Detail':<10}")
        print("-" * 85)

        for method, key in zip(methods, method_keys):
            first = rankings_count[key]['1st']
            second = rankings_count[key]['2nd']
            third = rankings_count[key]['3rd']
            fourth = rankings_count[key]['4th']
            avg_score = avg_scores[key]
            avg_detail = avg_detailed_scores[key]
            print(f"{method:<20} {first:<8} {second:<8} {third:<8} {fourth:<8} {avg_score:<12.2f} {avg_detail:<10.1f}")

        print("-" * 85)

        print(f"\n{criterion_label.title()} Win Rates (1st place percentage):")
        for method, key in zip(methods, method_keys):
            if total_comparisons > 0:
                win_rate = rankings_count[key]['1st'] / total_comparisons * 100
                print(f"{method}: {win_rate:.1f}%")

        if total_comparisons > 0:
            best_method = max(avg_scores.items(), key=lambda x: x[1])
            criterion_display = criterion.replace('_', ' ')
            print(f"\nBest performing method for {criterion_display}: {best_method[0].replace('_', '-')} (avg score: {best_method[1]:.2f})")

    print(f"\n{'='*60}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*60}")

    overall_wins = {'dof': 0, 'zero_shot': 0, 'few_shot': 0, 'simple_instruction': 0}
    for criterion in criteria:
        criterion_data = criteria_stats[criterion]
        rankings_count = criterion_data['rankings_count']
        for method in method_keys:
            overall_wins[method] += rankings_count[method]['1st']

    print("Total 1st place wins across all criteria:")
    for method, key in zip(methods, method_keys):
        total_wins = overall_wins[key]
        win_percentage = (total_wins / (total_comparisons * 3)) * 100 if total_comparisons > 0 else 0.0
        print(f"{method}: {total_wins} wins ({win_percentage:.1f}%)")

def load_results_summary(summary_path: Path) -> Dict[str, Any]:
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="LLM-as-a-Judge (4-way): Separate Evaluation of Creativity, Lexical Diversity & Coherence")
    parser.add_argument("--results-dir", type=str,
                       default=r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\results\gemini\gemini-2.0-flash\essay",
                       help="Directory containing the essay results")
    parser.add_argument("--n-comparisons", type=int, default=100,
                       help="Number of essay comparisons to evaluate")
    parser.add_argument("--model-name", type=str, default="gemini-2.0-flash",
                       help="Judge model name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")

    parser.add_argument("--num-processes", type=int, default=4,
                       help="Number of processes for multiprocessing")
    parser.add_argument("--batch-size", type=int, default=5,
                       help="Batch size for processing")
    parser.add_argument("--use-threading", action="store_true", default=True,
                       help="Use threading instead of multiprocessing")
    parser.add_argument("--threads-per-process", type=int, default=3,
                       help="Number of threads per process when using threading")

    parser.add_argument("--viz-from-summary", type=str, default=None,
                        help="Path to a saved judge_results_*.json to generate ONLY visualization and summary printing (no evaluation).")
    parser.add_argument("--viz-out-dir", type=str, default=None,
                        help="Custom directory to save the generated visualization image (defaults to --output-dir or JUDGE_RESULTS_BASE).")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else JUDGE_RESULTS_BASE

    if args.viz_from_summary:
        summary_path = Path(args.viz_from_summary)
        if not summary_path.exists():
            raise FileNotFoundError(f"--viz-from-summary file not found: {summary_path}")
        results_summary = load_results_summary(summary_path)
        viz_out_dir = Path(args.viz_out_dir) if args.viz_out_dir else output_dir
        ensure_dir(viz_out_dir)
        print_summary_stats(results_summary)
        if not args.no_viz:
            create_visualization(results_summary, viz_out_dir)
        return

    try:
        results_summary = run_judge_evaluation(
            results_dir=results_dir,
            n_comparisons=args.n_comparisons,
            model_name=args.model_name,
            seed=args.seed,
            output_dir=output_dir,
            num_processes=args.num_processes,
            batch_size=args.batch_size,
            use_threading=args.use_threading,
            threads_per_process=args.threads_per_process
        )

        print_summary_stats(results_summary)

        if not args.no_viz:
            create_visualization(results_summary, output_dir)

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
