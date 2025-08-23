# python essay_main.py --model gemini --model-name gemini-2.0-flash --prompt-file C:\Users\dsng3\Documents\GitHub\DoF_Experiment\prompts\essay\zero_shot_prompt7.tmpl --prompt-type zero_shot --n-samples 10000 --preview
# python essay_main.py --model gemini --model-name gemini-2.0-flash --prompt-file C:\Users\dsng3\Documents\GitHub\DoF_Experiment\prompts\essay\few_shot_exp1_prompt7.tmpl --prompt-type few_shot --dataset train --essay-set 7 --few-shot-count 3 --n-samples 10000 --preview

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from multiprocessing import Pool, set_start_method, current_process
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

BASE = Path(__file__).resolve().parent
RESULTS_BASE = Path("/Users/jangseongyu/Documents/GitHub/DoF_Experiment/results/gemini/gemini-2.0-flash/essay")
PROMPTS_BASE = Path("/Users/jangseongyu/Documents/GitHub/DoF_Experiment/prompts/essay")

DATA_PATHS: Dict[str, Path] = {
    "train": Path("/Users/jangseongyu/Documents/GitHub/DoF_Experiment/data/essay/training_set_rel3.tsv"),
    "test": Path("/Users/jangseongyu/Documents/GitHub/DoF_Experiment/data/essay/valid_set.tsv")
}

USE_MODEL: Dict[str, Dict[str, Dict[str, float | int]]] = {
    "gemini": {
        "gemini-2.0-flash-lite": {"temperature": 1.0, "max_output_tokens": 8192},
        "gemini-2.0-flash": {"temperature": 0.7, "max_output_tokens": 8192},
        "gemini-2.5-flash": {"temperature": 1.0, "max_output_tokens": 8192},
        "gemini-2.5-pro": {"temperature": 1.0, "max_output_tokens": 8192}
    }
}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def key_of(sample_id: int) -> Tuple[int]:
    return (int(sample_id),)

def parse_result_key(obj: Dict[str, Any]) -> Tuple[int]:
    return key_of(int(obj["sample_id"]))

def load_prompt_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8", errors="replace")

def get_prompt_name(prompt_file: str) -> str:
    return Path(prompt_file).stem

def init_env():
    loaded = load_dotenv(dotenv_path=find_dotenv(usecwd=True))
    if not loaded:
        load_dotenv(BASE / ".env")
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY not found. Ensure your .env is discoverable.\n"
            f"cwd={os.getcwd()} base={BASE}"
        )

def load_dataset(
    dataset_name: str,
    sample_size: Optional[int] = None,
    target_ids: Optional[List[int]] = None,
    essay_set: Optional[int] = None,
    seed: int = 42
) -> pd.DataFrame:
    data_path = DATA_PATHS[dataset_name]
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    try:
        if str(data_path).endswith('.tsv'):
            df = pd.read_csv(data_path, sep='\t', encoding='latin1')
        else:
            df = pd.read_csv(data_path, encoding='utf-8')
    except Exception as e:
        try:
            if str(data_path).endswith('.tsv'):
                df = pd.read_csv(data_path, sep='\t', encoding='utf-8', errors='ignore')
            else:
                df = pd.read_csv(data_path, encoding='latin1')
        except Exception as e2:
            raise RuntimeError(f"Failed to read file at {data_path}. Tried both latin1 and utf-8 encodings. Error: {repr(e)}, {repr(e2)}")
    print(f"Original dataset size: {len(df)} rows")
    print(f"Available essay_sets: {sorted(df['essay_set'].unique()) if 'essay_set' in df.columns else 'No essay_set column'}")
    if essay_set is not None:
        if 'essay_set' not in df.columns:
            raise KeyError("Dataset does not contain 'essay_set' column")
        original_size = len(df)
        df = df[df['essay_set'] == essay_set]
        print(f"Filtered to essay_set={essay_set}: {len(df)} rows (from {original_size})")
        if len(df) == 0:
            raise ValueError(f"No data found for essay_set={essay_set}")
    if "essay" in df.columns and "text" not in df.columns:
        df["text"] = df["essay"]
    if "text" not in df.columns:
        raise KeyError(
            "Input dataset must contain a 'text' or 'essay' column. "
            f"Columns found: {list(df.columns)}"
        )
    if target_ids:
        df = df[df.index.isin(target_ids)]
        print(f"Filtered to target_ids: {len(df)} rows")
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)
        print(f"Sampled {sample_size} rows from {len(df)} available")
    df = df.sort_index()
    return df

def get_few_shot_examples(df: pd.DataFrame, n_examples: int, seed: int = 42) -> str:
    random.seed(seed)
    if len(df) < n_examples:
        print(f"Warning: Requested {n_examples} examples but only {len(df)} available")
        n_examples = len(df)
    sample_essays = df.sample(n=n_examples, random_state=seed)
    examples = []
    for i, (_, row) in enumerate(sample_essays.iterrows(), 1):
        essay_text = row['text']
        examples.append(f"Example {i}: {essay_text}")
    return "\n\n".join(examples)

def prepare_prompt_text(prompt_text: str, prompt_type: str, few_shot_count: Optional[int] = None, examples_df: Optional[pd.DataFrame] = None, seed: int = 42) -> str:
    if prompt_type == "few_shot":
        if few_shot_count is None or examples_df is None:
            raise ValueError("Few-shot requires few_shot_count and examples_df")
        if "{few_shot}" not in prompt_text:
            raise ValueError("Few-shot prompt template must contain {few_shot} placeholder")
        few_shot_examples = get_few_shot_examples(examples_df, few_shot_count, seed)
        return prompt_text.replace("{few_shot}", few_shot_examples)
    return prompt_text

def initialize_llm(model_type: str, model_name: str):
    if model_type != "gemini":
        raise ValueError(f"Only 'gemini' is supported in this script. Got: {model_type}")
    cfg = USE_MODEL[model_type][model_name]
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=cfg["temperature"],
        max_output_tokens=cfg["max_output_tokens"]
    )

def _build_chain(model_type: str, model_name: str, prompt_text: str):
    llm = initialize_llm(model_type, model_name)
    prompt = PromptTemplate.from_template(prompt_text)
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    return chain, prompt

def build_payload(prompt_text: str, nonce: Optional[int]):
    tmpl = PromptTemplate.from_template(prompt_text)
    vars_ = set(tmpl.input_variables)
    payload: Dict[str, Any] = {}
    if "input_length" in vars_:
        payload["input_length"] = 600
    if "nonce" in vars_ and nonce is not None:
        payload["nonce"] = nonce
    return payload

def render_preview(prompt: PromptTemplate, prompt_text: str) -> str:
    use_nonce = "{nonce}" in prompt_text
    nonce = 123456789 if use_nonce else None
    kwargs = build_payload(prompt_text, nonce)
    return prompt.format(**kwargs)

def load_existing_results(jsonl_path: Path) -> Tuple[Dict[Tuple[int], Dict[str, Any]], set, set]:
    results_map: Dict[Tuple[int], Dict[str, Any]] = {}
    success_keys, error_keys = set(), set()
    if not jsonl_path.exists():
        return results_map, success_keys, error_keys
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            try:
                k = parse_result_key(obj)
            except Exception:
                continue
            results_map[k] = obj
            err = obj.get("error")
            cont = obj.get("continuation", "")
            if err is None or err == "":
                if isinstance(cont, str) and cont.strip() != "":
                    success_keys.add(k)
                else:
                    error_keys.add(k)
            else:
                error_keys.add(k)
    return results_map, success_keys, error_keys

def worker_process_batch(args):
    (batch_items, model_type, model_name, prompt_text, prompt_name, seed) = args
    try:
        wid = current_process()._identity[0]
    except Exception:
        wid = 1
    wid_pos = wid if wid >= 1 else 1
    random.seed(seed + wid_pos)
    chain, _ = _build_chain(model_type, model_name, prompt_text)
    out: List[Dict[str, Any]] = []
    pbar = tqdm(
        total=len(batch_items),
        position=wid_pos,
        desc=f"P{wid_pos} | items={len(batch_items)}",
        leave=False
    )
    use_nonce = "{nonce}" in prompt_text
    for sample_id in batch_items:
        nonce = random.getrandbits(64) if use_nonce else None
        payload = build_payload(prompt_text, nonce)
        retry = 0
        last_err = None
        while retry < 5:
            try:
                res = chain.invoke(payload)
                result = {
                    "sample_id": sample_id,
                    "prompt_name": prompt_name,
                    "continuation": res.get("continuation", ""),
                    "reasoning": res.get("reasoning", ""),
                    "error": None
                }
                out.append(result)
                break
            except Exception as e:
                last_err = repr(e)
                retry += 1
        if retry >= 5:
            result = {
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "continuation": "",
                "reasoning": "",
                "error": f"max_retries_exceeded: {last_err}"
            }
            out.append(result)
        pbar.update(1)
    pbar.close()
    return out

def run_experiment(
    model_type: str,
    model_name: str,
    prompt_file: str,
    prompt_type: str,
    n_samples: int,
    dataset_name: Optional[str] = None,
    essay_set: Optional[int] = None,
    few_shot_count: Optional[int] = None,
    num_processes: int = 8,
    batch_size: int = 2,
    preview: bool = False,
    seed: int = 42,
    resume_from: Optional[Path] = None,
    rerun_errors_only: bool = False,
    output_path: Optional[Path] = None
):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    print(f"\n{'='*60}")
    print(f"Essay Generation Experiment")
    print(f"Model: {model_type}/{model_name}")
    print(f"Prompt Type: {prompt_type}")
    print(f"Prompt File: {prompt_file}")
    if prompt_type == "few_shot":
        print(f"Few-shot Examples: {few_shot_count}")
        print(f"Dataset: {dataset_name} (for examples)")
        if essay_set is not None:
            print(f"Essay Set: {essay_set}")
    print(f"Total samples to generate: {n_samples}")
    print(f"Processes: {num_processes}  |  Batch size (items): {batch_size}")
    print(f"Seed: {seed}")
    if resume_from:
        print(f"Resume from: {resume_from}  |  Mode: {'errors-only' if rerun_errors_only else 'missing+errors'}")
    print(f"{'='*60}\n")
    print("[env] Loading .env and checking GOOGLE_API_KEY ...")
    init_env()
    print("[env] OK")
    print(f"[1/5] Loading prompt from: {prompt_file}")
    prompt_text = load_prompt_file(prompt_file)
    prompt_name = get_prompt_name(prompt_file)
    if prompt_type == "few_shot":
        if not dataset_name:
            raise ValueError("Few-shot requires dataset_name for examples")
        print(f"[1.1/5] Preparing few-shot examples...")
        examples_df = load_dataset(dataset_name, essay_set=essay_set, seed=seed + 1000)
        prompt_text = prepare_prompt_text(prompt_text, prompt_type, few_shot_count, examples_df, seed)
        prompt_name += f"_{few_shot_count}shot"
    if preview:
        _, prompt_obj = _build_chain(model_type, model_name, prompt_text)
        rendered = render_preview(prompt_obj, prompt_text)
        print("\n[PREVIEW] ===== Rendered Prompt Sent to Gemini =====")
        print(rendered)
        print("===== /PREVIEW =====================================\n")
    print("[2/5] Building work items...")
    items: List[int] = list(range(n_samples))
    existing_map: Dict[Tuple[int], Dict[str, Any]] = {}
    success_keys, error_keys = set(), set()
    if resume_from:
        if not Path(resume_from).exists():
            raise FileNotFoundError(f"--resume-from not found: {resume_from}")
        existing_map, success_keys, error_keys = load_existing_results(Path(resume_from))
        if rerun_errors_only:
            items = [it for it in items if key_of(it) in error_keys]
        else:
            done = success_keys
            items = [it for it in items if key_of(it) not in done]
    total_expected = n_samples
    total_pending = len(items)
    print(f"[i] Total expected outputs: {total_expected}")
    if resume_from:
        print(f"[i] Existing file entries: {len(existing_map)}  (success: {len(success_keys)}, errors: {len(error_keys)})")
    print(f"[i] Pending items to run:   {total_pending}")
    if output_path:
        final_jsonl_path = Path(output_path)
        ensure_dir(final_jsonl_path.parent)
    else:
        if resume_from:
            final_jsonl_path = Path(resume_from)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = RESULTS_BASE 
            ensure_dir(out_dir)
            final_jsonl_path = out_dir / f"{model_type}_{model_name}_prompt-{prompt_name}_{timestamp}.jsonl"
    meta_path = final_jsonl_path.with_suffix(".meta.json")
    print("[3/5] Saving meta (initial snapshot)...")
    meta = {
        "model_type": model_type,
        "model_name": model_name,
        "prompt_type": prompt_type,
        "prompt_file": prompt_file,
        "dataset_name": dataset_name if prompt_type == "few_shot" else None,
        "essay_set": essay_set if prompt_type == "few_shot" else None,
        "few_shot_count": few_shot_count if prompt_type == "few_shot" else None,
        "n_samples": n_samples,
        "total_expected_outputs": int(total_expected),
        "batch_size_items": batch_size,
        "num_processes": num_processes,
        "output_jsonl": str(final_jsonl_path),
        "prompt_name": prompt_name,
        "prompt_text_snapshot": prompt_text,
        "seed": seed,
        "resume_from": str(resume_from) if resume_from else None,
        "resume_mode": "errors-only" if rerun_errors_only else ("missing+errors" if resume_from else None),
        "existing_entries": len(existing_map) if resume_from else 0
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[i] Meta saved: {meta_path}")
    print(f"[4/5] Parallel generation (item-level) ... processes={num_processes}")
    new_results_count = 0
    open_mode = "a" if final_jsonl_path.exists() else "w"
    with open(final_jsonl_path, open_mode, encoding="utf-8") as fout:
        if total_pending > 0:
            task_batches = list(chunk_list(items, batch_size))
            with Pool(processes=num_processes) as pool:
                for batch_result in tqdm(
                    pool.imap_unordered(
                        worker_process_batch,
                        [
                            (batch, model_type, model_name, prompt_text, prompt_name, seed)
                            for batch in task_batches
                        ]
                    ),
                    total=len(task_batches),
                    desc="Generating (batches)",
                    position=0,
                    leave=True
                ):
                    for obj in batch_result:
                        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    fout.flush()
                    try:
                        os.fsync(fout.fileno())
                    except OSError:
                        pass
                    new_results_count += len(batch_result)
    print("\n[5/5] Finalize...")
    print("\n[✓] Done.")
    print(f"Output JSONL: {final_jsonl_path}")
    print(f"New lines written this run: {new_results_count}")
    if resume_from:
        print(f"(Resumed) Existing entries previously in file: {len(existing_map)}")
    else:
        print(f"Expected total for this run: {total_pending}")
    return str(final_jsonl_path)

def setup_project_structure():
    ensure_dir(PROMPTS_BASE)  
    ensure_dir(RESULTS_BASE)  
    env_path = BASE / ".env.example"
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write("GOOGLE_API_KEY=your_google_api_key_here\n")
        print(f"Created example environment file: {env_path}")
    print(f"Project structure created.")
    print(f"Results will be saved to: {RESULTS_BASE}")

def parse_args():
    p = argparse.ArgumentParser(
        description="Essay generation experiment system with external prompt templates"
    )
    p.add_argument("--model", choices=list(USE_MODEL.keys()), required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--prompt-file", type=str, required=True, help="Path to external prompt template (.txt/.tmpl)")
    p.add_argument("--prompt-type", choices=["dof", "zero_shot", "few_shot", "simple_instruction"], required=True, help="Type of prompt to use")
    p.add_argument("--n-samples", type=int, required=True, help="Total number of samples to generate")
    p.add_argument("--dataset", choices=["train", "test"], help="Dataset for few-shot examples (required if prompt-type is few_shot)")
    p.add_argument("--essay-set", type=int, help="Filter by specific essay_set number for few-shot examples")
    p.add_argument("--few-shot-count", type=int, help="Number of examples for few-shot prompt (required if prompt-type is few_shot)")
    p.add_argument("--num-processes", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4, help="Batch size in ITEMS")
    p.add_argument("--preview", action="store_true", help="Print a rendered prompt preview before running")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    p.add_argument("--resume-from", type=str, help="Path to an existing JSONL to resume (run only missing or failed items)")
    p.add_argument("--rerun-errors-only", action="store_true", help="When resuming, re-run only rows that had non-empty error")
    p.add_argument("--output", type=str, help="Optional explicit JSONL output path")
    p.add_argument("--setup", action="store_true", help="Setup project structure")
    return p.parse_args()

def main():
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    args = parse_args()
    if args.setup:
        print("Setting up project structure...")
        setup_project_structure()
        print("Setup complete!")
        return
    if args.model not in USE_MODEL:
        print(f"Unknown model type: {args.model}")
        return
    if args.model_name not in USE_MODEL[args.model]:
        print(f"Unknown model name '{args.model_name}' for {args.model}")
        print(f"Available: {list(USE_MODEL[args.model].keys())}")
        return
    if args.prompt_type == "few_shot":
        if args.few_shot_count is None:
            print("Error: --few-shot-count is required when using few_shot prompt type")
            return
        if args.dataset is None:
            print("Error: --dataset is required when using few_shot prompt type")
            return
    resume_from = Path(args.resume_from) if args.resume_from else None
    output_path = Path(args.output) if args.output else None
    out = run_experiment(
        model_type=args.model,
        model_name=args.model_name,
        prompt_file=args.prompt_file,
        prompt_type=args.prompt_type,
        n_samples=args.n_samples,
        dataset_name=args.dataset,
        essay_set=args.essay_set,
        few_shot_count=args.few_shot_count,
        num_processes=args.num_processes,
        batch_size=args.batch_size,
        preview=args.preview,
        seed=args.seed,
        resume_from=resume_from,
        rerun_errors_only=args.rerun_errors_only,
        output_path=output_path
    )
    print(f"\nOutput file: {out}")

if __name__ == "__main__":
    main()
