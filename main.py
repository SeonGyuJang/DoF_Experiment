# python main.py \
#   --model gemini \
#   --model-name gemini-2.0-flash \
#   --dataset train \
#   --sample 1000 \
#   --dofs 0.0,0.5,1.0 \
#   --n-per-dof 10 \
#   --num-processes 4 \
#   --batch-size 2 \
#   --prompt-file C:\Users\dsng3\Documents\GitHub\DoF_Experiment\prompts\exp1.tmpl \
#   --preview

# python main.py --model gemini --model-name gemini-2.0-flash --dataset train --sample 1000 --dofs 0.0,0.5,1.0 --n-per-dof 10 --num-processes 4 --batch-size 2 --prompt-file C:\Users\dsng3\Documents\GitHub\DoF_Experiment\prompts\exp3.tmpl --preview

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
RESULTS_BASE = BASE / "results"

DATA_PATHS: Dict[str, Path] = {
    "train": Path(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\train-00000-of-00001.parquet"),
    "test": Path(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\test-00000-of-00001.parquet")
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

def key_of(index: int, dof_value: float, sample_id: int) -> Tuple[int, str, int]:
    return (int(index), f"{float(dof_value):.6f}", int(sample_id))

def parse_result_key(obj: Dict[str, Any]) -> Tuple[int, str, int]:
    return key_of(int(obj["index"]), float(obj["dof_value"]), int(obj["sample_id"]))

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
            "GOOGLE_API_KEY not found. Ensure your .env is discoverable on Windows.\n"
            f"cwd={os.getcwd()} base={BASE}"
        )

def load_dataset(
    dataset_name: str,
    sample_size: Optional[int] = None,
    target_ids: Optional[List[int]] = None,
    seed: int = 42
) -> pd.DataFrame:
    data_path = DATA_PATHS[dataset_name]
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    try:
        df = pd.read_parquet(data_path)
    except Exception as e0:
        try:
            import pyarrow
            df = pd.read_parquet(data_path, engine="pyarrow")
        except Exception:
            try:
                import fastparquet
                df = pd.read_parquet(data_path, engine="fastparquet")
            except Exception:
                raise RuntimeError(
                    f"Failed to read parquet at {data_path}. "
                    f"Install pyarrow or fastparquet. Original error: {repr(e0)}"
                )
    if "text" not in df.columns:
        raise KeyError(
            "Input dataset must contain a 'text' column. "
            f"Columns found: {list(df.columns)}"
        )
    if target_ids:
        df = df[df.index.isin(target_ids)]
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)
    df = df.sort_index()
    return df

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

def build_payload(prompt_text: str, text_value: str, dof_value: float, nonce: Optional[int]):
    tmpl = PromptTemplate.from_template(prompt_text)
    vars_ = set(tmpl.input_variables)
    payload: Dict[str, Any] = {}
    if "text" in vars_:
        payload["text"] = text_value
    if "sentence" in vars_:
        payload["sentence"] = text_value
    if "dof_value" in vars_:
        payload["dof_value"] = dof_value
    if "nonce" in vars_ and nonce is not None:
        payload["nonce"] = nonce
    return payload

def render_preview(prompt: PromptTemplate, text_value: str, dof_value: float, prompt_text: str) -> str:
    use_nonce = "{nonce}" in prompt_text
    nonce = 123456789 if use_nonce else None
    kwargs = build_payload(prompt_text, text_value, dof_value, nonce)
    return prompt.format(**kwargs)

def load_existing_results(jsonl_path: Path) -> Tuple[Dict[Tuple[int, str, int], Dict[str, Any]], set, set]:
    results_map: Dict[Tuple[int, str, int], Dict[str, Any]] = {}
    success_keys, error_keys = set(), set()
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
    for idx, text_value, dof, sample_id in batch_items:
        nonce = random.getrandbits(64) if use_nonce else None
        payload = build_payload(prompt_text, text_value, dof, nonce)
        retry = 0
        last_err = None
        while retry < 5:
            try:
                res = chain.invoke(payload)
                out.append({
                    "index": idx,
                    "original_text": text_value,
                    "dof_value": dof,
                    "sample_id": sample_id,
                    "prompt_name": prompt_name,
                    "continuation": res.get("continuation", ""),
                    "reasoning": res.get("reasoning", ""),
                    "error": None
                })
                break
            except Exception as e:
                last_err = repr(e)
                retry += 1
        if retry >= 5:
            out.append({
                "index": idx,
                "original_text": text_value,
                "dof_value": dof,
                "sample_id": sample_id,
                "prompt_name": prompt_name,
                "continuation": "",
                "reasoning": "",
                "error": f"max_retries_exceeded: {last_err}"
            })
        pbar.update(1)
    pbar.close()
    return out

def run_experiment(
    model_type: str,
    model_name: str,
    dataset_name: str,
    dofs: List[float],
    n_per_dof: int,
    prompt_text: str,
    prompt_name: str,
    sample_size: Optional[int] = None,
    target_ids: Optional[List[int]] = None,
    num_processes: int = 8,
    batch_size: int = 2,
    preview: bool = False,
    preview_sentence_index: int = 0,
    preview_dof_index: int = 0,
    seed: int = 42,
    resume_from: Optional[Path] = None,
    rerun_errors_only: bool = False,
    output_path: Optional[Path] = None
):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    print(f"\n{'='*60}")
    print(f"DoF Multi-Sample Generation")
    print(f"Model: {model_type}/{model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Prompt: {prompt_name}")
    print(f"DoFs: {dofs}  |  n_per_dof: {n_per_dof}")
    print(f"Processes: {num_processes}  |  Batch size (items): {batch_size}")
    print(f"Seed: {seed}")
    if resume_from:
        print(f"Resume from: {resume_from}  |  Mode: {'errors-only' if rerun_errors_only else 'missing+errors'}")
    print(f"{'='*60}\n")
    print("[env] Loading .env and checking GOOGLE_API_KEY ...")
    init_env()
    print("[env] OK")
    print(f"[1/5] Loading dataset from: {DATA_PATHS[dataset_name]}")
    df = load_dataset(dataset_name, sample_size, target_ids, seed=seed)
    print(f"Loaded {len(df)} rows (expects 'text' column).")
    if preview:
        if not (0 <= preview_sentence_index < len(df)):
            preview_sentence_index = 0
        if not (0 <= preview_dof_index < len(dofs)):
            preview_dof_index = 0
        text_value = df.iloc[preview_sentence_index]["text"]
        dof_val = dofs[preview_dof_index]
        _, prompt_obj = _build_chain(model_type, model_name, prompt_text)
        rendered = render_preview(prompt_obj, text_value, dof_val, prompt_text)
        print("\n[PREVIEW] ===== Rendered Prompt Sent to Gemini =====")
        print(rendered)
        print("===== /PREVIEW =====================================\n")
    print("[2/5] Building work items...")
    items: List[Tuple[int, str, float, int]] = []
    for idx, row in df.iterrows():
        text_value = row["text"]
        for d in dofs:
            for k in range(n_per_dof):
                items.append((int(idx), str(text_value), float(d), int(k)))
    existing_map: Dict[Tuple[int, str, int], Dict[str, Any]] = {}
    success_keys, error_keys = set(), set()
    if resume_from:
        if not Path(resume_from).exists():
            raise FileNotFoundError(f"--resume-from not found: {resume_from}")
        existing_map, success_keys, error_keys = load_existing_results(Path(resume_from))
        if rerun_errors_only:
            items = [it for it in items if key_of(it[0], it[2], it[3]) in error_keys]
        else:
            done = success_keys
            items = [it for it in items if key_of(it[0], it[2], it[3]) not in done]
    total_expected = len(df) * len(dofs) * n_per_dof
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
            out_dir = RESULTS_BASE / model_type / model_name
            ensure_dir(out_dir)
            final_jsonl_path = out_dir / f"{model_type}_{model_name}_{dataset_name}_prompt-{prompt_name}_DoF_multi_{timestamp}.jsonl"
    meta_path = final_jsonl_path.with_suffix(".meta.json")
    print("[3/5] Saving meta (initial snapshot)...")
    meta = {
        "model_type": model_type,
        "model_name": model_name,
        "dataset": dataset_name,
        "dofs": dofs,
        "n_per_dof": n_per_dof,
        "num_input_rows": int(len(df)),
        "total_expected_outputs": int(total_expected),
        "batch_size_items": batch_size,
        "num_processes": num_processes,
        "output_jsonl": str(final_jsonl_path),
        "prompt_file_used": True,
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

def parse_args():
    p = argparse.ArgumentParser(
        description="DoF multi-sample generator (API-only, external prompt file, JSONL streaming (real-time writes), prompt-aware filenames, preview, reproducible sampling, resume)"
    )
    p.add_argument("--model", choices=list(USE_MODEL.keys()), required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--dataset", choices=["train", "test"], required=True)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--all", action="store_true", help="Use all rows in dataset")
    grp.add_argument("--sample", type=int, help="Number of rows to sample")
    grp.add_argument("--ids", type=str, help="Comma-separated list of index IDs")
    p.add_argument("--dofs", type=str, required=True, help='Comma-separated DoF list, e.g. "0.0,0.5,1.0"')
    p.add_argument("--n-per-dof", type=int, default=30, help="How many samples per DoF for each row")
    p.add_argument("--num-processes", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2, help="Batch size in ITEMS (row×DoF×sample_id)")
    p.add_argument("--prompt-file", type=str, required=True, help="Path to external prompt template (.txt/.tmpl)")
    p.add_argument("--preview", action="store_true", help="Print a rendered prompt preview before running")
    p.add_argument("--preview-sentence-index", type=int, default=0, help="Row index in df (after sampling) to preview")
    p.add_argument("--preview-dof-index", type=int, default=0, help="Index into --dofs list to preview")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible input sampling and nonce generation")
    p.add_argument("--resume-from", type=str, help="Path to an existing JSONL to resume (run only missing or failed items)")
    p.add_argument("--rerun-errors-only", action="store_true", help="When resuming, re-run only rows that had non-empty error")
    p.add_argument("--output", type=str, help="Optional explicit JSONL output path. If omitted and --resume-from is set, we append to that file; otherwise a timestamped file is created.")
    return p.parse_args()

def main():
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    args = parse_args()
    if args.model not in USE_MODEL:
        print(f"Unknown model type: {args.model}")
        return
    if args.model_name not in USE_MODEL[args.model]:
        print(f"Unknown model name '{args.model_name}' for {args.model}")
        print(f"Available: {list(USE_MODEL[args.model].keys())}")
        return
    try:
        dofs = [float(x.strip()) for x in args.dofs.split(",")]
    except Exception:
        print('Error: --dofs must be comma-separated floats, e.g., "0.0,0.5,1.0"')
        return
    for d in dofs:
        if not (0.0 <= d <= 1.0):
            print(f"Error: DoF {d} out of range [0.0, 1.0]")
            return
    sample_size = None
    target_ids = None
    if args.sample is not None:
        sample_size = args.sample
    elif args.ids:
        target_ids = [int(x.strip()) for x in args.ids.split(",")]
    elif not args.all:
        print("You must provide one of --all, --sample, or --ids.")
        return
    prompt_text = load_prompt_file(args.prompt_file)
    prompt_name = get_prompt_name(args.prompt_file)
    resume_from = Path(args.resume_from) if args.resume_from else None
    output_path = Path(args.output) if args.output else None
    out = run_experiment(
        model_type=args.model,
        model_name=args.model_name,
        dataset_name=args.dataset,
        dofs=dofs,
        n_per_dof=args.n_per_dof,
        prompt_text=prompt_text,
        prompt_name=prompt_name,
        sample_size=sample_size,
        target_ids=target_ids,
        num_processes=args.num_processes,
        batch_size=args.batch_size,
        preview=args.preview,
        preview_sentence_index=args.preview_sentence_index,
        preview_dof_index=args.preview_dof_index,
        seed=args.seed,
        resume_from=resume_from,
        rerun_errors_only=args.rerun_errors_only,
        output_path=output_path
    )
    print(f"\nOutput file: {out}")

if __name__ == "__main__":
    main()
