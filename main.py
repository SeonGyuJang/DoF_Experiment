# python main.py --model gemini --model-name gemini-2.0-flash --dataset train --sample 1000 --dofs 0.0,0.5,1.0 --n-per-dof 30 --num-processes 8 --batch-size 2 --prompt-file prompts/exp1.tmpl --preview

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from multiprocessing import Pool, set_start_method, current_process
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

BASE = Path(__file__).resolve().parent
DATA_PATHS: Dict[str, Path] = {
    "train": Path(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\train-00000-of-00001.parquet"),
    "validation": Path(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\validation-00000-of-00001.parquet"),
    "test": Path(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\test-00000-of-00001.parquet")
}
RESULTS_BASE = BASE / "results"

USE_MODEL: Dict[str, Dict[str, Dict[str, float | int]]] = {
    "gemini": {
        "gemini-2.0-flash-lite": {"temperature": 1.0, "max_output_tokens": 8192},
        "gemini-2.0-flash": {"temperature": 1.0, "max_output_tokens": 8192},
        "gemini-2.5-flash": {"temperature": 1.0, "max_output_tokens": 8192},
        "gemini-2.5-pro": {"temperature": 1.0, "max_output_tokens": 8192}
    }
}

# ---------------- I/O utils ----------------
def initialize_llm(model_type: str, model_name: str):
    if model_type != "gemini":
        raise ValueError(f"Only 'gemini' is supported in this script. Got: {model_type}")
    cfg = USE_MODEL[model_type][model_name]
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=cfg["temperature"],
        max_output_tokens=cfg["max_output_tokens"]
    )

def load_dataset(dataset_name: str, sample_size: Optional[int] = None, 
                 target_ids: Optional[List[int]] = None) -> pd.DataFrame:
    data_path = DATA_PATHS[dataset_name]
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    df = pd.read_parquet(data_path)
    if target_ids:
        df = df[df.index.isin(target_ids)]
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    return df

def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_prompt_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8")

def get_prompt_name(prompt_file: str) -> str:
    """exp6.tmpl -> exp6  /  my_prompt.txt -> my_prompt"""
    return Path(prompt_file).stem

# ---------------- chain builder & preview ----------------
def _build_chain(model_type: str, model_name: str, prompt_text: str):
    """Create LLM chain: PromptTemplate -> LLM -> JSON parser"""
    llm = initialize_llm(model_type, model_name)
    prompt = PromptTemplate.from_template(prompt_text)
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    return chain, prompt

def render_preview(prompt: PromptTemplate, sentence: str, dof_value: float, prompt_text: str) -> str:
    """실제 모델에 들어갈 최종 텍스트(포맷된 프롬프트)를 미리보기로 반환"""
    use_nonce = "{nonce}" in prompt_text
    kwargs = {"sentence": sentence, "dof_value": dof_value}
    if use_nonce:
        kwargs["nonce"] = 123456789  # 미리보기용 고정 nonce
    return prompt.format(**kwargs)

# ---------------- worker ----------------
def worker_process_batch(args):
    """
    args: (batch_sentences, model_type, model_name, dofs, n_per_dof, prompt_text, prompt_name)
    batch_sentences: List[Tuple[int, str]]
    """
    batch_sentences, model_type, model_name, dofs, n_per_dof, prompt_text, prompt_name = args

    # Worker ID (1..num_processes) -> 진행바 position
    try:
        wid = current_process()._identity[0]
    except Exception:
        wid = 1  # fallback
    wid_pos = wid if wid >= 1 else 1  # 메인은 position=0, 워커는 1~

    chain, _ = _build_chain(model_type, model_name, prompt_text)
    out: List[Dict[str, Any]] = []

    total_steps = len(batch_sentences) * len(dofs) * n_per_dof
    pbar = tqdm(
        total=total_steps,
        position=wid_pos,   # per-process progress bar position
        desc=f"P{wid_pos} | batch={len(batch_sentences)}",
        leave=False
    )

    use_nonce = "{nonce}" in prompt_text

    for idx, sentence in batch_sentences:
        for dof in dofs:
            for k in range(n_per_dof):
                payload = {"sentence": sentence, "dof_value": dof}
                if use_nonce:
                    payload["nonce"] = random.getrandbits(64)

                retry = 0
                last_err = None
                while retry < 5:
                    try:
                        res = chain.invoke(payload)
                        out.append({
                            "index": idx,
                            "original_sentence": sentence,
                            "dof_value": dof,
                            "sample_id": k,
                            "prompt_name": prompt_name,
                            "continuation": res.get("continuation", ""),
                            "reasoning": res.get("reasoning", ""),
                            "error": None
                        })
                        break
                    except Exception as e:
                        last_err = str(e)
                        retry += 1
                if retry >= 5:
                    out.append({
                        "index": idx,
                        "original_sentence": sentence,
                        "dof_value": dof,
                        "sample_id": k,
                        "prompt_name": prompt_name,
                        "continuation": "",
                        "reasoning": "",
                        "error": f"max_retries_exceeded: {last_err}"
                    })
                pbar.update(1)

    pbar.close()
    return out

# ---------------- runner ----------------
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
):
    print(f"\n{'='*60}")
    print(f"DoF Multi-Sample Generation")
    print(f"Model: {model_type}/{model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Prompt: {prompt_name}")
    print(f"DoFs: {dofs}  |  n_per_dof: {n_per_dof}")
    print(f"Processes: {num_processes}  |  Batch size: {batch_size}")
    print(f"{'='*60}\n")

    print("[1/4] Loading dataset...")
    df = load_dataset(dataset_name, sample_size, target_ids)
    print(f"Loaded {len(df)} sentences")

    # [Optional] Preview: 실제로 모델에 들어갈 렌더링 텍스트 보여주기
    if preview:
        if not (0 <= preview_sentence_index < len(df)):
            preview_sentence_index = 0
        if not (0 <= preview_dof_index < len(dofs)):
            preview_dof_index = 0
        sentence = df.iloc[preview_sentence_index]["sentence"]
        dof_val = dofs[preview_dof_index]
        _, prompt_obj = _build_chain(model_type, model_name, prompt_text)  # prompt 객체만 사용
        rendered = render_preview(prompt_obj, sentence, dof_val, prompt_text)
        print("\n[PREVIEW] ===== Rendered Prompt Sent to Gemini =====")
        print(rendered)
        print("===== /PREVIEW =====================================\n")

    print("[2/4] Building tasks...")
    tasks = [(idx, row["sentence"]) for idx, row in df.iterrows()]
    task_batches = list(chunk_list(tasks, batch_size))

    # output paths (prompt_name 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_BASE / model_type / model_name
    ensure_dir(out_dir)
    jsonl_path = out_dir / f"{model_type}_{model_name}_{dataset_name}_prompt-{prompt_name}_DoF_multi_{timestamp}.jsonl"
    meta_path = out_dir / f"{model_type}_{model_name}_{dataset_name}_prompt-{prompt_name}_DoF_multi_{timestamp}.meta.json"

    print("[3/4] Saving meta...")
    meta = {
        "model_type": model_type,
        "model_name": model_name,
        "dataset": dataset_name,
        "timestamp": timestamp,
        "dofs": dofs,
        "n_per_dof": n_per_dof,
        "num_input_sentences": int(len(df)),
        "total_expected_outputs": int(len(df) * len(dofs) * n_per_dof),
        "batch_size": batch_size,
        "num_processes": num_processes,
        "output_jsonl": str(jsonl_path),
        "prompt_file_used": True,
        "prompt_name": prompt_name,
        "prompt_text_snapshot": prompt_text  # 사용 프롬프트 스냅샷 보관
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[i] Meta saved: {meta_path}")

    print("[4/4] Parallel generation & JSONL streaming...")
    total_written = 0
    with Pool(processes=num_processes) as pool, open(jsonl_path, "a", encoding="utf-8") as fout:
        # 메인 진행바: 배치 단위
        for batch_result in tqdm(
            pool.imap_unordered(
                worker_process_batch,
                [(batch, model_type, model_name, dofs, n_per_dof, prompt_text, prompt_name) for batch in task_batches]
            ),
            total=len(task_batches),
            desc="Generating (batches)",
            position=0,
            leave=True
        ):
            for obj in batch_result:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                total_written += 1

    print("\n[✓] Done.")
    print(f"Output JSONL: {jsonl_path}")
    print(f"Total lines written: {total_written}")
    print(f"Expected: {len(df) * len(dofs) * n_per_dof}")
    return jsonl_path

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="DoF multi-sample generator (API-only, external prompt file, JSONL streaming, prompt-aware filenames, preview)")
    p.add_argument("--model", choices=list(USE_MODEL.keys()), required=True)
    p.add_argument("--model-name", required=True)

    p.add_argument("--dataset", choices=["train","validation","test"], required=True)

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--all", action="store_true", help="Use all sentences in dataset")
    grp.add_argument("--sample", type=int, help="Number of sentences to sample")
    grp.add_argument("--ids", type=str, help="Comma-separated list of index IDs")

    p.add_argument("--dofs", type=str, required=True, help='Comma-separated DoF list, e.g. "0.0,0.5,1.0"')
    p.add_argument("--n-per-dof", type=int, default=30, help="How many samples per DoF for each sentence")

    p.add_argument("--num-processes", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)

    p.add_argument("--prompt-file", type=str, required=True, help="Path to external prompt template (.txt/.tmpl)")
    p.add_argument("--preview", action="store_true", help="Print a rendered prompt preview before running")
    p.add_argument("--preview-sentence-index", type=int, default=0, help="Row index in df (after sampling) to preview")
    p.add_argument("--preview-dof-index", type=int, default=0, help="Index into --dofs list to preview")
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

    prompt_text = load_prompt_file(args.prompt_file)
    prompt_name = get_prompt_name(args.prompt_file)

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
        preview_dof_index=args.preview_dof_index
    )
    print(f"\nOutput file: {out}")

if __name__ == "__main__":
    main()
