# python main.py --model gemini --model-name gemini-2.0-flash --dataset train --sample 1000 --dofs 0.0,0.5,1.0 --n-per-dof 10 --num-processes 8 --batch-size 4 --prompt-file prompts/exp1.tmpl --preview

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

def load_dataset(
    dataset_name: str,
    sample_size: Optional[int] = None,
    target_ids: Optional[List[int]] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    - sample_size가 주어지면 random_state=seed로 재현 가능한 샘플링 수행
    - 샘플링 후 index 오름차순 정렬로 순서를 고정
    - target_ids가 주어지면 해당 index만 선택 (이 경우도 정렬 유지)
    """
    data_path = DATA_PATHS[dataset_name]
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    df = pd.read_parquet(data_path)

    if target_ids:
        df = df[df.index.isin(target_ids)]

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)

    df = df.sort_index()
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

# ---------------- resume helpers ----------------
def key_of(index: int, dof_value: float, sample_id: int) -> Tuple[int, str, int]:
    """정확한 매칭을 위해 DoF를 고정 소수(6) 문자열로 보관."""
    return (int(index), f"{float(dof_value):.6f}", int(sample_id))

def parse_result_key(obj: Dict[str, Any]) -> Tuple[int, str, int]:
    return key_of(int(obj["index"]), float(obj["dof_value"]), int(obj["sample_id"]))

def load_existing_results(jsonl_path: Path) -> Tuple[Dict[Tuple[int,str,int], Dict[str,Any]], set, set]:
    """
    기존 JSONL을 읽어 전체 map과 성공/오류 key 집합을 반환.
    성공 판단: error 필드가 비어있고(또는 없음) + continuation이 비어있지 않음.
    """
    results_map: Dict[Tuple[int,str,int], Dict[str,Any]] = {}
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
                    # 내용 없음은 오류로 취급
                    error_keys.add(k)
            else:
                error_keys.add(k)

    return results_map, success_keys, error_keys

# ---------------- chain builder & preview ----------------
def _build_chain(model_type: str, model_name: str, prompt_text: str):
    """Create LLM chain: PromptTemplate -> LLM -> JSON parser"""
    llm = initialize_llm(model_type, model_name)
    prompt = PromptTemplate.from_template(prompt_text)
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    return chain, prompt

def render_preview(prompt: PromptTemplate, sentence: str, dof_value: float, prompt_text: str) -> str:
    """실제 모델에 들어갈 최종 텍스트(포맷된 프롬프트) 미리보기"""
    use_nonce = "{nonce}" in prompt_text
    kwargs = {"sentence": sentence, "dof_value": dof_value}
    if use_nonce:
        kwargs["nonce"] = 123456789  # 미리보기용 고정 nonce
    return prompt.format(**kwargs)

# ---------------- worker (item-level) ----------------
def worker_process_batch(args):
    """
    args: (batch_items, model_type, model_name, prompt_text, prompt_name, seed)
    batch_items: List[Tuple[int, str, float, int]]  -> (index, sentence, dof_value, sample_id)
    """
    (batch_items, model_type, model_name, prompt_text, prompt_name, seed) = args

    # Worker ID (1..num_processes) -> 진행바 position
    try:
        wid = current_process()._identity[0]
    except Exception:
        wid = 1
    wid_pos = wid if wid >= 1 else 1

    # 워커별 결정적 난수
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

    for idx, sentence, dof, sample_id in batch_items:
        payload = {"sentence": sentence, "dof_value": dof}
        if use_nonce:
            payload["nonce"] = random.getrandbits(64)

        retry = 0
        last_err = None
        # 여기서 "최대 5회까지 해당 항목만 재시도" 보장
        while retry < 5:
            try:
                res = chain.invoke(payload)
                out.append({
                    "index": idx,
                    "original_sentence": sentence,
                    "dof_value": dof,
                    "sample_id": sample_id,
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
                "sample_id": sample_id,
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
    seed: int = 42,
    resume_from: Optional[Path] = None,
    rerun_errors_only: bool = False,
    output_path: Optional[Path] = None
):
    # 전역 재현성
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

    print("[1/5] Loading dataset...")
    df = load_dataset(dataset_name, sample_size, target_ids, seed=seed)
    print(f"Loaded {len(df)} sentences")

    # [Optional] Preview
    if preview:
        if not (0 <= preview_sentence_index < len(df)):
            preview_sentence_index = 0
        if not (0 <= preview_dof_index < len(dofs)):
            preview_dof_index = 0
        sentence = df.iloc[preview_sentence_index]["sentence"]
        dof_val = dofs[preview_dof_index]
        _, prompt_obj = _build_chain(model_type, model_name, prompt_text)
        rendered = render_preview(prompt_obj, sentence, dof_val, prompt_text)
        print("\n[PREVIEW] ===== Rendered Prompt Sent to Gemini =====")
        print(rendered)
        print("===== /PREVIEW =====================================\n")

    print("[2/5] Building work items...")
    # 전체 아이템: 문장 × DoF × sample_id
    items: List[Tuple[int, str, float, int]] = []
    for idx, row in df.iterrows():
        s = row["sentence"]
        for d in dofs:
            for k in range(n_per_dof):
                items.append((int(idx), s, float(d), int(k)))

    # Resume 모드 처리
    existing_map = {}
    success_keys = set()
    error_keys = set()
    if resume_from:
        if not Path(resume_from).exists():
            raise FileNotFoundError(f"--resume-from not found: {resume_from}")
        existing_map, success_keys, error_keys = load_existing_results(Path(resume_from))

        if rerun_errors_only:
            # 오류난 key만 재실행
            items = [it for it in items if key_of(it[0], it[2], it[3]) in error_keys]
        else:
            # 성공 제외(오류 + 누락)만 재실행
            done = success_keys
            items = [it for it in items if key_of(it[0], it[2], it[3]) not in done]

    total_expected = len(df) * len(dofs) * n_per_dof
    total_pending = len(items)
    print(f"[i] Total expected outputs: {total_expected}")
    if resume_from:
        print(f"[i] Existing file entries: {len(existing_map)}  (success: {len(success_keys)}, errors: {len(error_keys)})")
    print(f"[i] Pending items to run:   {total_pending}")

    # 출력 경로 구성
    if output_path:
        final_jsonl_path = Path(output_path)
        ensure_dir(final_jsonl_path.parent)
    else:
        if resume_from:
            final_jsonl_path = Path(resume_from)  # 덮어쓰기(원자적 교체)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = RESULTS_BASE / model_type / model_name
            ensure_dir(out_dir)
            final_jsonl_path = out_dir / f"{model_type}_{model_name}_{dataset_name}_prompt-{prompt_name}_DoF_multi_{timestamp}.jsonl"

    # 메타 파일 경로
    meta_path = final_jsonl_path.with_suffix(".meta.json")

    print("[3/5] Saving meta (initial snapshot)...")
    meta = {
        "model_type": model_type,
        "model_name": model_name,
        "dataset": dataset_name,
        "dofs": dofs,
        "n_per_dof": n_per_dof,
        "num_input_sentences": int(len(df)),
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

    print("[4/5] Parallel generation (item-level) ...")
    new_results: List[Dict[str, Any]] = []
    total_written = 0

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
                new_results.extend(batch_result)
                total_written += len(batch_result)

    print("\n[5/5] Consolidating & writing results...")
    # 기존 map에 신규 결과 덮어쓰기(동일 key 교체)
    if resume_from:
        for obj in new_results:
            existing_map[parse_result_key(obj)] = obj
        # 안정된 정렬( index, dof_str, sample_id )
        all_items_sorted = sorted(existing_map.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))
        # 원자적 교체를 위해 tmp 파일로 쓴 뒤 교체
        tmp_path = final_jsonl_path.with_suffix(final_jsonl_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fout:
            for _, obj in all_items_sorted:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        os.replace(tmp_path, final_jsonl_path)
    else:
        # 신규 실행이면 스트리밍이 아니라 한 번에 기록
        ensure_dir(final_jsonl_path.parent)
        with open(final_jsonl_path, "w", encoding="utf-8") as fout:
            for obj in new_results:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("\n[✓] Done.")
    print(f"Output JSONL: {final_jsonl_path}")
    print(f"New lines written this run: {total_written}")
    if resume_from:
        print(f"Total lines in consolidated file: {len(existing_map)}")
    else:
        print(f"Expected: {total_expected}")
    return str(final_jsonl_path)

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="DoF multi-sample generator (API-only, external prompt file, JSONL streaming, prompt-aware filenames, preview, reproducible sampling, resume)"
    )
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
    p.add_argument("--batch-size", type=int, default=2, help="Batch size in ITEMS (sentence×DoF×sample_id)")

    p.add_argument("--prompt-file", type=str, required=True, help="Path to external prompt template (.txt/.tmpl)")
    p.add_argument("--preview", action="store_true", help="Print a rendered prompt preview before running")
    p.add_argument("--preview-sentence-index", type=int, default=0, help="Row index in df (after sampling) to preview")
    p.add_argument("--preview-dof-index", type=int, default=0, help="Index into --dofs list to preview")

    # 재현성 보장용 시드
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible input sampling and nonce generation")

    # Resume & Output
    p.add_argument("--resume-from", type=str, help="Path to an existing JSONL to resume (run only missing or failed items)")
    p.add_argument("--rerun-errors-only", action="store_true", help="When resuming, re-run only rows that had non-empty error")
    p.add_argument("--output", type=str, help="Optional explicit JSONL output path. If omitted and --resume-from is set, the original file is overwritten atomically; otherwise a timestamped file is created.")
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