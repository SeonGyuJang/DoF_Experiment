# main.py
import argparse
import json
import os
import sys
import uuid
import threading
from pathlib import Path
from typing import List, Tuple, Optional
from multiprocessing import Pool, set_start_method, current_process, Manager
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# =========================================================
# Config
# =========================================================
load_dotenv()

USE_MODEL = {
    "gemini": {
        "gemini-2.0-flash-lite": {"temperature": 1.0, "max_output_tokens": 8192},
        "gemini-2.0-flash": {"temperature": 1.0, "max_output_tokens": 8192},
        "gemini-2.5-flash": {"temperature": 1.0, "max_output_tokens": 8192},
        "gemini-2.5-pro": {"temperature": 1.0, "max_output_tokens": 8192},
    }
}

PROMPT = PromptTemplate(
    input_variables=["sentence", "dof_value"],
    template="""
DoF={dof_value} is a control value for the fidelity–novelty trade-off.
Internally choose a continuation that minimizes:
Loss = (1-DoF)*DistanceFromInput + DoF*NoveltyGain
Implement this privately; do not expose the process.

Input: "{sentence}"

Return JSON only:
{{
  "continuation": "<continuation>",
  "reasoning": "<how the DoF-weighted trade-off guided the result>"
}}
""".strip(),
)

PARSER = JsonOutputParser()

# =========================================================
# Helpers
# =========================================================
def init_llm(model_type: str, model_name: str, api_key: Optional[str], candidate_count: int):
    cfg = USE_MODEL[model_type][model_name]
    api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY (or GOOGLE_GENERATIVE_AI_API_KEY). "
            "Set it in .env or environment variables."
        )
    return ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=cfg["temperature"],
        max_output_tokens=cfg["max_output_tokens"],
        model_kwargs={"candidate_count": max(1, int(candidate_count))},
        max_retries=0,  # 재시도는 우리가 수동으로
    )

def chunk_list(lst: List, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_lines_to_file(src_paths: List[Path], dst_path: Path):
    with dst_path.open("w", encoding="utf-8") as out_f:
        for p in src_paths:
            with p.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)

def call_with_timeout(fn, kwargs, timeout: float):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, kwargs)
        return fut.result(timeout=timeout)

# =========================================================
# Worker
# =========================================================
def worker_run(args):
    (
        batch_id,
        batch,                # List[Tuple[int, str]]
        model_type,
        model_name,
        dofs,
        samples_per_dof,
        retries,
        shard_dir,
        request_timeout,
        api_key,
        candidate_count,
        progress_queue,       # multiprocessing.Manager().Queue()
    ) = args

    proc = current_process().name.replace(" ", "_")
    shard_path = shard_dir / f"shard_{batch_id}_{proc}_{uuid.uuid4().hex}.jsonl"

    llm = init_llm(model_type, model_name, api_key, candidate_count)
    chain = PROMPT | llm | PARSER
    meta_now = datetime.utcnow().isoformat() + "Z"

    with shard_path.open("w", encoding="utf-8") as f:
        for idx, sentence in batch:
            for dof in dofs:
                for k in range(1, samples_per_dof + 1):
                    attempt = 0
                    record = {
                        "index": int(idx),
                        "original_sentence": sentence,
                        "dof_value": float(dof),
                        "sample_id": int(k),
                        "model_type": model_type,
                        "model_name": model_name,
                        "timestamp": meta_now,
                        "continuation": "",
                        "reasoning": "",
                        "error": None,
                        "retry_count": 0,
                    }
                    while attempt <= retries:
                        try:
                            res = call_with_timeout(
                                chain.invoke,
                                {"sentence": sentence, "dof_value": dof},
                                timeout=request_timeout,
                            )
                            record["continuation"] = res.get("continuation", "")
                            record["reasoning"] = res.get("reasoning", "")
                            record["retry_count"] = attempt
                            break
                        except FuturesTimeout:
                            attempt += 1
                            record["retry_count"] = attempt
                            record["error"] = f"timeout>{request_timeout}s"
                        except Exception as e:
                            attempt += 1
                            record["retry_count"] = attempt
                            record["error"] = str(e)
                        if attempt > retries:
                            pass

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    # 진행 1칸
                    try:
                        progress_queue.put((batch_id, 1))
                    except Exception:
                        pass

    # 배치 종료 알림
    try:
        progress_queue.put((batch_id, "DONE"))
    except Exception:
        pass

    return str(shard_path)

# =========================================================
# Main
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description="DoF JSONL Generator (multi-progress)")
    p.add_argument("--model", default="gemini", choices=list(USE_MODEL.keys()))
    p.add_argument("--model-name", required=True, help="e.g., gemini-2.0-flash")
    p.add_argument("--data-path", required=True, help="Parquet file with column 'sentence'")
    p.add_argument("--output", required=True, help="Output JSONL file path")
    p.add_argument("--dofs", default="0,0.5,1", help="Comma-separated DoF values")
    p.add_argument("--samples-per-dof", type=int, default=50)
    p.add_argument("--limit", type=int, default=100, help="Max # of input sentences (start small!)")
    p.add_argument("--processes", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2, help="Input sentences per worker batch")
    p.add_argument("--max-retries", type=int, default=3, help="per-call retries on error/timeout")
    p.add_argument("--request-timeout", type=float, default=30.0, help="seconds per request timeout")
    p.add_argument("--api-key", type=str, default=None, help="explicit GOOGLE_API_KEY (optional)")
    p.add_argument("--candidate-count", type=int, default=1, help="per-call candidates if supported")
    p.add_argument("--max-bars", type=int, default=32, help="max number of batch bars to render")
    return p.parse_args()

def main():
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    args = parse_args()

    if args.model not in USE_MODEL or args.model_name not in USE_MODEL[args.model]:
        print(f"[!] Unknown model {args.model}/{args.model_name}", file=sys.stderr)
        print(f"    Available for {args.model}: {list(USE_MODEL.get(args.model, {}).keys())}", file=sys.stderr)
        sys.exit(1)

    # 데이터
    df = pd.read_parquet(args.data_path)
    if "sentence" not in df.columns:
        print("[!] Input parquet must have a 'sentence' column.", file=sys.stderr)
        sys.exit(1)
    df = df.head(args.limit).reset_index(drop=False)
    tasks: List[Tuple[int, str]] = [(int(row["index"]), str(row["sentence"])) for _, row in df.iterrows()]

    dofs = [float(x.strip()) for x in args.dofs.split(",") if x.strip() != ""]
    est_total = args.limit * len(dofs) * args.samples_per_dof
    if est_total > 100000:
        print(f"[!] You are about to request ~{est_total:,} generations. This may take a very long time and cost money.", file=sys.stderr)

    # 배치 나누기
    batches = list(chunk_list(tasks, args.batch_size))
    num_batches = len(batches)

    # 경로/샤드
    output_path = Path(args.output).resolve()
    ensure_dir(output_path.parent)
    shard_dir = output_path.parent / f"_shards_{uuid.uuid4().hex}"
    ensure_dir(shard_dir)

    print("\n===============================================")
    print("DoF JSONL Generation")
    print(f"Model          : {args.model}/{args.model_name}")
    print(f"Inputs         : {len(tasks)} sentences (limit={args.limit})")
    print(f"DoFs           : {dofs}")
    print(f"Samples/DoF    : {args.samples_per_dof}")
    print(f"Processes      : {args.processes}, Batch size: {args.batch_size}")
    print(f"Timeout/Retry  : {args.request_timeout}s per call, max_retries={args.max_retries}")
    print(f"CandidateCount : {args.candidate_count}")
    print(f"Estimated Gens : ~{est_total:,} lines")
    print(f"Output         : {output_path}")
    print("===============================================\n")

    # ---- 진행률 바 준비 (배치별) ----
    # 각 배치 총 작업량 = len(batch) * len(dofs) * samples_per_dof
    batch_totals = [len(b) * len(dofs) * args.samples_per_dof for b in batches]

    # 너무 많은 배치 바 생성을 방지하기 위해 상한
    max_bars = max(1, int(args.max_bars))
    show_all = num_batches <= max_bars

    manager = Manager()
    progress_queue = manager.Queue()

    # tqdm 바들 생성
    bars = []
    if show_all:
        for i in range(num_batches):
            bars.append(
                tqdm(
                    total=batch_totals[i],
                    position=i,
                    leave=True,
                    desc=f"Batch {i+1}/{num_batches}",
                    dynamic_ncols=True,
                )
            )
    else:
        # 앞쪽 max_bars-1 개는 개별, 나머지는 'Others'
        for i in range(max_bars - 1):
            bars.append(
                tqdm(
                    total=batch_totals[i],
                    position=i,
                    leave=True,
                    desc=f"Batch {i+1}/{num_batches}",
                    dynamic_ncols=True,
                )
            )
        # Others bar
        others_total = sum(batch_totals[max_bars - 1 :])
        bars.append(
            tqdm(
                total=others_total,
                position=max_bars - 1,
                leave=True,
                desc=f"Batch {max_bars}-{num_batches} (Others)",
                dynamic_ncols=True,
            )
        )

    # 큐 리스너 스레드: 워커가 올리는 진행 이벤트를 수신 → 막대 업데이트
    done_batches = set()

    def listener():
        nonlocal done_batches
        while len(done_batches) < num_batches:
            msg = progress_queue.get()
            if not isinstance(msg, tuple) or len(msg) != 2:
                continue
            b_id, inc = msg
            if inc == "DONE":
                done_batches.add(b_id)
                # 바를 완료 지점으로 이동
                target_idx = b_id if show_all or b_id < max_bars - 1 else max_bars - 1
                remaining = bars[target_idx].total - bars[target_idx].n
                if remaining > 0:
                    bars[target_idx].update(remaining)
                continue
            # 증가 1칸
            target_idx = b_id if show_all or b_id < max_bars - 1 else max_bars - 1
            bars[target_idx].update(int(inc))

    listener_thread = threading.Thread(target=listener, daemon=True)
    listener_thread.start()

    # ---- 멀티프로세싱 실행 ----
    pool_args = [
        (
            i,                    # batch_id
            batch,                # batch
            args.model,
            args.model_name,
            dofs,
            args.samples_per_dof,
            args.max_retries,
            shard_dir,
            args.request_timeout,
            args.api_key,
            args.candidate_count,
            progress_queue,
        )
        for i, batch in enumerate(batches)
    ]

    shard_paths: List[str] = []
    with Pool(processes=args.processes) as pool:
        # 여기 tqdm은 "배치 단위 완료" 진행률(막대 하나)만 표시
        for shard in tqdm(
            pool.imap_unordered(worker_run, pool_args),
            total=len(pool_args),
            desc="Batches completed",
            position=(max_bars if show_all else max_bars),
            leave=True,
            dynamic_ncols=True,
        ):
            shard_paths.append(shard)

    # 워커 종료 후 리스너 종료 대기
    listener_thread.join(timeout=5.0)
    for b in bars:
        b.close()

    # 머지
    shard_paths_sorted = sorted([Path(p) for p in shard_paths], key=os.path.getmtime)
    write_lines_to_file(shard_paths_sorted, output_path)

    # 정리
    for p in shard_paths_sorted:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    try:
        Path(shard_dir).rmdir()
    except Exception:
        pass

    total_lines = sum(1 for _ in open(output_path, "r", encoding="utf-8"))
    print("\n===============================================")
    print("Generation completed!")
    print(f"JSONL saved to : {output_path}")
    print(f"Total lines    : {total_lines}")
    print("Each line = one (sentence, DoF, sample_id) generation record.")
    print("===============================================\n")

if __name__ == "__main__":
    main()
