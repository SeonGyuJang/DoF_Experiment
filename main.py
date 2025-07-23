'''
# 기본 실행 (DoF 0.5, 10개 샘플)
python main.py --model gemini --model-name gemini-2.0-flash --dataset train --dof 1.0 --sample 10

'''

import argparse
import json
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from multiprocessing import Pool, set_start_method
from datetime import datetime

from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

USE_MODEL: Dict[str, Dict[str, Dict[str, float | int]]] = {
    "gpt": {
        "gpt-3.5-turbo": {"temperature": 1, "max_output_tokens": 4096},
        "gpt-4": {"temperature": 1, "max_output_tokens": 4096},
        "gpt-4o": {"temperature": 1, "max_output_tokens": 4096},
        "gpt-o1": {"temperature": 1, "max_output_tokens": 4096}
    },
    "claude": {
        "claude-4-sonnet": {"temperature": 1, "max_output_tokens": 4096},
        "claude-4-opus": {"temperature": 1, "max_output_tokens": 4096}
    },
    "gemini": {
        "gemini-2.0-flash-lite": {"temperature": 1, "max_output_tokens": 8192},
        "gemini-2.0-flash": {"temperature": 1, "max_output_tokens": 8192},
        "gemini-2.5-flash": {"temperature": 1, "max_output_tokens": 8192},
        "gemini-2.5-pro": {"temperature": 1, "max_output_tokens": 8192}
    },
    "llama": {
        "llama-3-8b": {"temperature": 1, "max_output_tokens": 4096},
        "llama-3-70b": {"temperature": 1, "max_output_tokens": 4096},
        "llama-4-scout": {"temperature": 1, "max_output_tokens": 4096},
}}

BASE = Path(__file__).resolve().parent
DATA_PATHS: Dict[str, Path] = {
    "train": Path(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\train-00000-of-00001.parquet"),
    "validation": Path(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\validation-00000-of-00001.parquet"),
    "test": Path(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\test-00000-of-00001.parquet")
}

RESULTS_BASE = BASE / "results"

def initialize_llm(model_type: str, model_name: str):
    config = USE_MODEL[model_type][model_name]
    
    if model_type == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=config["temperature"],
            max_output_tokens=config["max_output_tokens"]
        )
    # elif model_type == "gpt":
    #     return ChatOpenAI(
    #         model=model_name,
    #         temperature=config["temperature"],
    #         max_tokens=config["max_output_tokens"]
    #     )
    # elif model_type == "claude":
    #     return ChatAnthropic(
    #         model=model_name,
    #         temperature=config["temperature"],
    #         max_tokens=config["max_output_tokens"]
    #     )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

text_generation_prompt = PromptTemplate(
    input_variables=["sentence", "dof_value"],
    template="""
Given text: "{sentence}"

DEGREE OF FREEDOM: {dof_value} (0.0-1.0 scale)
- This parameter controls how extensively you access your internal knowledge networks and creative capabilities
- 0.0 = minimal access 
- 1.0 = maximum access 

Your task is to read the given movie review sentence and generate the continuation.

Return your response in JSON format:
{{
  "continuation": "<your generated continuation>",
  "reasoning": "<brief explanation of your generation choices>"
}}

"""
)

parser = JsonOutputParser()

def load_dataset(dataset_name: str, sample_size: Optional[int] = None, 
                 target_ids: Optional[List[int]] = None) -> pd.DataFrame:
    data_path = DATA_PATHS[dataset_name]
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    df = pd.read_parquet(data_path)
    
    if target_ids:
        df = df[df.index.isin(target_ids)]
    
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    
    return df

def save_results(results: List[Dict[str, Any]], output_dir: Path, 
                 model_type: str, model_name: str, dataset_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_type}_{model_name}_{dataset_name}_{timestamp}.json"
    output_path = output_dir / filename
    
    experiment_info = {
        "model_type": model_type,
        "model_name": model_name,
        "dataset": dataset_name,
        "timestamp": timestamp,
        "num_samples": len(results),
        "results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_info, f, ensure_ascii=False, indent=2)
    
    print(f"[✓] Results saved to: {output_path}")
    return output_path

def generate_continuation(llm, sentence: str, dof_value: float) -> Dict[str, Any]:
    chain = text_generation_prompt | llm | parser
    
    try:
        result = chain.invoke({
            "sentence": sentence,
            "dof_value": dof_value
        })
        return result
    except Exception as e:
        return {
            "continuation": "",
            "reasoning": "",
            "error": str(e)
        }

def worker_process_sample(args):
    idx, sentence, model_type, model_name, dof_value = args
    
    llm = initialize_llm(model_type, model_name)
    generation_result = generate_continuation(llm, sentence, dof_value)
    
    return {
        "index": idx,
        "original_sentence": sentence,
        "dof_value": dof_value,
        "generated_continuation": generation_result.get("continuation", ""),
        "reasoning": generation_result.get("reasoning", ""),
        "error": generation_result.get("error", None)
    }

def run_experiment(model_type: str, model_name: str, dataset_name: str,
                   sample_size: Optional[int] = None, target_ids: Optional[List[int]] = None,
                   dof_value: float = 0.5,
                   use_multiprocessing: bool = False,
                   num_processes: int = 4):
    
    print(f"\n{'='*60}")
    print(f"DoF Text Generation Experiment")
    print(f"Model: {model_type}/{model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Degree of Freedom: {dof_value}")
    print(f"{'='*60}\n")
    
    print("[1/4] Loading dataset...")
    df = load_dataset(dataset_name, sample_size, target_ids)
    print(f"Loaded {len(df)} samples")
    
    print(f"\n[2/4] Initializing {model_type}/{model_name}...")
    
    results = []
    
    if use_multiprocessing:
        print(f"\n[3/4] Generating continuations (multiprocessing with {num_processes} processes)...")
        tasks = [
            (idx, row['sentence'], model_type, model_name, dof_value)
            for idx, row in df.iterrows()
        ]
        
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap_unordered(worker_process_sample, tasks),
                total=len(tasks),
                desc="Processing"
            ))
    else:
        llm = initialize_llm(model_type, model_name)
        print("\n[3/4] Generating continuations...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            sentence = row['sentence']
            generation_result = generate_continuation(llm, sentence, dof_value)
            
            result = {
                "index": idx,
                "original_sentence": sentence,
                "dof_value": dof_value,
                "generated_continuation": generation_result.get("continuation", ""),
                "reasoning": generation_result.get("reasoning", ""),
                "error": generation_result.get("error", None)
            }
            results.append(result)
    
    print("\n[4/4] Saving results...")
    output_dir = RESULTS_BASE / model_type / model_name
    output_path = save_results(results, output_dir, model_type, model_name, dataset_name)
    
    num_errors = sum(1 for r in results if r.get("error"))
    print(f"\n{'='*60}")
    print(f"Experiment completed!")
    print(f"Total samples: {len(results)}")
    print(f"Successful: {len(results) - num_errors}")
    print(f"Errors: {num_errors}")
    print(f"{'='*60}\n")
    
    return output_path

def parse_args():
    parser = argparse.ArgumentParser(description="DoF Text Generation Experiment")
    
    parser.add_argument("--model", choices=list(USE_MODEL.keys()), required=True,
                       help="Model type (gpt, claude, gemini, llama)")
    parser.add_argument("--model-name", required=True,
                       help="Specific model name")
    
    parser.add_argument("--dataset", choices=["train", "validation", "test"], required=True,
                       help="Dataset to use")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true",
                      help="Process all samples in the dataset")
    group.add_argument("--sample", type=int,
                      help="Number of samples to process")
    group.add_argument("--ids", type=str,
                      help="Comma-separated list of sample IDs to process")
    
    parser.add_argument("--dof", type=float, default=0.5,
                       help="Degree of Freedom value (0.0-1.0)")
    parser.add_argument("--multiprocessing", action="store_true",
                       help="Use multiprocessing for parallel execution")
    parser.add_argument("--num-processes", type=int, default=4,
                       help="Number of processes for multiprocessing")
    
    return parser.parse_args()

def main():
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    
    args = parse_args()
    
    if args.model not in USE_MODEL:
        print(f"Error: Unknown model type '{args.model}'")
        return
    
    if args.model_name not in USE_MODEL[args.model]:
        print(f"Error: Unknown model name '{args.model_name}' for {args.model}")
        print(f"Available models: {list(USE_MODEL[args.model].keys())}")
        return
    
    sample_size = None
    target_ids = None
    
    if args.sample:
        sample_size = args.sample
    elif args.ids:
        target_ids = [int(x.strip()) for x in args.ids.split(",")]
    
    try:
        run_experiment(
            model_type=args.model,
            model_name=args.model_name,
            dataset_name=args.dataset,
            sample_size=sample_size,
            target_ids=target_ids,
            dof_value=args.dof,
            use_multiprocessing=args.multiprocessing,
            num_processes=args.num_processes
        )
    except Exception as e:
        print(f"\nError during experiment: {e}")
        raise

if __name__ == "__main__":
    main()