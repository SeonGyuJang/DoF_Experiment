import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# 데이터 경로 설정
DATA_PATHS = {
    "train": Path(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\essay\training_set_rel3.tsv"),
    "test": Path(r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\data\essay\valid_set.tsv")
}

def load_dataset(dataset_name: str, essay_set: int = None):
    """데이터셋 로드"""
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
            raise RuntimeError(f"Failed to read file at {data_path}. Error: {repr(e)}, {repr(e2)}")
    
    print(f"Original dataset size: {len(df)} rows")
    print(f"Available essay_sets: {sorted(df['essay_set'].unique()) if 'essay_set' in df.columns else 'No essay_set column'}")
    
    # essay_set 필터링
    if essay_set is not None:
        if 'essay_set' not in df.columns:
            raise KeyError("Dataset does not contain 'essay_set' column")
        original_size = len(df)
        df = df[df['essay_set'] == essay_set]
        print(f"Filtered to essay_set={essay_set}: {len(df)} rows (from {original_size})")
        if len(df) == 0:
            raise ValueError(f"No data found for essay_set={essay_set}")
    
    return df

def analyze_essay_lengths(df: pd.DataFrame, essay_set: int = None):
    """에세이 길이 분석"""
    
    if 'essay' not in df.columns:
        raise KeyError("Dataset must contain 'essay' column")
    
    # 에세이 텍스트에서 길이 계산
    essays = df['essay'].astype(str)
    
    # 다양한 길이 측정
    char_lengths = essays.str.len()
    word_lengths = essays.str.split().str.len()
    
    # 통계 계산
    char_stats = {
        'mean': char_lengths.mean(),
        'median': char_lengths.median(),
        'std': char_lengths.std(),
        'min': char_lengths.min(),
        'max': char_lengths.max(),
        'q25': char_lengths.quantile(0.25),
        'q75': char_lengths.quantile(0.75)
    }
    
    word_stats = {
        'mean': word_lengths.mean(),
        'median': word_lengths.median(),
        'std': word_lengths.std(),
        'min': word_lengths.min(),
        'max': word_lengths.max(),
        'q25': word_lengths.quantile(0.25),
        'q75': word_lengths.quantile(0.75)
    }
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"Essay Length Analysis")
    if essay_set is not None:
        print(f"Essay Set: {essay_set}")
    print(f"Total Essays: {len(df)}")
    print(f"{'='*60}")
    
    print(f"\n📊 Character Length Statistics:")
    print(f"  Mean:     {char_stats['mean']:.1f} characters")
    print(f"  Median:   {char_stats['median']:.1f} characters")
    print(f"  Std Dev:  {char_stats['std']:.1f} characters")
    print(f"  Min:      {char_stats['min']:.0f} characters")
    print(f"  Max:      {char_stats['max']:.0f} characters")
    print(f"  Q25:      {char_stats['q25']:.1f} characters")
    print(f"  Q75:      {char_stats['q75']:.1f} characters")
    
    print(f"\n📝 Word Count Statistics:")
    print(f"  Mean:     {word_stats['mean']:.1f} words")
    print(f"  Median:   {word_stats['median']:.1f} words")
    print(f"  Std Dev:  {word_stats['std']:.1f} words")
    print(f"  Min:      {word_stats['min']:.0f} words")
    print(f"  Max:      {word_stats['max']:.0f} words")
    print(f"  Q25:      {word_stats['q25']:.1f} words")
    print(f"  Q75:      {word_stats['q75']:.1f} words")
    
    # 길이 분포 분석
    print(f"\n📈 Length Distribution:")
    
    # 단어 수 기준 분포
    word_bins = [0, 50, 100, 150, 200, 250, 300, 400, 500, float('inf')]
    word_labels = ['0-50', '51-100', '101-150', '151-200', '201-250', '251-300', '301-400', '401-500', '500+']
    word_distribution = pd.cut(word_lengths, bins=word_bins, labels=word_labels, right=False)
    word_counts = word_distribution.value_counts().sort_index()
    
    print("  Word Count Distribution:")
    for label, count in word_counts.items():
        percentage = (count / len(df)) * 100
        print(f"    {label:>8} words: {count:>4} essays ({percentage:>5.1f}%)")
    
    # 에세이 셋별 통계 (전체 데이터인 경우)
    if essay_set is None and 'essay_set' in df.columns:
        print(f"\n📋 Statistics by Essay Set:")
        essay_set_stats = df.groupby('essay_set')['essay'].agg([
            ('count', 'count'),
            ('avg_chars', lambda x: x.str.len().mean()),
            ('avg_words', lambda x: x.str.split().str.len().mean())
        ]).round(1)
        
        print(f"{'Set':>3} | {'Count':>5} | {'Avg Chars':>9} | {'Avg Words':>9}")
        print(f"{'-'*3}-+-{'-'*5}-+-{'-'*9}-+-{'-'*9}")
        for set_num, row in essay_set_stats.iterrows():
            print(f"{set_num:>3} | {row['count']:>5.0f} | {row['avg_chars']:>9.1f} | {row['avg_words']:>9.1f}")
    
    return {
        'char_stats': char_stats,
        'word_stats': word_stats,
        'total_essays': len(df)
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze essay length statistics")
    parser.add_argument("--dataset", choices=["train", "test"], default="train", 
                       help="Dataset to analyze (default: train)")
    parser.add_argument("--essay-set", type=int, 
                       help="Specific essay set to analyze (if not specified, analyzes all)")
    parser.add_argument("--show-examples", action="store_true", 
                       help="Show examples of shortest and longest essays")
    
    args = parser.parse_args()
    
    try:
        # 데이터 로드
        df = load_dataset(args.dataset, args.essay_set)
        
        # 길이 분석
        stats = analyze_essay_lengths(df, args.essay_set)
        
        # 예시 보기 (옵션)
        if args.show_examples:
            print(f"\n📄 Essay Examples:")
            essays = df['essay'].astype(str)
            word_lengths = essays.str.split().str.len()
            
            # 가장 짧은 에세이
            shortest_idx = word_lengths.idxmin()
            shortest_essay = essays.loc[shortest_idx]
            shortest_words = word_lengths.loc[shortest_idx]
            
            print(f"\n🔍 Shortest Essay ({shortest_words} words):")
            print(f"Essay ID: {df.loc[shortest_idx, 'essay_id'] if 'essay_id' in df.columns else 'N/A'}")
            print(f"Text: {shortest_essay[:200]}{'...' if len(shortest_essay) > 200 else ''}")
            
            # 가장 긴 에세이
            longest_idx = word_lengths.idxmax()
            longest_essay = essays.loc[longest_idx]
            longest_words = word_lengths.loc[longest_idx]
            
            print(f"\n🔍 Longest Essay ({longest_words} words):")
            print(f"Essay ID: {df.loc[longest_idx, 'essay_id'] if 'essay_id' in df.columns else 'N/A'}")
            print(f"Text: {longest_essay[:200]}{'...' if len(longest_essay) > 200 else ''}")
        
        # 요약
        word_mean = stats['word_stats']['mean']
        char_mean = stats['char_stats']['mean']
        
        print(f"\n🎯 Summary:")
        print(f"  Average essay length: {word_mean:.1f} words ({char_mean:.0f} characters)")
        print(f"  Total essays analyzed: {stats['total_essays']:,}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())