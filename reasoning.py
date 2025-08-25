import json
from pathlib import Path

def extract_reasonings(input_path: str, output_path: str):
    input_file = Path(input_path)
    output_file = Path(output_path)

    # 저장할 디렉토리가 없으면 생성
    output_file.parent.mkdir(parents=True, exist_ok=True)

    reasonings = []

    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if "reasoning" in data:
                reasonings.append(data["reasoning"])

    # reasoning만 줄 단위로 저장
    with output_file.open("w", encoding="utf-8") as f:
        for reasoning in reasonings:
            f.write(reasoning + "\n")

    print(f"✅ {len(reasonings)}개의 reasoning을 '{output_path}'에 저장했습니다.")

# 실행 예시
extract_reasonings(
    r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\results\gemini\gemini-2.0-flash\essay\prompt8\gemini_gemini-2.0-flash_prompt-dof_prompt8_20250822_194232.jsonl",
    r"C:\Users\dsng3\Documents\GitHub\DoF_Experiment\Reasoning\Task8_reasonings.txt"
)
