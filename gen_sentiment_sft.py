"""
gen_sentiment_sft.py
从 ModelScope 下载 Financial PhraseBank，转换为 MiniMind SFT 格式
输出：/root/minimind/dataset/sft_sentiment.jsonl
"""

import json
from modelscope.msdatasets import MsDataset

OUTPUT_PATH = "/root/minimind/dataset/sft_sentiment.jsonl"
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

def build_conversation(sentence, label_str):
    return {
        "conversations": [
            {
                "role": "user",
                "content": (
                    "What is the sentiment of this financial news sentence? "
                    f"Sentence: '{sentence}' "
                    "Answer in one word: positive, negative, or neutral."
                )
            },
            {
                "role": "assistant",
                "content": label_str
            }
        ]
    }

def main():
    print("正在从 ModelScope 下载数据集...")
    ds = MsDataset.load("modelgod2025/financial_phrasebank_50agree", split="train")

    print(f"下载完成，共 {len(ds)} 条数据")
    print("示例数据：", ds[0])

    count = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in ds:
            sentence = item["sentence"].strip()
            label_int = item["label"]
            label_str = LABEL_MAP.get(label_int)

            if not sentence or label_str is None:
                continue

            record = build_conversation(sentence, label_str)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"转换完成，共写入 {count} 条，路径：{OUTPUT_PATH}")

if __name__ == "__main__":
    main()
