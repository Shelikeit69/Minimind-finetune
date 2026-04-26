"""
eval_sentiment.py
评估情感分析模型的 accuracy 和 macro F1
"""

import json
import random
import torch
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM, MiniMindConfig

WEIGHT_PATH = "/root/minimind/out/sentiment_sft_768.pth"
DATA_PATH = "/root/minimind/dataset/sft_sentiment.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER_PATH = "/root/minimind/model"
SAMPLES_PER_CLASS = 50  # 每类各取50条，共150条
RANDOM_SEED = 42

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
    state = torch.load(WEIGHT_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE)
    model.eval()
    return model, tokenizer

def predict(model, tokenizer, sentence):
    prompt = (
        "What is the sentiment of this financial news sentence? "
        f"Sentence: '{sentence}' "
        "Answer in one word: positive, negative, or neutral."
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            max_new_tokens=5,
            temperature=0.1,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    pred = decoded.strip().lower().split()[0] if decoded.strip() else "neutral"
    pred = pred.strip(".,!?")
    if pred not in LABEL2ID:
        pred = "neutral"
    return pred

def main():
    print("加载模型...")
    model, tokenizer = load_model()

    print("读取并分层抽样数据集...")
    buckets = {"negative": [], "neutral": [], "positive": []}
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            convs = item["conversations"]
            user_text = convs[0]["content"]
            label = convs[1]["content"].strip().lower()
            sentence = user_text.split("Sentence: '")[1].split("' Answer")[0]
            if label in buckets:
                buckets[label].append((sentence, label))

    random.seed(RANDOM_SEED)
    samples = []
    for label, items in buckets.items():
        chosen = random.sample(items, min(SAMPLES_PER_CLASS, len(items)))
        samples.extend(chosen)
        print(f"  {label}: {len(chosen)} 条（共 {len(items)} 条）")
    random.shuffle(samples)

    print(f"\n开始推理，共 {len(samples)} 条...")
    y_true, y_pred = [], []
    for i, (sentence, true_label) in enumerate(samples):
        pred = predict(model, tokenizer, sentence)
        y_true.append(true_label)
        y_pred.append(pred)
        if (i + 1) % 30 == 0:
            print(f"进度：{i+1}/{len(samples)}")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", labels=["negative", "neutral", "positive"])
    print(f"\n===== 评估结果 =====")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Macro F1:   {f1:.4f}")
    print(f"\n详细报告：")
    print(classification_report(y_true, y_pred, labels=["negative", "neutral", "positive"]))

if __name__ == "__main__":
    main()
