"""
bert_end_to_end.py

- (옵션) Google Drive에서 bert_sentiment.zip 있으면 unzip 해서 사용
- ./bert_sentiment 디렉토리가 없으면:
    1) pip install (transformers, datasets, accelerate, scikit-learn)
    2) comments_labeled_for_training.csv 로 XLM-RoBERTa 학습
    3) ./bert_sentiment 에 저장
- 최종적으로 comments_for_inference.csv 에 대해 inference + 분포 그래프 + 대표 문장 출력
"""

import os
import zipfile
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# -------------------------
# 기본 설정
# -------------------------
MODEL_DIR = "./bert_sentiment"
TEXT_COL = "text"  # 학습/추론 공통 text 컬럼명
LABEL_COL = "label"

# Colab + Google Drive에서 zip 사용할 경우 경로
DRIVE_ZIP_PATH = "/content/drive/MyDrive/bert_sentiment.zip"
DRIVE_MOUNT_POINT = "/content/drive"

FOR_LEARNING_CSV = "comments_labeled_for_training.csv"
INFERENCE_CSV = "comments_for_inference.csv"

label2id = {"neg": 0, "neu": 1, "pos": 2}
id2label = {v: k for k, v in label2id.items()}


# -------------------------
# 유틸 함수들
# -------------------------
def ensure_packages_installed():
    """필요한 라이브러리 없으면 pip 로 설치"""
    try:
        import transformers  # noqa
        import datasets      # noqa
        import sklearn       # noqa
    except ImportError:
        print("[INFO] Installing transformers, datasets, accelerate, scikit-learn ...")
        subprocess.check_call([
            sys.executable,
            "-m", "pip",
            "install",
            "-U",
            'transformers[torch]',
            "datasets",
            "accelerate",
            "scikit-learn",
        ])


def maybe_mount_drive_and_unzip():
    """
    Google Drive 에서 bert_sentiment.zip 이 있으면 unzip 해서 MODEL_DIR 생성.
    여기서는 drive.mount()를 절대 호출하지 않고,
    사용자가 노트북에서 미리 mount했다고 가정한다.
    """
    if os.path.isdir(MODEL_DIR):
        print(f"[INFO] MODEL_DIR already exists: {MODEL_DIR}")
        return

    # Drive가 mount 되어 있는지 체크만 하고, 아니면 스킵
    if not os.path.ismount(DRIVE_MOUNT_POINT):
        print("[INFO] Drive is not mounted. If you need bert_sentiment.zip,")
        print("       먼저 노트북 셀에서 다음을 실행하세요:")
        print("         from google.colab import drive")
        print("         drive.mount('/content/drive')")
        return

    if not os.path.exists(DRIVE_ZIP_PATH):
        print(f"[INFO] No zip found at {DRIVE_ZIP_PATH}. Will train model later if needed.")
        return

    print(f"[INFO] Found zip: {DRIVE_ZIP_PATH}, extracting to /content ...")
    with zipfile.ZipFile(DRIVE_ZIP_PATH, 'r') as zf:
        zf.extractall("/content")
    print("[INFO] Unzip done.")

    if os.path.isdir(MODEL_DIR):
        print(f"[INFO] bert_sentiment folder exists at: {MODEL_DIR}")
        print("       Files:", os.listdir(MODEL_DIR))
    else:
        print("[WARN] bert_sentiment folder not found after unzip. Check zip structure.")



def train_if_needed():
    """
    MODEL_DIR 이:
      - 없으면: 새로 학습해서 저장
      - 있더라도: tokenizer / model 이 정상 로드되는지 확인.
                   로드 실패하면 기존 폴더 삭제 후 다시 학습.
    """
    # 패키지 설치 (없으면만 설치)
    ensure_packages_installed()

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    from transformers import TrainingArguments, Trainer
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    # (1) MODEL_DIR 존재할 경우, 먼저 "정상 로드 가능한지" 검사
    if os.path.isdir(MODEL_DIR):
        print(f"[INFO] MODEL_DIR exists: {MODEL_DIR}, checking if it is valid ...")
        try:
            _ = AutoTokenizer.from_pretrained(MODEL_DIR)
            _ = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            print("[INFO] Existing MODEL_DIR is valid. Skip training.")
            return
        except Exception as e:
            print("[WARN] Existing MODEL_DIR seems broken or incompatible.")
            print("       Error while loading:", repr(e))
            print("       Removing the folder and retraining from scratch...")
            import shutil
            shutil.rmtree(MODEL_DIR)

    # (2) 여기까지 왔다는 건:
    #     - MODEL_DIR 이 아예 없거나
    #     - 있었지만 로드 실패 → 삭제 완료 상태
    if not os.path.exists(FOR_LEARNING_CSV):
        raise FileNotFoundError(
            f"[ERROR] '{FOR_LEARNING_CSV}' not found. "
            "훈련용 CSV가 없으면 새로 학습할 수 없습니다."
        )

    print(f"[INFO] Loading training data from {FOR_LEARNING_CSV} ...")
    df = pd.read_csv(FOR_LEARNING_CSV)

    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(
            f"[ERROR] '{FOR_LEARNING_CSV}' must contain columns '{TEXT_COL}' and '{LABEL_COL}'."
        )

    # label 정리
    df = df[df[LABEL_COL].isin(label2id.keys())].copy()
    df["label_id"] = df[LABEL_COL].map(label2id)
    print("[INFO] Label distribution:")
    print(df[LABEL_COL].value_counts())

    # train / valid split
    train_df, valid_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label_id"],
    )
    print("[INFO] #train =", len(train_df), ", #valid =", len(valid_df))

    # tokenizer
    model_name = "xlm-roberta-base"
    print(f"[INFO] Loading tokenizer/model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        texts = [str(x) if x is not None else "" for x in examples[TEXT_COL]]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    # HF Dataset
    train_ds = Dataset.from_pandas(train_df[[TEXT_COL, "label_id"]])
    valid_ds = Dataset.from_pandas(valid_df[[TEXT_COL, "label_id"]])

    train_ds = train_ds.map(tokenize_fn, batched=True)
    valid_ds = valid_ds.map(tokenize_fn, batched=True)

    train_ds = train_ds.rename_column("label_id", "labels")
    valid_ds = valid_ds.rename_column("label_id", "labels")

    # index 등 불필요 컬럼 제거
    cols_to_remove = [
        c for c in train_ds.column_names
        if c not in ["input_ids", "attention_mask", "labels", TEXT_COL]
    ]
    train_ds = train_ds.remove_columns([c for c in cols_to_remove if c != TEXT_COL])
    valid_ds = valid_ds.remove_columns([c for c in cols_to_remove if c != TEXT_COL])

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    valid_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # model
    num_labels = 3
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "macro_f1": macro_f1}

    # transformers 구버전 호환을 위해 evaluation_strategy 등은 빼둔 버전
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        # evaluation_strategy="epoch",
        # save_strategy="epoch",
    )

    print("[INFO] Start training ...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("[INFO] Training finished. Saving model to", MODEL_DIR)
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    # optional: eval report 출력
    print("[INFO] Final evaluation on valid set:")
    eval_res = trainer.evaluate()
    print(eval_res)

    preds = trainer.predict(valid_ds)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=-1)
    print(classification_report(
        y_true,
        y_pred,
        target_names=["neg", "neu", "pos"],
        digits=4,
    ))



def run_inference_and_report():
    """
    ./bert_sentiment 모델을 로드해서
    comments_for_inference.csv 에 대해 inference + 라벨 분포 그래프 + 대표 문장 출력
    """
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"[ERROR] MODEL_DIR '{MODEL_DIR}' does not exist. "
            "학습된 BERT 모델 폴더 경로를 확인하세요."
        )

    if not os.path.exists(INFERENCE_CSV):
        raise FileNotFoundError(
            f"[ERROR] INFERENCE_CSV '{INFERENCE_CSV}' not found. "
            "새 댓글 CSV를 {INFERENCE_CSV} 이름으로 두고 실행하세요."
        )

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from datasets import Dataset
    from torch.utils.data import DataLoader

    print(f"[INFO] Loading tokenizer/model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[INFO] Using device: {device}")

    # tokenize_fn
    def tokenize_fn(examples):
        texts = [str(x) if x is not None else "" for x in examples[TEXT_COL]]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    # inference dataset
    print(f"[INFO] Loading inference data from {INFERENCE_CSV} ...")
    df_new = pd.read_csv(INFERENCE_CSV)
    if TEXT_COL not in df_new.columns:
        raise ValueError(
            f"[ERROR] '{INFERENCE_CSV}' must contain column '{TEXT_COL}'."
        )

    new_texts = df_new[TEXT_COL].astype(str).tolist()
    new_ds = Dataset.from_dict({TEXT_COL: new_texts})
    new_ds = new_ds.map(tokenize_fn, batched=True)
    new_ds = new_ds.remove_columns(
        [c for c in new_ds.column_names if c not in ["input_ids", "attention_mask", TEXT_COL]]
    )
    new_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # batch inference
    print("[INFO] Running batch inference ...")
    all_logits = []
    loader = DataLoader(new_ds, batch_size=32)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu().numpy())

    logits = np.vstack(all_logits)
    pred_ids = np.argmax(logits, axis=-1)

    # id2label 은 model.config 에도 있으나, 우리가 정의한 것과 align
    id2label_local = model.config.id2label
    pred_labels = [id2label_local[int(i)] for i in pred_ids]

    df_new["pred_label"] = pred_labels
    out_path = "comments_new_with_bert_pred.csv"
    df_new.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Saved →", out_path)

    # -----------------
    # 2) label 분포 그래프
    # -----------------
    label_counts = df_new["pred_label"].value_counts()
    labels = label_counts.index.tolist()
    counts = label_counts.values.tolist()

    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts)
    plt.title("Prediction Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print("\n[INFO] Label distribution:")
    print(label_counts)

    # -----------------
    # 3) 각 라벨별 대표 문장 20개 출력
    # -----------------
    N = 20
    for lab in labels:
        print("\n" + "=" * 80)
        print(f"대표 문장 {N}개 — label: {lab}")
        print("=" * 80)

        subset = df_new[df_new["pred_label"] == lab][TEXT_COL].head(N)
        for i, sent in enumerate(subset, start=1):
            print(f"{i:02d}. {sent}")


# -------------------------
# main
# -------------------------
def main():
    # 1) Colab + Drive zip 이 있는 경우 unzip 시도
    maybe_mount_drive_and_unzip()

    # 2) MODEL_DIR 없으면 학습
    train_if_needed()

    # 3) 최종 inference
    run_inference_and_report()


if __name__ == "__main__":
    main()
