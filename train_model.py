import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Attention, Bidirectional, Concatenate, Dense, Dropout, Embedding, GlobalAveragePooling1D, GlobalMaxPool1D, Input, LSTM, LayerNormalization, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from utils.text_processing import clean_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset/news.csv")
    parser.add_argument("--fake-data", type=str, default="")
    parser.add_argument("--real-data", type=str, default="")
    parser.add_argument(
        "--mix-extra-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When training with --fake-data and --real-data, also mix in --data if available.",
    )
    parser.add_argument("--model-dir", type=str, default="model")
    parser.add_argument("--max-words", type=int, default=20000)
    parser.add_argument("--max-len", type=int, default=360)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--full-batch", action="store_true", help="Train with all training rows in a single batch per epoch")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--recent-ratio", type=float, default=0.35)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--recurrent-dropout", type=float, default=0.0)
    return parser.parse_args()


def normalize_label(value) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"real", "true", "1", "genuine"}:
            return 1
        if normalized in {"fake", "false", "0", "fraud"}:
            return 0
    if value in {1, True}:
        return 1
    if value in {0, False}:
        return 0
    raise ValueError(f"Unsupported label value: {value}")


def _is_useful_clean_text(text: str) -> bool:
    if not text:
        return False
    tokens = text.split()
    if len(tokens) < 6:
        return False
    if len(text) < 40:
        return False
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    return unique_ratio >= 0.35


def _finalize_dataset(
    texts: list[str],
    labels: np.ndarray,
    random_state: int,
    dates: list[str] | None = None,
    recent_ratio: float = 0.35,
) -> tuple[list[str], np.ndarray]:
    frame = pd.DataFrame({"text": texts, "label": labels})
    if dates is not None:
        frame["date"] = pd.to_datetime(pd.Series(dates), errors="coerce", utc=False)

    frame = frame.dropna(subset=["text", "label"]).copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"].map(_is_useful_clean_text)]
    frame = frame.drop_duplicates(subset=["text"])

    if frame["label"].nunique() != 2:
        raise ValueError("Dataset must contain both REAL and FAKE samples")

    min_count = int(frame["label"].value_counts().min())
    samples = []
    has_date = "date" in frame.columns and frame["date"].notna().any()

    for label in sorted(frame["label"].unique().tolist()):
        class_frame = frame[frame["label"] == label].copy()
        if has_date:
            class_frame = class_frame.sort_values("date", ascending=False)
            dated_rows = class_frame[class_frame["date"].notna()]
            recent_n = min(int(min_count * max(0.0, min(recent_ratio, 0.9))), len(dated_rows))

            recent_part = dated_rows.head(recent_n)
            remaining_needed = min_count - len(recent_part)
            pool = class_frame.drop(index=recent_part.index, errors="ignore")
            if remaining_needed > 0:
                other_part = pool.sample(n=remaining_needed, random_state=random_state)
                selected = pd.concat([recent_part, other_part], axis=0)
            else:
                selected = recent_part
            samples.append(selected)
        else:
            samples.append(class_frame.sample(n=min_count, random_state=random_state))

    balanced = pd.concat(samples, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if len(balanced) < 200:
        raise ValueError("Dataset is too small after balancing and cleaning. Provide more data.")

    return balanced["text"].tolist(), balanced["label"].astype(np.int32).to_numpy()


def load_dataset(
    path: Path,
    random_state: int,
    recent_ratio: float = 0.35,
    min_rows: int = 200,
    finalize: bool = True,
) -> tuple[list[str], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    lowered_cols = {col.lower().strip(): col for col in df.columns}

    text_col = None
    for candidate in ["text", "news", "content", "article", "statement"]:
        if candidate in lowered_cols:
            text_col = lowered_cols[candidate]
            break
    if text_col is None:
        raise ValueError("Dataset must include one text column: text/news/content/article/statement")

    label_col = None
    for candidate in ["label", "class", "target", "is_fake"]:
        if candidate in lowered_cols:
            label_col = lowered_cols[candidate]
            break
    if label_col is None:
        raise ValueError("Dataset must include one label column: label/class/target/is_fake")

    df = df[[text_col, label_col]].dropna().copy()
    df[text_col] = df[text_col].astype(str)
    cleaned_texts = clean_texts(df[text_col].tolist())

    rows = []
    labels = []
    for cleaned, raw_label in zip(cleaned_texts, df[label_col].tolist()):
        if not _is_useful_clean_text(cleaned):
            continue
        try:
            labels.append(normalize_label(raw_label))
            rows.append(cleaned)
        except ValueError:
            continue

    if len(rows) < min_rows:
        raise ValueError(f"Dataset is too small after cleaning. Provide at least {min_rows} usable rows.")

    label_array = np.array(labels, dtype=np.int32)
    if not finalize:
        return rows, label_array

    return _finalize_dataset(rows, label_array, random_state=random_state, recent_ratio=recent_ratio)


def _select_text_column(df: pd.DataFrame) -> str:
    lowered_cols = {col.lower().strip(): col for col in df.columns}
    for candidate in ["text", "news", "content", "article", "statement"]:
        if candidate in lowered_cols:
            return lowered_cols[candidate]
    raise ValueError("Dataset must include one text column: text/news/content/article/statement")


def _resolve_data_path(path_like: str) -> Path:
    given_path = Path(path_like)
    if given_path.exists():
        return given_path
    dataset_fallback = Path("dataset") / path_like
    if dataset_fallback.exists():
        return dataset_fallback
    return given_path


def load_kaggle_dual_dataset(
    fake_path: Path,
    real_path: Path,
    random_state: int,
    recent_ratio: float = 0.35,
) -> tuple[list[str], np.ndarray]:
    if not fake_path.exists():
        raise FileNotFoundError(f"Fake dataset file not found: {fake_path}")
    if not real_path.exists():
        raise FileNotFoundError(f"Real dataset file not found: {real_path}")

    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)

    fake_text_col = _select_text_column(fake_df)
    real_text_col = _select_text_column(real_df)

    fake_title_col = next((col for col in fake_df.columns if col.lower().strip() == "title"), None)
    real_title_col = next((col for col in real_df.columns if col.lower().strip() == "title"), None)
    fake_date_col = next((col for col in fake_df.columns if col.lower().strip() == "date"), None)
    real_date_col = next((col for col in real_df.columns if col.lower().strip() == "date"), None)

    fake_text = fake_df[fake_text_col].fillna("").astype(str)
    real_text = real_df[real_text_col].fillna("").astype(str)

    if fake_title_col is not None:
        fake_title = fake_df[fake_title_col].fillna("").astype(str)
        fake_text = (fake_title + " " + fake_text).str.strip()
    if real_title_col is not None:
        real_title = real_df[real_title_col].fillna("").astype(str)
        real_text = (real_title + " " + real_text).str.strip()

    combined_texts = fake_text.tolist() + real_text.tolist()
    combined_labels = np.array([0] * len(fake_text) + [1] * len(real_text), dtype=np.int32)
    combined_dates = []
    if fake_date_col is not None:
        combined_dates.extend(fake_df[fake_date_col].fillna("").astype(str).tolist())
    else:
        combined_dates.extend([""] * len(fake_text))
    if real_date_col is not None:
        combined_dates.extend(real_df[real_date_col].fillna("").astype(str).tolist())
    else:
        combined_dates.extend([""] * len(real_text))

    cleaned = clean_texts(combined_texts)
    rows = []
    labels = []
    dates = []
    for text, label, date_value in zip(cleaned, combined_labels, combined_dates):
        if not _is_useful_clean_text(text):
            continue
        rows.append(text)
        labels.append(label)
        dates.append(date_value)

    if len(rows) < 200:
        raise ValueError("Dataset is too small after cleaning. Provide at least 200 usable rows.")

    return _finalize_dataset(
        rows,
        np.array(labels, dtype=np.int32),
        random_state=random_state,
        dates=dates,
        recent_ratio=recent_ratio,
    )


def _balanced_accuracy_from_preds(y_true: np.ndarray, preds: np.ndarray) -> float:
    matrix = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float((tpr + tnr) / 2.0)


def calibrate_thresholds(y_val: np.ndarray, val_probs: np.ndarray) -> tuple[float, float, dict]:
    best_score = None
    best_info = None

    for threshold in np.linspace(0.25, 0.70, 91):
        preds = (val_probs >= threshold).astype(np.int32)
        f1 = float(f1_score(y_val, preds, zero_division=0))
        bal_acc = _balanced_accuracy_from_preds(y_val, preds)
        real_as_fake_rate = float(np.mean((y_val == 1) & (preds == 0)))
        score = f1 * 0.50 + bal_acc * 0.30 + (1.0 - real_as_fake_rate) * 0.20
        if best_score is None or score > best_score:
            best_score = score
            best_info = {
                "threshold": float(threshold),
                "f1": f1,
                "balanced_accuracy": bal_acc,
                "real_as_fake_rate": real_as_fake_rate,
                "preds": preds,
            }

    if best_info is None:
        return 0.5, 0.7, {"note": "fallback thresholds"}

    best_preds = best_info["preds"]
    confidence = np.where(best_preds == 1, val_probs, 1.0 - val_probs)
    correct_mask = best_preds == y_val
    if np.any(correct_mask):
        uncertainty_threshold = float(np.percentile(confidence[correct_mask], 20))
    else:
        uncertainty_threshold = 0.7

    uncertainty_threshold = float(np.clip(uncertainty_threshold, 0.6, 0.9))
    return best_info["threshold"], uncertainty_threshold, {
        "val_f1": best_info["f1"],
        "val_balanced_accuracy": best_info["balanced_accuracy"],
        "val_real_as_fake_rate": best_info["real_as_fake_rate"],
    }


def build_model(vocab_size: int, max_len: int, embedding_dim: int, recurrent_dropout: float) -> tf.keras.Model:
    inputs = Input(shape=(max_len,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    x = SpatialDropout1D(0.3)(x)
    sequence = Bidirectional(LSTM(96, return_sequences=True, dropout=0.2, recurrent_dropout=recurrent_dropout))(x)
    attended = Attention(use_scale=True)([sequence, sequence])
    merged = Concatenate(axis=-1)([sequence, attended])
    pooled_max = GlobalMaxPool1D()(merged)
    pooled_avg = GlobalAveragePooling1D()(merged)
    x = Concatenate()([pooled_max, pooled_avg])
    x = LayerNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.45)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=7e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    return model


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.fake_data and args.real_data:
        fake_path = _resolve_data_path(args.fake_data)
        real_path = _resolve_data_path(args.real_data)
        texts, labels = load_kaggle_dual_dataset(
            fake_path,
            real_path,
            random_state=args.random_state,
            recent_ratio=args.recent_ratio,
        )

        if args.mix_extra_data and args.data:
            data_path = _resolve_data_path(args.data)
            if data_path.exists():
                extra_texts, extra_labels = load_dataset(
                    data_path,
                    random_state=args.random_state,
                    recent_ratio=args.recent_ratio,
                    min_rows=20,
                    finalize=False,
                )
                texts = texts + extra_texts
                labels = np.concatenate([labels, extra_labels], axis=0)
                texts, labels = _finalize_dataset(
                    texts,
                    labels,
                    random_state=args.random_state,
                    recent_ratio=args.recent_ratio,
                )
    else:
        data_path = _resolve_data_path(args.data)
        texts, labels = load_dataset(
            data_path,
            random_state=args.random_state,
            recent_ratio=args.recent_ratio,
        )

    if not (0.05 <= args.val_size <= 0.4):
        raise ValueError("val-size must be between 0.05 and 0.4")

    x_train_texts, x_test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        stratify=labels,
        random_state=args.random_state,
    )

    x_train_texts, x_val_texts, y_train, y_val = train_test_split(
        x_train_texts,
        y_train,
        test_size=args.val_size,
        stratify=y_train,
        random_state=args.random_state,
    )

    tokenizer = Tokenizer(num_words=args.max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train_texts)

    x_train = pad_sequences(tokenizer.texts_to_sequences(x_train_texts), maxlen=args.max_len, padding="post", truncating="post")
    x_val = pad_sequences(tokenizer.texts_to_sequences(x_val_texts), maxlen=args.max_len, padding="post", truncating="post")
    x_test = pad_sequences(tokenizer.texts_to_sequences(x_test_texts), maxlen=args.max_len, padding="post", truncating="post")

    vocab_size = min(args.max_words, len(tokenizer.word_index) + 1)
    model = build_model(
        vocab_size=vocab_size,
        max_len=args.max_len,
        embedding_dim=args.embedding_dim,
        recurrent_dropout=args.recurrent_dropout,
    )

    effective_batch_size = len(x_train) if args.full_batch else args.batch_size
    if effective_batch_size <= 0:
        raise ValueError("batch-size must be > 0")

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_labels = np.unique(y_train)
    class_weight_dict = {int(label): float(weight) for label, weight in zip(class_labels, class_weights)}

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, min_delta=1e-4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=effective_batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    val_probs = model.predict(x_val, verbose=0).ravel()
    calibrated_threshold, calibrated_uncertainty_threshold, val_calibration = calibrate_thresholds(y_val, val_probs)

    probs = model.predict(x_test, verbose=0).ravel()
    preds = (probs >= calibrated_threshold).astype(np.int32)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "report": classification_report(y_test, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "real_as_fake_rate": float(np.mean((y_test == 1) & (preds == 0))),
        "train_size": int(len(x_train)),
        "val_size": int(len(x_val)),
        "test_size": int(len(x_test)),
        "threshold": calibrated_threshold,
        "uncertainty_threshold": calibrated_uncertainty_threshold,
        "val_calibration": val_calibration,
    }

    model.save(model_dir / "fake_news_model.keras")
    with open(model_dir / "tokenizer.json", "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())

    metadata = {
        "max_len": args.max_len,
        "max_words": args.max_words,
        "embedding_dim": args.embedding_dim,
        "batch_size": int(effective_batch_size),
        "full_batch": bool(args.full_batch),
        "recurrent_dropout": float(args.recurrent_dropout),
        "threshold": calibrated_threshold,
        "uncertainty_threshold": calibrated_uncertainty_threshold,
        "mix_extra_data": bool(args.mix_extra_data),
        "recent_ratio": float(args.recent_ratio),
        "val_size": float(args.val_size),
    }

    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Training batch size used: {effective_batch_size}")
    print("Model training completed.")
    print(json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2))
    print(metrics["report"])


if __name__ == "__main__":
    main()
