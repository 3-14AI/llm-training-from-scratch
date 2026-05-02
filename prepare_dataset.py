"""
prepare_dataset.py
==================
Загружает небольшой мультиязычный корпус для предобучения:
  - Русский (основной): ~70%  — Wikipedia RU
  - Английский (примесь): ~20% — Wikipedia EN
  - Китайский (примесь): ~10% — Wikipedia ZH

Для e2e-теста берём очень маленький чанк (num_samples_per_lang),
чтобы всё работало без GPU и за разумное время.

Результат: файл multilingual_corpus.txt в рабочей директории.
"""

import os
import random
import argparse
from datasets import load_dataset


# Маппинг языков на датасеты HuggingFace (parquet-совместимые, без loading scripts)
_DATASET_MAP = {
    "ru": ("wikimedia/wikipedia", "20231101.ru"),
    "en": ("wikimedia/wikipedia", "20231101.en"),
    "zh": ("wikimedia/wikipedia", "20231101.zh"),
}


def load_wiki_sample(lang: str, num_samples: int, seed: int = 42) -> list:
    """Загружает num_samples текстов из Wikipedia для заданного языка."""
    dataset_id, subset = _DATASET_MAP[lang]
    print(f"  Loading {dataset_id} ({subset}, {num_samples} samples)...")
    ds = load_dataset(dataset_id, subset, split="train", streaming=True)
    texts = []
    for i, example in enumerate(ds):
        if i >= num_samples:
            break
        text = example.get("text", "").strip()
        if text:
            # Берём первые 500 символов статьи, чтобы не перегружать корпус
            texts.append(text[:500])
    return texts


def build_corpus(
    output_path: str = "multilingual_corpus.txt",
    ru_samples: int = 700,
    en_samples: int = 200,
    zh_samples: int = 100,
    seed: int = 42,
):
    """Собирает корпус из трёх языков и сохраняет в один файл."""
    random.seed(seed)

    all_texts = []

    print("Fetching Russian texts (Wikipedia RU)...")
    ru = load_wiki_sample("ru", ru_samples, seed)
    all_texts.extend(ru)

    print("Fetching English texts (Wikipedia EN)...")
    en = load_wiki_sample("en", en_samples, seed)
    all_texts.extend(en)

    print("Fetching Chinese texts (Wikipedia ZH)...")
    zh = load_wiki_sample("zh", zh_samples, seed)
    all_texts.extend(zh)

    random.shuffle(all_texts)

    with open(output_path, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write(text + "\n")

    total_chars = sum(len(t) for t in all_texts)
    print(
        f"\nCorpus saved to '{output_path}': "
        f"{len(all_texts)} documents, ~{total_chars:,} chars"
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare multilingual pretraining corpus")
    parser.add_argument("--ru", type=int, default=700, help="Number of Russian samples")
    parser.add_argument("--en", type=int, default=200, help="Number of English samples")
    parser.add_argument("--zh", type=int, default=100, help="Number of Chinese samples")
    parser.add_argument("--output", type=str, default="multilingual_corpus.txt")
    args = parser.parse_args()

    build_corpus(
        output_path=args.output,
        ru_samples=args.ru,
        en_samples=args.en,
        zh_samples=args.zh,
    )
