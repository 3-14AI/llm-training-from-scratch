"""
run_e2e_tests.py
================
E2E тест-раннер для всех серий экспериментов.

Режимы:
  --mode e2e    — быстрый прогон на CPU (маленький чанк данных, 1 эпоха, 3 батча)
  --mode full   — полный прогон (требует GPU ≥12 ГБ VRAM)

Пример запуска e2e-теста:
  python run_e2e_tests.py --mode e2e --data multilingual_corpus.txt

Пример полного запуска:
  python run_e2e_tests.py --mode full --data multilingual_corpus.txt
"""

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

# Добавляем корень репозитория в PYTHONPATH
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from model_architecture.transformer import Transformer, CompressedTransformer
from data_preparation.dataset_creator import create_dataset
from experiment_configs import ALL_EXPERIMENTS, get_e2e_configs


# ─────────────────────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: dict, vocab_size: int, device: torch.device) -> nn.Module:
    """Создаёт модель по конфигу."""
    common = dict(
        src_vocab_size=vocab_size,
        trg_vocab_size=vocab_size,
        src_pad_idx=0,
        trg_pad_idx=0,
        embed_size=cfg["embed_size"],
        num_layers=cfg["num_layers"],
        forward_expansion=cfg["forward_expansion"],
        heads=cfg["heads"],
        dropout=cfg["dropout"],
        device=device,
        max_length=cfg["max_length"],
    )
    if cfg["use_compression"]:
        model = CompressedTransformer(
            **common,
            chunk_size=cfg["chunk_size"],
            compressor_layers=cfg["compressor_layers"],
        )
    else:
        model = Transformer(**common)
    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int = None,
) -> float:
    """Обучает одну эпоху, возвращает средний loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (src, trg) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        output = model(src, trg[:, :-1])
        output = output.reshape(-1, output.shape[-1])
        target = trg[:, 1:].reshape(-1)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция запуска одного эксперимента
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    exp_name: str,
    cfg: dict,
    data_file: str,
    vocab_file: str,
    device: torch.device,
    mode: str = "e2e",
) -> dict:
    """
    Запускает один эксперимент и возвращает словарь с результатами.
    """
    print(f"\n{'─'*60}")
    print(f"Experiment : {exp_name}")
    print(f"Series     : {cfg.get('series', '?')}  |  Group: {cfg.get('group', '?')}")
    print(f"Compression: {cfg['use_compression']}")
    if cfg["use_compression"]:
        print(f"  chunk_size={cfg['chunk_size']}, compressor_layers={cfg['compressor_layers']}")
    print(f"Model      : embed={cfg['embed_size']}, layers={cfg['num_layers']}, heads={cfg['heads']}")
    print(f"Seq length : {cfg['block_size']}  |  batch={cfg['batch_size']}")
    print(f"{'─'*60}")

    t0 = time.time()

    # Датасет
    dataloader, vocab_size = create_dataset(
        data_file, vocab_file,
        block_size=cfg["block_size"],
        batch_size=cfg["batch_size"],
    )

    # Модель
    model = build_model(cfg, vocab_size, device)
    n_params = count_params(model)
    print(f"Parameters : {n_params:,}")

    optimizer = Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    max_batches = cfg.get("max_batches_per_epoch", None)
    epochs = cfg["epochs"]

    epoch_losses = []
    for epoch in range(epochs):
        loss = train_one_epoch(model, dataloader, optimizer, criterion, device, max_batches)
        epoch_losses.append(loss)
        print(f"  Epoch {epoch+1}/{epochs}  loss={loss:.4f}")

    elapsed = time.time() - t0
    final_loss = epoch_losses[-1]

    result = {
        "exp_name": exp_name,
        "series": cfg.get("series"),
        "group": cfg.get("group"),
        "use_compression": cfg["use_compression"],
        "embed_size": cfg["embed_size"],
        "num_layers": cfg["num_layers"],
        "heads": cfg["heads"],
        "block_size": cfg["block_size"],
        "chunk_size": cfg.get("chunk_size"),
        "compressor_layers": cfg.get("compressor_layers"),
        "n_params": n_params,
        "epoch_losses": epoch_losses,
        "final_loss": final_loss,
        "elapsed_seconds": round(elapsed, 2),
        "status": "ok",
    }
    print(f"  Done in {elapsed:.1f}s  |  final_loss={final_loss:.4f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Запуск всех экспериментов
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="E2E experiment runner")
    parser.add_argument(
        "--mode", choices=["e2e", "full"], default="e2e",
        help="'e2e' — быстрый CPU-тест; 'full' — полный GPU-прогон"
    )
    parser.add_argument(
        "--data", type=str, default="multilingual_corpus.txt",
        help="Путь к текстовому корпусу"
    )
    parser.add_argument(
        "--vocab", type=str, default="multilingual_vocab.pt",
        help="Путь к файлу словаря (создаётся автоматически)"
    )
    parser.add_argument(
        "--output", type=str, default="experiment_results.json",
        help="Файл для сохранения результатов"
    )
    parser.add_argument(
        "--series", type=int, nargs="*", default=None,
        help="Запустить только указанные серии (1, 2, 3). По умолчанию — все."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Mode   : {args.mode}")
    print(f"Device : {device}")
    print(f"Data   : {args.data}")
    print(f"{'='*60}\n")

    if not os.path.exists(args.data):
        print(f"ERROR: Data file '{args.data}' not found.")
        print("Run 'python prepare_dataset.py' first to download the corpus.")
        sys.exit(1)

    # Выбор конфигов
    if args.mode == "e2e":
        configs = get_e2e_configs()
    else:
        configs = ALL_EXPERIMENTS

    # Фильтрация по сериям
    if args.series:
        configs = {k: v for k, v in configs.items() if v.get("series") in args.series}

    print(f"Total experiments to run: {len(configs)}\n")

    results = {}
    failed = []

    for exp_name, cfg in configs.items():
        try:
            res = run_experiment(
                exp_name, cfg, args.data, args.vocab, device, mode=args.mode
            )
            results[exp_name] = res
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"\n[ERROR] {exp_name}: {exc}\n{tb}")
            results[exp_name] = {
                "exp_name": exp_name,
                "status": "error",
                "error": str(exc),
                "traceback": tb,
            }
            failed.append(exp_name)

    # Сохраняем результаты
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Итоговая сводка
    print(f"\n{'='*60}")
    print(f"SUMMARY  ({args.mode} mode)")
    print(f"{'='*60}")
    print(f"{'Experiment':<35} {'Status':<8} {'Params':>12} {'FinalLoss':>10} {'Time(s)':>8}")
    print(f"{'─'*35} {'─'*8} {'─'*12} {'─'*10} {'─'*8}")
    for name, r in results.items():
        if r["status"] == "ok":
            print(
                f"{name:<35} {'OK':<8} {r['n_params']:>12,} "
                f"{r['final_loss']:>10.4f} {r['elapsed_seconds']:>8.1f}"
            )
        else:
            print(f"{name:<35} {'ERROR':<8}")

    print(f"\nResults saved to: {args.output}")
    if failed:
        print(f"Failed experiments: {failed}")
        sys.exit(1)
    else:
        print("All experiments completed successfully.")


if __name__ == "__main__":
    main()
