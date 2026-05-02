import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model_architecture.transformer import Transformer, CompressedTransformer
from data_preparation.dataset_creator import create_dataset
import time
import json
import sys
from pathlib import Path

# Добавляем корень репозитория в PYTHONPATH
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from experiment_configs import ALL_EXPERIMENTS, get_e2e_configs


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


def run_experiment(
    exp_name: str,
    cfg: dict,
    data_file: str,
    vocab_file: str,
    device: torch.device,
    mode: str = "e2e",
    override_epochs: int = None,
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
    epochs = override_epochs if override_epochs is not None else cfg["epochs"]
    epoch_losses = []
    for epoch in range(epochs):
        print(f"  Epoch {epoch+1}/{epochs} started...")
        loss = train_one_epoch(model, dataloader, optimizer, criterion, device, max_batches)
        epoch_losses.append(loss)
        print(f"  Epoch {epoch+1}/{epochs} finished, loss={loss:.4f}")

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
    print(f"Experiment {exp_name} finished in {elapsed:.1f}s with final loss={final_loss:.4f}")
    return result


def main():
    import argparse
    import traceback

    parser = argparse.ArgumentParser(description="Experiment runner")
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
        "--exp_name", type=str, default=None,
        help="Запустить только указанный эксперимент. Если не указан, запускаются все."
    )
    parser.add_argument(
        "--series", type=int, nargs="*", default=None,
        help="Запустить только указанные серии (1, 2, 3). По умолчанию — все."
    )
    parser.add_argument(
        "--log_file", type=str, default=None,
        help="Путь к файлу для записи логов (stdout/stderr)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Переопределить количество эпох обучения"
    )
    # Кастомные параметры модели
    parser.add_argument("--embed_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--forward_expansion", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--use_compression", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--compressor_layers", type=int, default=2)

    args = parser.parse_args()

    # Перенаправляем stdout/stderr, если указан log_file
    if args.log_file:
        sys.stdout = open(args.log_file, 'w', encoding='utf-8', buffering=1) # Line-buffered
        sys.stderr = sys.stdout

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
    if args.exp_name == "custom":
        # Создаем кастомный конфиг из аргументов
        configs = {
            "custom": {
                "series": 0,
                "group": "custom",
                "use_compression": args.use_compression,
                "embed_size": args.embed_size or 128,
                "num_layers": args.num_layers or 2,
                "heads": args.heads or 4,
                "forward_expansion": args.forward_expansion,
                "dropout": args.dropout,
                "max_length": args.max_length,
                "block_size": args.max_length,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "epochs": args.epochs or 1,
                "chunk_size": args.chunk_size,
                "compressor_layers": args.compressor_layers,
            }
        }
    else:
        if args.mode == "e2e":
            configs = get_e2e_configs()
        else:
            configs = ALL_EXPERIMENTS

        # Фильтрация по сериям
        if args.series:
            configs = {k: v for k, v in configs.items() if v.get("series") in args.series}

        # Фильтрация по имени эксперимента
        if args.exp_name:
            if args.exp_name not in configs:
                print(f"Error: Experiment '{args.exp_name}' not found in selected configs.")
                sys.exit(1)
            configs = {args.exp_name: configs[args.exp_name]}

    print(f"Total experiments to run: {len(configs)}\n")

    results = {}
    failed = []

    for exp_name, cfg in configs.items():
        try:
            res = run_experiment(
                exp_name, cfg, args.data, args.vocab, device, mode=args.mode, override_epochs=args.epochs
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
