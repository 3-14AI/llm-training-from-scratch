"""
experiment_configs.py
=====================
Полные конфигурации для трёх серий экспериментов.

Серия 1 — Зависимость качества от размера модели
  Три пары (малая / средняя / большая) × (baseline / compressed).
  Все конфиги рассчитаны так, чтобы модель помещалась в 12 ГБ VRAM.
  Для e2e-теста на CPU параметры уменьшены (флаг e2e=True в runner).

Серия 2 — Зависимость от размера компрессора (на средней модели)
  Фиксируем архитектуру средней модели, варьируем compressor_layers.

Серия 3 — Зависимость от длины обучающих последовательностей
  Фиксируем малую модель, варьируем block_size / max_length.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательная функция: масштабирование для e2e-теста на CPU
# ─────────────────────────────────────────────────────────────────────────────

def _scale(cfg: dict, e2e: bool) -> dict:
    """Уменьшает embed_size и num_layers для быстрого e2e-теста на CPU."""
    if not e2e:
        return cfg
    out = cfg.copy()
    # Сжимаем до минимума, сохраняя структуру
    out["embed_size"] = max(32, cfg["embed_size"] // 8)
    out["num_layers"] = max(1, cfg["num_layers"] // 2)
    out["batch_size"] = 2
    out["epochs"] = 1
    out["max_batches_per_epoch"] = 3   # ограничение числа батчей для e2e
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Серия 1: Зависимость от размера модели
# ─────────────────────────────────────────────────────────────────────────────
# Ограничения по VRAM (12 ГБ):
#   small  — embed=256,  layers=2,  heads=8   → ~15M params
#   medium — embed=512,  layers=4,  heads=8   → ~85M params
#   large  — embed=768,  layers=6,  heads=12  → ~280M params
# batch_size подобран так, чтобы активации + параметры вписывались в 12 ГБ.

SERIES_1 = {
    "s1_small_baseline": {
        "embed_size": 256, "num_layers": 2, "heads": 8,
        "forward_expansion": 4, "dropout": 0.1,
        "max_length": 128, "block_size": 128,
        "batch_size": 16, "epochs": 5, "lr": 1e-4,
        "use_compression": False,
        "chunk_size": None, "compressor_layers": None,
        "series": 1, "group": "small",
    },
    "s1_small_compressed": {
        "embed_size": 256, "num_layers": 2, "heads": 8,
        "forward_expansion": 4, "dropout": 0.1,
        "max_length": 128, "block_size": 128,
        "batch_size": 16, "epochs": 5, "lr": 1e-4,
        "use_compression": True,
        "chunk_size": 8, "compressor_layers": 2,
        "series": 1, "group": "small",
    },

    "s1_medium_baseline": {
        "embed_size": 512, "num_layers": 4, "heads": 8,
        "forward_expansion": 4, "dropout": 0.1,
        "max_length": 256, "block_size": 256,
        "batch_size": 8, "epochs": 5, "lr": 5e-5,
        "use_compression": False,
        "chunk_size": None, "compressor_layers": None,
        "series": 1, "group": "medium",
    },
    "s1_medium_compressed": {
        "embed_size": 512, "num_layers": 4, "heads": 8,
        "forward_expansion": 4, "dropout": 0.1,
        "max_length": 256, "block_size": 256,
        "batch_size": 8, "epochs": 5, "lr": 5e-5,
        "use_compression": True,
        "chunk_size": 16, "compressor_layers": 3,
        "series": 1, "group": "medium",
    },

    "s1_large_baseline": {
        "embed_size": 768, "num_layers": 6, "heads": 12,
        "forward_expansion": 4, "dropout": 0.1,
        "max_length": 512, "block_size": 512,
        "batch_size": 4, "epochs": 3, "lr": 1e-5,
        "use_compression": False,
        "chunk_size": None, "compressor_layers": None,
        "series": 1, "group": "large",
    },
    "s1_large_compressed": {
        "embed_size": 768, "num_layers": 6, "heads": 12,
        "forward_expansion": 4, "dropout": 0.1,
        "max_length": 512, "block_size": 512,
        "batch_size": 4, "epochs": 3, "lr": 1e-5,
        "use_compression": True,
        "chunk_size": 32, "compressor_layers": 4,
        "series": 1, "group": "large",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Серия 2: Зависимость от размера компрессора (средняя модель)
# ─────────────────────────────────────────────────────────────────────────────
# Базовая архитектура: medium (embed=512, layers=4, heads=8).
# Варьируем compressor_layers: 1, 2, 3, 4, 6.
# Baseline — та же средняя модель без компрессии (s1_medium_baseline).

SERIES_2 = {
    "s2_medium_baseline": {
        "embed_size": 512, "num_layers": 4, "heads": 8,
        "forward_expansion": 4, "dropout": 0.1,
        "max_length": 256, "block_size": 256,
        "batch_size": 8, "epochs": 5, "lr": 5e-5,
        "use_compression": False,
        "chunk_size": None, "compressor_layers": None,
        "series": 2, "group": "comp_baseline",
    },
    "s2_comp_layers_1": {
        "embed_size": 512, "num_layers": 4, "heads": 8,
        "forward_expansion": 4, "dropout": 0.1,
        "max_length": 256, "block_size": 256,
        "batch_size": 8, "epochs": 5, "lr": 5e-5,
        "use_compression": True,
        "chunk_size": 16, "compressor_layers": 1,
        "series": 2, "group": "comp_layers_1",
    },
    "s2_comp_layers_2": {
        "embed_size": 512, "num_layers": 4, "heads": 8,
        "forward_expansion": 4, "dropout": 0.1,
        "max_length": 256, "block_size": 256,
        "batch_size": 8, "epochs": 5, "lr": 5e-5,
        "use_compression": True,
        "chunk_size": 16, "compressor_layers": 2,
        "series": 2, "group": "comp_layers_2",
    },
    "s2_comp_layers_3": {
        "embed_size": 512, "num_layers": 4, "heads": 8,
        "forward_expansion": 4, "dropout": 0.1,
        "max_length": 256, "block_size": 256,
        "batch_size": 8, "epochs": 5, "lr": 5e-5,
        "use_compression": True,
        "chunk_size": 16, "compressor_layers": 3,
        "series": 2, "group": "comp_layers_3",
    },
    "s2_comp_layers_4": {
        "embed_size": 512, "num_layers": 4, "heads": 8,
        "forward_expansion": 4, "dropout": 0.1,
        "max_length": 256, "block_size": 256,
        "batch_size": 8, "epochs": 5, "lr": 5e-5,
        "use_compression": True,
        "chunk_size": 16, "compressor_layers": 4,
        "series": 2, "group": "comp_layers_4",
    },
    "s2_comp_layers_6": {
        "embed_size": 512, "num_layers": 4, "heads": 8,
        "forward_expansion": 4, "dropout": 0.1,
        "max_length": 256, "block_size": 256,
        "batch_size": 8, "epochs": 5, "lr": 5e-5,
        "use_compression": True,
        "chunk_size": 16, "compressor_layers": 6,
        "series": 2, "group": "comp_layers_6",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Серия 3: Зависимость от длины обучающих последовательностей
# ─────────────────────────────────────────────────────────────────────────────
# Фиксируем малую модель (embed=256, layers=2, heads=8).
# Варьируем block_size: 64, 128, 256, 512.
# chunk_size масштабируется пропорционально (block_size // 8).

SERIES_3 = {
    "s3_seq64_baseline": {
        "embed_size": 256, "num_layers": 2, "heads": 8,
        "forward_expansion": 4, "dropout": 0.1,
        "max_length": 64, "block_size": 64,
        "batch_size": 16, "epochs": 5, "lr": 1e-4,
        "use_compression": False,
        "chunk_size": None, "compressor_layers": None,
        "series": 3, "group": "seq64",
    },
    "s3_seq64_compressed": {
        "embed_size": 256, "num_layers": 2, "heads": 8,
        "forward_expansion": 4, "dropout": 0.1,
        "max_length": 64, "block_size": 64,
        "batch_size": 16, "epochs": 5, "lr": 1e-4,
        "use_compression": True,
        "chunk_size": 8, "compressor_layers": 2,
        "series": 3, "group": "seq64",
    },

    "s3_seq128_baseline": {
        "embed_size": 256, "num_layers": 2, "heads": 8,
        "forward_expansion": 4, "dropout": 0.1,
        "max_length": 128, "block_size": 128,
        "batch_size": 16, "epochs": 5, "lr": 1e-4,
        "use_compression": False,
        "chunk_size": None, "compressor_layers": None,
        "series": 3, "group": "seq128",
    },
    "s3_seq128_compressed": {
        "embed_size": 256, "num_layers": 2, "heads": 8,
        "forward_expansion": 4, "dropout": 0.1,
        "max_length": 128, "block_size": 128,
        "batch_size": 16, "epochs": 5, "lr": 1e-4,
        "use_compression": True,
        "chunk_size": 16, "compressor_layers": 2,
        "series": 3, "group": "seq128",
    },

    "s3_seq256_baseline": {
        "embed_size": 256, "num_layers": 2, "heads": 8,
        "forward_expansion": 4, "dropout": 0.1,
        "max_length": 256, "block_size": 256,
        "batch_size": 8, "epochs": 5, "lr": 1e-4,
        "use_compression": False,
        "chunk_size": None, "compressor_layers": None,
        "series": 3, "group": "seq256",
    },
    "s3_seq256_compressed": {
        "embed_size": 256, "num_layers": 2, "heads": 8,
        "forward_expansion": 4, "dropout": 0.1,
        "max_length": 256, "block_size": 256,
        "batch_size": 8, "epochs": 5, "lr": 1e-4,
        "use_compression": True,
        "chunk_size": 32, "compressor_layers": 2,
        "series": 3, "group": "seq256",
    },

    "s3_seq512_baseline": {
        "embed_size": 256, "num_layers": 2, "heads": 8,
        "forward_expansion": 4, "dropout": 0.1,
        "max_length": 512, "block_size": 512,
        "batch_size": 4, "epochs": 5, "lr": 1e-4,
        "use_compression": False,
        "chunk_size": None, "compressor_layers": None,
        "series": 3, "group": "seq512",
    },
    "s3_seq512_compressed": {
        "embed_size": 256, "num_layers": 2, "heads": 8,
        "forward_expansion": 4, "dropout": 0.1,
        "max_length": 512, "block_size": 512,
        "batch_size": 4, "epochs": 5, "lr": 1e-4,
        "use_compression": True,
        "chunk_size": 64, "compressor_layers": 2,
        "series": 3, "group": "seq512",
    },
}

# Все эксперименты объединены
ALL_EXPERIMENTS = {**SERIES_1, **SERIES_2, **SERIES_3}


def get_e2e_configs() -> dict:
    """Возвращает масштабированные конфиги для e2e-теста на CPU."""
    e2e = {}
    for name, cfg in ALL_EXPERIMENTS.items():
        c = _scale(cfg, e2e=True)
        e2e[name] = c
    return e2e


if __name__ == "__main__":
    import json
    print("=== Full configs ===")
    print(json.dumps(ALL_EXPERIMENTS, indent=2, default=str))
    print("\n=== E2E (CPU-scaled) configs ===")
    print(json.dumps(get_e2e_configs(), indent=2, default=str))
