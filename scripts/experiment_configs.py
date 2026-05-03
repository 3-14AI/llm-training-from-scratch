"""
experiment_configs.py
=====================
Словарь конфигураций для экспериментов.
Теперь загружается из TOML файла для удобства и независимости от Python кода.
"""
import toml
import os

# Загружаем конфигурации из TOML
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "experiment_configs.toml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _configs = toml.load(f)

ALL_EXPERIMENTS = _configs.get("ALL_EXPERIMENTS", {})
RUN_EXPERIMENTS_CONFIGS = _configs.get("RUN_EXPERIMENTS", {})

# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательная функция: масштабирование для e2e-теста на CPU
# ─────────────────────────────────────────────────────────────────────────────

def _scale(cfg: dict, e2e: bool) -> dict:
    """Уменьшает embed_size и num_layers для быстрого e2e-теста на CPU."""
    if not e2e:
        return cfg
    out = cfg.copy()
    # Сжимаем до минимума, сохраняя структуру
    out["embed_size"] = max(32, cfg.get("embed_size", 32) // 8)
    out["num_layers"] = max(1, cfg.get("num_layers", 1) // 2)
    out["batch_size"] = 2
    out["epochs"] = 1
    out["max_batches_per_epoch"] = 3   # ограничение числа батчей для e2e
    return out


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
