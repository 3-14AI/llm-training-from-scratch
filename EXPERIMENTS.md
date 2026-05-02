# Серия экспериментов: Context Compression vs Baseline

Данный документ описывает три серии экспериментов для сравнения `Transformer` (baseline) и `CompressedTransformer` (с контекстной компрессией) из репозитория [llm-training-from-scratch](https://github.com/3-14AI/llm-training-from-scratch).

---

## Структура файлов

```
llm-training-from-scratch/
├── prepare_dataset.py        # Загрузка мультиязычного корпуса (RU/EN/ZH)
├── experiment_configs.py     # Все конфиги экспериментов (3 серии, 20 пар)
├── run_e2e_tests.py          # E2E тест-раннер (режимы: e2e / full)
└── EXPERIMENTS.md            # Этот файл
```

---

## Датасет

Корпус формируется из трёх языков Wikipedia (через `wikimedia/wikipedia` на HuggingFace):

| Язык     | Доля | Источник                      |
|----------|------|-------------------------------|
| Русский  | ~70% | `wikimedia/wikipedia 20231101.ru` |
| Английский | ~20% | `wikimedia/wikipedia 20231101.en` |
| Китайский | ~10% | `wikimedia/wikipedia 20231101.zh` |

Для e2e-теста используется маленький чанк (70 RU + 20 EN + 10 ZH документов, ~87 КБ). Для полного обучения рекомендуется увеличить `--ru`, `--en`, `--zh` до нескольких тысяч.

**Загрузка корпуса:**

```bash
# E2E-тест (маленький чанк)
python prepare_dataset.py --ru 70 --en 20 --zh 10 --output multilingual_corpus.txt

# Полный корпус
python prepare_dataset.py --ru 7000 --en 2000 --zh 1000 --output multilingual_corpus.txt
```

---

## Серия 1: Зависимость от размера модели

Три пары (малая / средняя / большая) × (baseline / compressed). Конфиги рассчитаны на GPU ≤ 12 ГБ VRAM.

| Эксперимент          | embed | layers | heads | Компрессия | chunk | comp_layers | VRAM (est.) |
|----------------------|-------|--------|-------|------------|-------|-------------|-------------|
| s1_small_baseline    | 256   | 2      | 8     | Нет        | —     | —           | ~2 ГБ       |
| s1_small_compressed  | 256   | 2      | 8     | Да         | 8     | 2           | ~2.5 ГБ     |
| s1_medium_baseline   | 512   | 4      | 8     | Нет        | —     | —           | ~6 ГБ       |
| s1_medium_compressed | 512   | 4      | 8     | Да         | 16    | 3           | ~7 ГБ       |
| s1_large_baseline    | 768   | 6      | 12    | Нет        | —     | —           | ~10 ГБ      |
| s1_large_compressed  | 768   | 6      | 12    | Да         | 32    | 4           | ~12 ГБ      |

**Гипотеза:** Компрессия должна давать бо́льший прирост качества на больших моделях, так как увеличивает эффективный контекст при фиксированном VRAM.

---

## Серия 2: Зависимость от размера компрессора

Фиксируется средняя модель (embed=512, layers=4, heads=8), варьируется `compressor_layers`.

| Эксперимент       | comp_layers | Параметры компрессора |
|-------------------|-------------|----------------------|
| s2_medium_baseline | 0 (нет)    | 0                    |
| s2_comp_layers_1  | 1           | ~0.5M                |
| s2_comp_layers_2  | 2           | ~1M                  |
| s2_comp_layers_3  | 3           | ~1.5M                |
| s2_comp_layers_4  | 4           | ~2M                  |
| s2_comp_layers_6  | 6           | ~3M                  |

**Гипотеза:** Существует оптимальный размер компрессора — слишком маленький не успевает обучиться, слишком большой переобучается на чанках.

---

## Серия 3: Зависимость от длины последовательности

Фиксируется малая модель (embed=256, layers=2, heads=8), варьируется `block_size`. `chunk_size = block_size // 8`.

| Эксперимент         | block_size | chunk_size | Компрессия |
|---------------------|------------|------------|------------|
| s3_seq64_baseline   | 64         | —          | Нет        |
| s3_seq64_compressed | 64         | 8          | Да         |
| s3_seq128_baseline  | 128        | —          | Нет        |
| s3_seq128_compressed| 128        | 16         | Да         |
| s3_seq256_baseline  | 256        | —          | Нет        |
| s3_seq256_compressed| 256        | 32         | Да         |
| s3_seq512_baseline  | 512        | —          | Нет        |
| s3_seq512_compressed| 512        | 64         | Да         |

**Гипотеза:** При коротких последовательностях компрессия не даёт преимущества. При длинных — baseline страдает от квадратичного внимания, а CompressedTransformer сохраняет эффективность.

---

## Запуск

### E2E-тест (CPU, без GPU, быстро)

```bash
# 1. Загрузить датасет (маленький чанк)
python prepare_dataset.py --ru 70 --en 20 --zh 10

# 2. Запустить все серии
python run_e2e_tests.py --mode e2e --data multilingual_corpus.txt

# 3. Запустить только одну серию
python run_e2e_tests.py --mode e2e --data multilingual_corpus.txt --series 1
```

### Полный прогон (GPU ≥ 12 ГБ VRAM)

```bash
# 1. Загрузить полный датасет
python prepare_dataset.py --ru 7000 --en 2000 --zh 1000

# 2. Запустить все серии
python run_e2e_tests.py --mode full --data multilingual_corpus.txt

# 3. Запустить только серию 2
python run_e2e_tests.py --mode full --data multilingual_corpus.txt --series 2
```

Результаты сохраняются в `experiment_results.json`.

---

## Параметры e2e-масштабирования

В режиме `--mode e2e` все конфиги автоматически масштабируются для быстрого CPU-теста:

| Параметр         | Full GPU config | E2E CPU config |
|------------------|-----------------|----------------|
| embed_size       | оригинал        | оригинал / 8   |
| num_layers       | оригинал        | оригинал / 2   |
| batch_size       | оригинал        | 2              |
| epochs           | оригинал        | 1              |
| max_batches/epoch| не ограничено   | 3              |

---

## Метрики

Основная метрика — **cross-entropy loss** на обучающей выборке (языковое моделирование, next-token prediction). В полном прогоне рекомендуется также считать **perplexity** = exp(loss) на отдельном валидационном сплите.

---

## Ожидаемые результаты (полный прогон)

| Серия | Ожидаемый результат |
|-------|---------------------|
| 1     | Разрыв loss baseline vs compressed растёт с размером модели |
| 2     | Оптимальный compressor_layers ≈ num_layers // 2 основной модели |
| 3     | Компрессия начинает помогать при block_size ≥ 256 |
