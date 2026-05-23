"""
Скрипт для запуска оценки модели на бенчмарках с использованием LM Evaluation Harness.
"""
import argparse
import json
import os
import sys
import torch

# Ensure local modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lm_eval import simple_evaluate
from evaluation_module.lm_eval_wrapper import CustomLLMWrapper

def main() -> None:
    """
    Основная функция для запуска оценки.
    """
    parser = argparse.ArgumentParser(description="Оценка LLM с использованием LM Evaluation Harness.")
    parser.add_argument("--model_path", type=str, required=True, help="Путь к весам обученной модели")
    parser.add_argument("--use_compression", action="store_true", help="Использует ли модель сжатие контекста")
    parser.add_argument("--chunk_size", type=int, default=8, help="Размер чанка (если используется сжатие)")
    parser.add_argument("--tasks", type=str, default="hellaswag", help="Список задач через запятую (например, 'hellaswag,mmlu')")
    parser.add_argument("--limit", type=int, default=None, help="Ограничение количества примеров для каждой задачи (для тестирования)")
    parser.add_argument("--output_path", type=str, default="eval_results.json", help="Путь для сохранения результатов оценки")
    parser.add_argument("--device", type=str, default="cuda", help="Устройство для вычислений (cuda или cpu)")

    args = parser.parse_args()

    print(f"Инициализация модели из {args.model_path}...")
    model_wrapper = CustomLLMWrapper(
        model_path=args.model_path,
        use_compression=args.use_compression,
        chunk_size=args.chunk_size,
        device=args.device
    )

    tasks_list = args.tasks.split(",")
    print(f"Запуск оценки на задачах: {tasks_list}")

    # Run evaluation
    results = simple_evaluate(
        model=model_wrapper,
        tasks=tasks_list,
        limit=args.limit
    )

    # Print summary
    print("\nРезультаты оценки:")
    for task, task_res in results['results'].items():
        print(f"\nЗадача: {task}")
        for metric, value in task_res.items():
            if not metric.endswith("stderr") and metric != "alias":
                print(f"  {metric}: {value:.4f}")

    # Save to file
    with open(args.output_path, "w", encoding="utf-8") as f:
        # Avoid saving full config as it might contain non-serializable objects
        json.dump(results['results'], f, indent=4, ensure_ascii=False)
    print(f"\nРезультаты сохранены в {args.output_path}")

if __name__ == "__main__":
    main()
