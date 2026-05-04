"""
Тесты для скрипта run_evaluation.py.
"""
import os
import unittest
from unittest.mock import patch, MagicMock
import sys

# Добавляем корневую директорию проекта в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.run_evaluation import main

class TestRunEvaluation(unittest.TestCase):
    @patch("scripts.run_evaluation.argparse.ArgumentParser.parse_args")
    @patch("scripts.run_evaluation.CustomLLMWrapper")
    @patch("scripts.run_evaluation.simple_evaluate")
    @patch("builtins.open")
    def test_main(self, mock_open, mock_simple_evaluate, mock_wrapper, mock_parse_args):
        mock_parse_args.return_value = MagicMock(
            model_path="dummy.pth",
            use_compression=False,
            chunk_size=8,
            tasks="hellaswag",
            limit=None,
            output_path="test_eval.json",
            device="cpu"
        )

        mock_simple_evaluate.return_value = {
            "results": {
                "hellaswag": {
                    "acc": 0.5,
                    "acc_stderr": 0.1,
                    "alias": "hellaswag"
                }
            }
        }

        main()

        mock_wrapper.assert_called_once_with(
            model_path="dummy.pth",
            use_compression=False,
            chunk_size=8,
            device="cpu"
        )
        mock_simple_evaluate.assert_called_once()
        mock_open.assert_called_once_with("test_eval.json", "w", encoding="utf-8")

if __name__ == "__main__":
    unittest.main()
