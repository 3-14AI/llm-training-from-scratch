"""
Тесты для проверки интеграции с LM Evaluation Harness.
"""
import os
import unittest
from unittest.mock import patch
from evaluation.lm_eval_wrapper import CustomLLMWrapper

class TestEvaluationIntegration(unittest.TestCase):
    """
    Тестовый класс для проверки обертки CustomLLMWrapper.
    """
    def setUp(self) -> None:
        """
        Создает фиктивный файл словаря для предотвращения ошибок при инициализации токенизатора.
        """
        with open("pretrain_vocab.pt", "w") as f:
            f.write("dummy")

    def tearDown(self) -> None:
        """
        Удаляет фиктивный файл словаря.
        """
        if os.path.exists("pretrain_vocab.pt"):
            os.remove("pretrain_vocab.pt")

    @patch("evaluation.lm_eval_wrapper.BPETokenizer.load_vocab")
    def test_lm_eval_wrapper_initialization(self, mock_load) -> None:
        """
        Проверяет, что обертка успешно инициализируется со стандартной моделью.
        """
        wrapper = CustomLLMWrapper(model_path="dummy.pth", use_compression=False, device="cpu")
        self.assertIsNotNone(wrapper.model)

    @patch("evaluation.lm_eval_wrapper.BPETokenizer.load_vocab")
    def test_lm_eval_wrapper_compression_initialization(self, mock_load) -> None:
        """
        Проверяет, что обертка успешно инициализируется с моделью со сжатием контекста.
        """
        wrapper = CustomLLMWrapper(model_path="dummy.pth", use_compression=True, chunk_size=4, device="cpu")
        self.assertIsNotNone(wrapper.model)
        self.assertEqual(wrapper.model.chunk_size, 4)

if __name__ == "__main__":
    unittest.main()
