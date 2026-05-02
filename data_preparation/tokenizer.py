import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
from typing import List, Union

class BPETokenizer:
    """
    Класс для токенизации текста с использованием алгоритма BPE (Byte-Pair Encoding).
    """
    def __init__(self, vocab_size: int = 10000) -> None:
        """
        Инициализирует токенизатор BPE.

        Аргументы:
            vocab_size (int): Желаемый размер словаря.
        """
        self.vocab_size = vocab_size
        self._tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self._tokenizer.pre_tokenizer = Whitespace()
        self.pad_id = 0
        self.unk_id = 1
        self.sos_id = 2
        self.eos_id = 3

    def train(self, text: str) -> None:
        """
        Обучает токенизатор на предоставленном тексте.

        Аргументы:
            text (str): Текст для обучения токенизатора.
        """
        trainer = BpeTrainer(special_tokens=["<pad>", "<unk>", "<sos>", "<eos>"], vocab_size=self.vocab_size)
        self._tokenizer.train_from_iterator([text], trainer=trainer)

    def encode(self, text: str) -> List[int]:
        """
        Кодирует текст в список идентификаторов токенов.

        Аргументы:
            text (str): Входной текст для кодирования.

        Возвращает:
            List[int]: Список идентификаторов токенов.
        """
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Декодирует список идентификаторов токенов обратно в текст.

        Аргументы:
            token_ids (List[int]): Список идентификаторов токенов.

        Возвращает:
            str: Раскодированный текст.
        """
        return self._tokenizer.decode(token_ids)

    def save_vocab(self, path: str) -> None:
        """
        Сохраняет словарь токенизатора по указанному пути.

        Аргументы:
            path (str): Путь для сохранения словаря.
        """
        self._tokenizer.save(path)

    def load_vocab(self, path: str) -> None:
        """
        Загружает словарь токенизатора из указанного пути.

        Аргументы:
            path (str): Путь к файлу словаря.

        Исключения:
            FileNotFoundError: Если файл не найден.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary not found at {path}")
        self._tokenizer = Tokenizer.from_file(path)
        self.vocab_size = self._tokenizer.get_vocab_size()

    @property
    def current_id(self) -> int:
        """
        Свойство, возвращающее текущий размер словаря.

        Возвращает:
            int: Текущий размер словаря.
        """
        return self._tokenizer.get_vocab_size()
