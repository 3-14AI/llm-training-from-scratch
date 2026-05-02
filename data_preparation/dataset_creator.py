import torch
from torch.utils.data import Dataset, DataLoader
from data_preparation.tokenizer import BPETokenizer
from typing import List, Tuple, Any

class TextDataset(Dataset):
    """
    Пользовательский класс Dataset для текстовых данных.
    Разбивает токенизированные данные на блоки фиксированного размера для обучения модели.
    """
    def __init__(self, tokenized_data: List[int], block_size: int) -> None:
        """
        Инициализирует TextDataset.

        Аргументы:
            tokenized_data (List[int]): Список токенизированных данных (идентификаторы).
            block_size (int): Размер блока (последовательности) для обучения.
        """
        self.tokenized_data = tokenized_data
        self.block_size = block_size

    def __len__(self) -> int:
        """
        Возвращает количество доступных примеров (блоков) в наборе данных.

        Возвращает:
            int: Количество блоков.
        """
        if len(self.tokenized_data) <= self.block_size:
            return 0
        return len(self.tokenized_data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Получает один пример (входную и целевую последовательности) по индексу.

        Аргументы:
            idx (int): Индекс примера.

        Возвращает:
            Tuple[torch.Tensor, torch.Tensor]: Кортеж, содержащий тензор входов и тензор целей (сдвинутых на один шаг).
        """
        chunk = self.tokenized_data[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def create_dataset(text_file_path: str, vocab_path: str, block_size: int = 128, batch_size: int = 32) -> Tuple[DataLoader, int]:
    """
    Создает DataLoader и возвращает размер словаря токенизатора для заданного текстового файла.
    Если словарь по пути vocab_path не найден, обучает новый токенизатор и сохраняет его.

    Аргументы:
        text_file_path (str): Путь к файлу с текстом корпуса.
        vocab_path (str): Путь к файлу словаря (для сохранения/загрузки).
        block_size (int, optional): Размер обучающего блока (по умолчанию 128).
        batch_size (int, optional): Размер батча (по умолчанию 32).

    Возвращает:
        Tuple[DataLoader, int]: Кортеж, содержащий DataLoader и размер словаря.
    """
    # Load or train tokenizer
    tokenizer = BPETokenizer()
    try:
        tokenizer.load_vocab(vocab_path)
        print(f"Loaded vocabulary from {vocab_path}")
    except (FileNotFoundError, RuntimeError):
        print(f"Vocabulary not found or invalid at {vocab_path}. Training new tokenizer...")
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokenizer.train(text)
        tokenizer.save_vocab(vocab_path)
        print(f"Trained and saved vocabulary to {vocab_path}")

    # Tokenize the entire text
    with open(text_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenized_text = tokenizer.encode(text)

    # Create dataset and dataloader
    dataset = TextDataset(tokenized_text, block_size)
    if len(dataset) == 0:
        # Fallback for very short text
        print("Warning: Text is shorter than block_size. Padding...")
        tokenized_text = tokenized_text + [0] * (block_size + 1 - len(tokenized_text))
        dataset = TextDataset(tokenized_text, block_size)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, tokenizer.current_id
