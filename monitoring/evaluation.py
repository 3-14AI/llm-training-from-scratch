import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Optional

def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Оценивает модель на валидационном наборе данных.

    Аргументы:
        model (nn.Module): Модель для оценки.
        dataloader (DataLoader): DataLoader с валидационными данными.
        criterion (nn.Module): Функция потерь.
        device (torch.device): Устройство (CPU или GPU) для вычислений.

    Возвращает:
        Tuple[float, float]: Среднее значение функции потерь (loss) и перплексия (perplexity).
    """
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg[:, :-1])

            output = output.reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            total_loss += loss.item()
            total_steps += 1

    avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, perplexity
