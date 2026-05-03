"""
Обертка для использования моделей проекта с LM Evaluation Harness.
Позволяет проводить zero-shot тестирование на стандартных бенчмарках (MMLU, HellaSwag и др.).
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Any, Optional
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from model_architecture.transformer import Transformer, CompressedTransformer
from data_preparation.tokenizer import BPETokenizer

@register_model("custom_llm")
class CustomLLMWrapper(LM):
    """
    Класс-обертка для пользовательской LLM, обеспечивающий совместимость
    с фреймворком lm-eval.
    """
    def __init__(self, model_path: str, use_compression: bool = False, chunk_size: int = 8, device: str = "cuda") -> None:
        """
        Инициализирует обертку, загружает модель и токенизатор.

        Аргументы:
            model_path (str): Путь к файлу с весами модели.
            use_compression (bool): Использовать ли модель со сжатием контекста.
            chunk_size (int): Размер чанка для модели со сжатием.
            device (str): Устройство для вычислений ('cuda' или 'cpu').
        """
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_compression = use_compression
        self.chunk_size = chunk_size

        # Load tokenizer
        vocab_file = "pretrain_vocab.pt"
        self.tokenizer = BPETokenizer()
        import os
        if os.path.exists(vocab_file):
            self.tokenizer.load_vocab(vocab_file)
        else:
            print(f"Warning: Vocabulary file {vocab_file} not found. Ensure it is generated.")
            # Create a dummy one for testing purposes if it doesn't exist
            self.tokenizer.train("dummy text")

        src_vocab_size = self.tokenizer.current_id
        trg_vocab_size = self.tokenizer.current_id
        src_pad_idx = 0
        trg_pad_idx = 0
        embed_size = 256
        num_layers = 2
        heads = 8
        forward_expansion = 4
        dropout = 0.1
        max_length = 128

        if self.use_compression:
            self.model = CompressedTransformer(
                src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx,
                embed_size, num_layers, forward_expansion, heads, dropout, self.device, max_length,
                chunk_size=self.chunk_size
            ).to(self.device)
        else:
            self.model = Transformer(
                src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx,
                embed_size, num_layers, forward_expansion, heads, dropout, self.device, max_length
            ).to(self.device)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model weights from {model_path}")
        else:
            print(f"Warning: Model weights file {model_path} not found. Using uninitialized model.")

        self.model.eval()

    def _loglikelihood_tokens(self, requests: List[Any], disable_tqdm: bool = False) -> List[Tuple[float, bool]]:
        """
        Вычисляет логарифмическое правдоподобие для набора запросов.
        Основной метод для вычисления метрик accuracy на бенчмарках множественного выбора.

        Аргументы:
            requests (List[Any]): Список запросов из lm-eval.
            disable_tqdm (bool): Отключить ли tqdm.

        Возвращает:
            List[Tuple[float, bool]]: Список кортежей, где первый элемент - суммарный log-prob,
                                      второй - флаг, является ли продолжение наиболее вероятным (greedy).
        """
        res = []
        for req in requests:
            ctx, cont = req.args

            # encode string to ids
            ctx_enc = self.tokenizer.encode(ctx)
            cont_enc = self.tokenizer.encode(cont)

            # Check length to prevent shape errors
            if self.use_compression:
                 max_tokens = self.model.transformer.encoder.position_embedding.num_embeddings * self.chunk_size
            else:
                 max_tokens = self.model.encoder.position_embedding.num_embeddings

            # Simple handling: truncate context if too long
            inp = ctx_enc + cont_enc
            if len(inp) > max_tokens:
                inp = inp[-max_tokens:]

            inp_tensor = torch.tensor(inp, dtype=torch.long, device=self.device).unsqueeze(0)

            with torch.no_grad():
                output = self.model(inp_tensor, inp_tensor)

            # Loglikelihood calculation (simplified)
            # Find the log probability of the continuation given the context
            # Shift output logits and input targets
            logits = output[0, :-1, :]
            target = inp_tensor[0, 1:]

            # Calculate log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # Get the log probabilities for the actual continuation tokens
            cont_len = len(cont_enc)
            # The indices in log_probs corresponding to the continuation
            start_idx = len(inp) - cont_len - 1

            if start_idx < 0:
                start_idx = 0 # Truncated context

            cont_log_probs = []
            for i in range(cont_len):
                idx = start_idx + i
                if idx < len(log_probs):
                    token = target[idx]
                    prob = log_probs[idx, token].item()
                    cont_log_probs.append(prob)

            # Sum log probs for the continuation
            total_log_prob = sum(cont_log_probs)

            # Is greedy? True if the model's top choice matches the continuation
            is_greedy = True
            for i in range(cont_len):
                idx = start_idx + i
                if idx < len(logits):
                    top_token = torch.argmax(logits[idx]).item()
                    if top_token != target[idx].item():
                        is_greedy = False
                        break

            res.append((total_log_prob, is_greedy))

        return res

    def loglikelihood(self, requests: List[Any]) -> List[Tuple[float, bool]]:
        """
        Обертка над _loglikelihood_tokens для lm-eval.
        """
        return self._loglikelihood_tokens(requests)

    def loglikelihood_rolling(self, requests: List[Any]) -> List[float]:
        """
        Вычисляет rolling log-likelihood для перплексии.
        Пока реализовано как заглушка.
        """
        return [-1.0 for _ in requests]

    def generate_until(self, requests: List[Any]) -> List[str]:
        """
        Генерирует текст для набора запросов. Используется для бенчмарков с открытым ответом.

        Аргументы:
            requests (List[Any]): Список запросов.

        Возвращает:
            List[str]: Список сгенерированных строк.
        """
        res = []
        for req in requests:
            ctx = req.args[0]
            # Use generate.py logic here
            from inference.generate import generate_text
            gen = generate_text(self.model, self.tokenizer, ctx, max_new_tokens=20, device=self.device)
            res.append(gen)
        return res
