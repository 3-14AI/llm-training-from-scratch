from typing import Union
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    """
    Модуль Multi-Head Self Attention для трансформера.
    """
    def __init__(self, embed_size: int, heads: int) -> None:
        """
        Инициализирует MultiHeadSelfAttention.

        Аргументы:
            embed_size (int): Размерность эмбеддингов.
            heads (int): Количество голов внимания.
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values: torch.Tensor, keys: torch.Tensor, query: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Выполняет прямой проход механизма внимания.

        Аргументы:
            values (torch.Tensor): Тензор значений (Values).
            keys (torch.Tensor): Тензор ключей (Keys).
            query (torch.Tensor): Тензор запросов (Query).
            mask (Optional[torch.Tensor]): Маска внимания.

        Возвращает:
            torch.Tensor: Выходной тензор после механизма внимания.
        """
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Einsum does matrix multiplication for query @ key.T
        # query shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # out shape: (N, query_len, heads, head_dim)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    """
    Один блок трансформера (энкодера).
    """
    def __init__(self, embed_size: int, heads: int, dropout: float, forward_expansion: int) -> None:
        """
        Инициализирует блок трансформера.

        Аргументы:
            embed_size (int): Размерность эмбеддингов.
            heads (int): Количество голов внимания.
            dropout (float): Вероятность dropout.
            forward_expansion (int): Коэффициент расширения размерности в Feed Forward слое.
        """
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor, key: torch.Tensor, query: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Выполняет прямой проход блока трансформера.

        Аргументы:
            value (torch.Tensor): Тензор значений.
            key (torch.Tensor): Тензор ключей.
            query (torch.Tensor): Тензор запросов.
            mask (Optional[torch.Tensor]): Маска внимания.

        Возвращает:
            torch.Tensor: Выходной тензор блока.
        """
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class ContextCompressor(nn.Module):
    """
    Модуль для сжатия контекста (компрессор контекста).
    Разбивает входную последовательность на блоки и сжимает каждый блок в один вектор.
    """
    def __init__(
        self, vocab_size: int, embed_size: int, num_layers: int, heads: int, device: Union[str, torch.device], forward_expansion: int, dropout: float, chunk_size: int
    ) -> None:
        """
        Инициализирует компрессор контекста.

        Аргументы:
            vocab_size (int): Размер словаря.
            embed_size (int): Размерность эмбеддингов.
            num_layers (int): Количество слоев в компрессоре.
            heads (int): Количество голов внимания.
            device (Union[str, torch.device]): Устройство (CPU/GPU) для вычислений.
            forward_expansion (int): Расширение размерности в FF слоях.
            dropout (float): Вероятность dropout.
            chunk_size (int): Размер блока (chunk), который сжимается в один вектор.
        """
        super(ContextCompressor, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.chunk_size = chunk_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(chunk_size, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, heads, dropout=dropout, forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        
        # Learnable query for compression - one per chunk
        self.compress_query = nn.Parameter(torch.randn(1, 1, embed_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Выполняет прямой проход компрессора.

        Аргументы:
            x (torch.Tensor): Входной тензор токенов формы (N, seq_length).

        Возвращает:
            torch.Tensor: Тензор сжатых эмбеддингов формы (N, num_chunks, embed_size).
        """
        N, seq_length = x.shape
        # x shape: (N, seq_length)
        # We assume seq_length is a multiple of chunk_size for simplicity
        # or we pad/truncate accordingly in the wrapper.
        
        num_chunks = seq_length // self.chunk_size
        # Reshape to (N * num_chunks, chunk_size)
        x_reshaped = x.view(N * num_chunks, self.chunk_size)
        
        positions = torch.arange(0, self.chunk_size).expand(N * num_chunks, self.chunk_size).to(self.device)
        out = self.dropout(self.word_embedding(x_reshaped) + self.position_embedding(positions))

        for layer in self.layers:
            # Standard self-attention within chunk
            out = layer(out, out, out, None)
        
        # Now compress each chunk into one vector using the learnable query
        # out: (N * num_chunks, chunk_size, embed_size)
        # query: (N * num_chunks, 1, embed_size)
        query = self.compress_query.expand(N * num_chunks, 1, self.embed_size)
        
        # Use simple attention mechanism for compression instead of mean pooling
        # Compute dot product between query and the chunk's hidden states
        attention_scores = torch.bmm(query, out.transpose(1, 2)) / (self.embed_size ** 0.5) # (N*num_chunks, 1, chunk_size)
        attention_weights = torch.softmax(attention_scores, dim=-1) # (N*num_chunks, 1, chunk_size)

        compressed = torch.bmm(attention_weights, out) # (N*num_chunks, 1, embed_size)
        
        # Reshape back to (N, num_chunks, embed_size)
        compressed = compressed.view(N, num_chunks, self.embed_size)
        return compressed

class Encoder(nn.Module):
    """
    Энкодер трансформера.
    """
    def __init__(
        self, vocab_size: int, embed_size: int, num_layers: int, heads: int, device: Union[str, torch.device], forward_expansion: int, dropout: float, max_length: int
    ) -> None:
        """
        Инициализирует энкодер.

        Аргументы:
            vocab_size (int): Размер словаря.
            embed_size (int): Размерность эмбеддингов.
            num_layers (int): Количество слоев энкодера.
            heads (int): Количество голов внимания.
            device (Union[str, torch.device]): Устройство.
            forward_expansion (int): Расширение в FF слоях.
            dropout (float): Dropout.
            max_length (int): Максимальная длина последовательности для позиционных эмбеддингов.
        """
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, heads, dropout=dropout, forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Optional[torch.Tensor], mask: Optional[torch.Tensor], input_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Прямой проход энкодера.

        Аргументы:
            x (Optional[torch.Tensor]): Входной тензор токенов. Может быть None, если переданы input_embeddings.
            mask (Optional[torch.Tensor]): Маска внимания.
            input_embeddings (Optional[torch.Tensor]): Опциональные готовые эмбеддинги.

        Возвращает:
            torch.Tensor: Выходные представления энкодера.
        """
        if input_embeddings is not None:
            # Use provided embeddings (e.g. from compressor)
            out = input_embeddings
            N, seq_length, _ = out.shape
        else:
            N, seq_length = x.shape
            out = self.word_embedding(x)
            
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(out + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module):
    """
    Блок декодера трансформера.
    """
    def __init__(self, embed_size: int, heads: int, forward_expansion: int, dropout: float, device: Union[str, torch.device]) -> None:
        """
        Инициализирует блок декодера.
        """
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, value: torch.Tensor, key: torch.Tensor, src_mask: Optional[torch.Tensor], trg_mask: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход блока декодера.
        """
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    """
    Декодер трансформера.
    """
    def __init__(
        self, vocab_size: int, embed_size: int, num_layers: int, heads: int, forward_expansion: int, dropout: float, device: Union[str, torch.device], max_length: int
    ) -> None:
        """
        Инициализирует декодер.
        """
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, src_mask: Optional[torch.Tensor], trg_mask: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход декодера.
        """
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    """
    Стандартная модель Трансформера (seq2seq).
    """
    def __init__(
        self, src_vocab_size: int, trg_vocab_size: int, src_pad_idx: int, trg_pad_idx: int,
        embed_size: int = 256, num_layers: int = 6, forward_expansion: int = 4,
        heads: int = 8, dropout: float = 0.0, device: Union[str, torch.device] = "cuda", max_length: int = 100
    ) -> None:
        """
        Инициализирует модель Трансформер.
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length
        )

        self.decoder = Decoder(
            trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Создает маску для исходной последовательности (чтобы игнорировать pad-токены).
        """
        if src is None: return None
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        """
        Создает маску для целевой последовательности (чтобы предотвратить 'заглядывание' вперед).
        """
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход Трансформера.
        """
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

class CompressedTransformer(nn.Module):
    """
    Модель Трансформера со сжатием контекста.
    """
    def __init__(
        self, src_vocab_size: int, trg_vocab_size: int, src_pad_idx: int, trg_pad_idx: int,
        embed_size: int = 256, num_layers: int = 6, forward_expansion: int = 4,
        heads: int = 8, dropout: float = 0.0, device: Union[str, torch.device] = "cuda",
        max_length: int = 100, chunk_size: int = 8, compressor_layers: int = 2
    ) -> None:
        """
        Инициализирует модель CompressedTransformer.
        """
        super(CompressedTransformer, self).__init__()
        
        self.compressor = ContextCompressor(
            src_vocab_size, embed_size, compressor_layers, heads, device, forward_expansion, dropout, chunk_size
        )
        
        self.transformer = Transformer(
            src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, 
            embed_size, num_layers, forward_expansion, heads, dropout, device, max_length
        )
        
        self.chunk_size = chunk_size
        self.device = device

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход Трансформера.
        """
        # src shape: (N, src_len)
        # Ensure src_len is multiple of chunk_size
        N, src_len = src.shape
        if src_len % self.chunk_size != 0:
            padding_len = self.chunk_size - (src_len % self.chunk_size)
            src = F.pad(src, (0, padding_len), value=self.transformer.src_pad_idx)
        
        # Compress context
        compressed_embeddings = self.compressor(src) # (N, num_chunks, embed_size)
        
        # Forward through main transformer
        # We need to adapt the encoder to take embeddings directly
        trg_mask = self.transformer.make_trg_mask(trg)
        
        # Create a dummy src_mask for the compressed embeddings (all 1s since they are all valid)
        num_chunks = compressed_embeddings.shape[1]
        src_mask = torch.ones((N, 1, 1, num_chunks)).to(self.device)
        
        enc_src = self.transformer.encoder(None, src_mask, input_embeddings=compressed_embeddings)
        out = self.transformer.decoder(trg, enc_src, src_mask, trg_mask)
        
        return out

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    
    # Test original Transformer
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
    
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(device)
    out = model(x, trg[:, :-1])
    print(f"Original Transformer output shape: {out.shape}")
    
    # Test CompressedTransformer
    comp_model = CompressedTransformer(
        src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, 
        device=device, chunk_size=4
    ).to(device)
    out_comp = comp_model(x, trg[:, :-1])
    print(f"Compressed Transformer output shape: {out_comp.shape}")
