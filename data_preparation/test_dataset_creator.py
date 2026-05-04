import pytest
import os
import torch
from unittest.mock import patch, MagicMock
from torch.utils.data.distributed import DistributedSampler
from data_preparation.dataset_creator import create_dataset

def test_create_dataset_distributed():
    # Setup dummy file
    with open("dummy_test_data.txt", "w") as f:
        f.write("Some dummy test text for tokenization.")

    # Patch torch distributed as DistributedSampler needs a process group
    with patch("torch.distributed.is_available", return_value=True), \
         patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_world_size", return_value=2), \
         patch("torch.distributed.get_rank", return_value=0):

        dataloader, vocab_size = create_dataset("dummy_test_data.txt", "dummy_vocab.pt", block_size=4, batch_size=2, is_distributed=True)

        assert isinstance(dataloader.sampler, DistributedSampler)
        assert dataloader.batch_size == 2

    os.remove("dummy_test_data.txt")
    if os.path.exists("dummy_vocab.pt"):
        os.remove("dummy_vocab.pt")
