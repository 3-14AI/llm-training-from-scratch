import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Create dummy modules for mocking
sys.modules['monitoring'] = MagicMock()
sys.modules['monitoring.train_monitor'] = MagicMock()
sys.modules['monitoring.evaluation'] = MagicMock()

import torch
import deepspeed
from pretraining.pretrain import main, train_model

@patch('pretraining.pretrain.create_dataset')
@patch('pretraining.pretrain.train_model')
@patch('deepspeed.initialize')
@patch('deepspeed.init_distributed')
def test_pretrain_main_deepspeed(mock_init_dist, mock_ds_init, mock_train_model, mock_create_dataset):
    mock_create_dataset.return_value = (MagicMock(), 100) # Mock dataloader and vocab size
    mock_ds_init.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

    test_args = ["pretrain.py", "--deepspeed", "--local_rank", "0"]
    with patch.object(sys, 'argv', test_args), patch('torch.save'), patch('os.remove'):
        main()

    mock_init_dist.assert_called_once()
    mock_ds_init.assert_called_once()

    # Train model should be called with is_deepspeed=True
    assert mock_train_model.call_args[1].get('is_deepspeed') is True

def test_train_model_deepspeed_step():
    model = MagicMock()
    model.backward = MagicMock()
    model.step = MagicMock()

    dataloader = [ (torch.ones(1, 10, dtype=torch.long), torch.ones(1, 10, dtype=torch.long)) ]

    optimizer = MagicMock()
    criterion = MagicMock(return_value=torch.tensor(1.0))

    train_model(model, dataloader, optimizer, criterion, "cpu", epochs=1, is_deepspeed=True)

    model.backward.assert_called_once()
    model.step.assert_called_once()
    optimizer.step.assert_not_called()
    optimizer.zero_grad.assert_not_called()
