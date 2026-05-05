import pytest
import os
import sys
from unittest.mock import patch, MagicMock
import torch
from fine_tuning.finetune import train_model

# Create dummy modules for mocking
sys.modules['monitoring'] = MagicMock()
sys.modules['monitoring.train_monitor'] = MagicMock()
sys.modules['monitoring.evaluation'] = MagicMock()
sys.modules['wandb'] = MagicMock()

def test_checkpoint_saving_finetune():
    model = MagicMock()
    model.state_dict.return_value = {'model_key': 'model_val'}

    optimizer = MagicMock()
    optimizer.state_dict.return_value = {'opt_key': 'opt_val'}

    scheduler = MagicMock()
    scheduler.state_dict.return_value = {'sched_key': 'sched_val'}

    dataloader = [ (torch.ones(1, 10, dtype=torch.long), torch.ones(1, 10, dtype=torch.long)) ]
    criterion = MagicMock(return_value=torch.tensor(1.0, requires_grad=True))

    with patch('torch.save') as mock_save:
        train_model(model, dataloader, optimizer, criterion, "cpu", scheduler=scheduler, epochs=1, checkpoint_dir="test_checkpoints_finetune")

    assert mock_save.called
    args, kwargs = mock_save.call_args
    checkpoint_data = args[0]

    assert 'epoch' in checkpoint_data
    assert checkpoint_data['epoch'] == 1
    assert 'global_step' in checkpoint_data
    assert checkpoint_data['global_step'] == 1
    assert 'model_state_dict' in checkpoint_data
    assert checkpoint_data['model_state_dict'] == {'model_key': 'model_val'}
    assert 'optimizer_state_dict' in checkpoint_data
    assert checkpoint_data['optimizer_state_dict'] == {'opt_key': 'opt_val'}
    assert 'scheduler_state_dict' in checkpoint_data
    assert checkpoint_data['scheduler_state_dict'] == {'sched_key': 'sched_val'}
