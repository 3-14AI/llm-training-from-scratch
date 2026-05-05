import pytest
import os
import sys
from unittest.mock import patch, MagicMock
import torch

from pretraining.pretrain import main, train_model

# Create dummy modules for mocking
sys.modules['monitoring'] = MagicMock()
sys.modules['monitoring.train_monitor'] = MagicMock()
sys.modules['monitoring.evaluation'] = MagicMock()
sys.modules['deepspeed'] = MagicMock()
sys.modules['wandb'] = MagicMock()


@patch('pretraining.pretrain.create_dataset')
@patch('pretraining.pretrain.train_model')
@patch('deepspeed.initialize')
@patch('deepspeed.init_distributed')
@patch('pretraining.pretrain.deepspeed')
def test_pretrain_main_deepspeed(mock_deepspeed, mock_init_dist, mock_ds_init, mock_train_model, mock_create_dataset):
    mock_create_dataset.return_value = (MagicMock(), 100)
    mock_deepspeed.initialize.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

    test_args = ["pretrain.py", "--deepspeed", "--local_rank", "0"]
    with patch.object(sys, 'argv', test_args), patch('torch.save'), patch('os.remove'):
        main()

    mock_deepspeed.init_distributed.assert_called_once()
    mock_deepspeed.initialize.assert_called_once()

    # Train model should be called with is_deepspeed=True
    assert mock_train_model.call_args[1].get('is_deepspeed') is True

@patch('pretraining.pretrain.deepspeed')
def test_train_model_deepspeed_step(mock_deepspeed):
    model = MagicMock()
    model.backward = MagicMock()
    model.step = MagicMock()

    dataloader = [ (torch.ones(1, 10, dtype=torch.long), torch.ones(1, 10, dtype=torch.long)) ]

    optimizer = MagicMock()
    criterion = MagicMock(return_value=torch.tensor(1.0, requires_grad=True))

    with patch('torch.save') as mock_save:
        train_model(model, dataloader, optimizer, criterion, "cpu", epochs=1, is_deepspeed=True)

    model.backward.assert_called_once()
    model.step.assert_called_once()
    optimizer.step.assert_not_called()
    optimizer.zero_grad.assert_not_called()

def test_checkpoint_saving():
    model = MagicMock()
    model.state_dict.return_value = {'model_key': 'model_val'}

    optimizer = MagicMock()
    optimizer.state_dict.return_value = {'opt_key': 'opt_val'}

    scheduler = MagicMock()
    scheduler.state_dict.return_value = {'sched_key': 'sched_val'}

    dataloader = [ (torch.ones(1, 10, dtype=torch.long), torch.ones(1, 10, dtype=torch.long)) ]
    criterion = MagicMock(return_value=torch.tensor(1.0, requires_grad=True))

    with patch('torch.save') as mock_save:
        train_model(model, dataloader, optimizer, criterion, "cpu", scheduler=scheduler, epochs=1, is_deepspeed=False, checkpoint_dir="test_checkpoints")

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
