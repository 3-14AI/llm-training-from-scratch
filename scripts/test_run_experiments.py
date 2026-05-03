import pytest
from unittest.mock import patch, MagicMock, mock_open
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath("scripts"))
from run_experiments import train_model, run_experiment, main

def test_train_model():
    model = MagicMock(spec=nn.Module)
    model.return_value = torch.zeros((2, 10, 50), requires_grad=True)
    dataloader = [(torch.randint(0, 50, (2, 10)), torch.randint(0, 50, (2, 11)))]
    optimizer = MagicMock()
    criterion = MagicMock(return_value=torch.tensor(0.5, requires_grad=True))

    history = train_model(model, dataloader, optimizer, criterion, torch.device("cpu"), epochs=1)

    assert len(history) == 1
    assert history[0] == 0.5
    assert optimizer.step.called

@patch("run_experiments.create_dataset", return_value=(["dummy"], 100))
@patch("run_experiments.build_model")
@patch("run_experiments.train_model", return_value=[1.5])
@patch("run_experiments.Adam")
def test_run_experiment(mock_adam, mock_train, mock_build, mock_create):
    mock_model = MagicMock()
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
    mock_build.return_value = mock_model

    cfg = {"embed_size": 32, "num_layers": 2, "heads": 2, "use_compression": False, "max_length": 64, "block_size": 64, "batch_size": 2, "epochs": 1, "lr": 0.001}
    result = run_experiment("test", cfg, "data", "vocab", torch.device("cpu"))

    assert result["loss"] == 1.5
    assert "time_seconds" in result

@patch("run_experiments.run_experiment")
@patch("run_experiments.json.dump")
def test_main(mock_dump, mock_run):
    mock_run.return_value = {"loss": 1.5, "params": 100, "time_seconds": 10}
    with patch("run_experiments.EXPERIMENTS", {"test_exp": {"max_length": 64}}):
        with patch("run_experiments.open", mock_open()):
            main()
    mock_run.assert_called()
