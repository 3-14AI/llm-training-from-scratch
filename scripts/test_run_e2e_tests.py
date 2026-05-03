import pytest
from unittest.mock import patch, MagicMock, mock_open
import torch
import torch.nn as nn
import sys
import os
import argparse

sys.path.insert(0, os.path.abspath("scripts"))
from run_e2e_tests import train_one_epoch, run_experiment, main

def test_train_one_epoch():
    model = MagicMock(spec=nn.Module)
    model.return_value = torch.zeros((2, 10, 50), requires_grad=True)
    dataloader = [(torch.randint(0, 50, (2, 10)), torch.randint(0, 50, (2, 11)))]
    optimizer = MagicMock()
    criterion = MagicMock(return_value=torch.tensor(0.5, requires_grad=True))

    loss = train_one_epoch(model, dataloader, optimizer, criterion, torch.device("cpu"))
    assert loss == 0.5
    assert optimizer.step.called

@patch("run_e2e_tests.create_dataset", return_value=([], 100))
@patch("run_e2e_tests.build_model")
@patch("run_e2e_tests.train_one_epoch", return_value=1.5)
@patch("run_e2e_tests.Adam")
def test_run_experiment(mock_adam, mock_train, mock_build, mock_create):
    mock_model = MagicMock()
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
    mock_build.return_value = mock_model

    cfg = {"embed_size": 32, "num_layers": 2, "heads": 2, "use_compression": False, "max_length": 64, "block_size": 64, "batch_size": 2, "epochs": 1, "lr": 0.001}
    result = run_experiment("test", cfg, "data", "vocab", torch.device("cpu"))

    assert result["exp_name"] == "test"
    assert result["status"] == "ok"

@patch("argparse.ArgumentParser.parse_args")
@patch("os.path.exists", return_value=True)
@patch("run_e2e_tests.run_experiment")
@patch("run_e2e_tests.json.dump")
def test_main(mock_dump, mock_run, mock_exists, mock_args):
    mock_args.return_value = argparse.Namespace(mode="e2e", data="dummy.txt", vocab="vocab.pt", output="out.json", series=None)
    mock_run.return_value = {"status": "ok", "n_params": 100, "final_loss": 1.5, "elapsed_seconds": 10}

    with patch("run_e2e_tests.get_e2e_configs", return_value={"test_exp": {"epochs": 1}}):
        with patch("run_e2e_tests.open", mock_open()):
            main()

    mock_run.assert_called()
