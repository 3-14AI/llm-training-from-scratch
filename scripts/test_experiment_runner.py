import pytest
from unittest.mock import patch, MagicMock, mock_open
import torch
import torch.nn as nn
import sys
import os
import argparse

sys.path.insert(0, os.path.abspath("scripts"))
from experiment_runner import train_one_epoch, run_experiment, main

def test_train_one_epoch():
    model = MagicMock(spec=nn.Module)
    output_tensor = torch.zeros((2, 10, 50), requires_grad=True)
    model.return_value = output_tensor

    src = torch.randint(0, 50, (2, 10))
    trg = torch.randint(0, 50, (2, 11))

    dataloader = [(src, trg), (src, trg)]
    optimizer = MagicMock()
    criterion = MagicMock()

    loss_tensor = torch.tensor(0.5, requires_grad=True)
    criterion.return_value = loss_tensor

    device = torch.device("cpu")

    loss = train_one_epoch(model, dataloader, optimizer, criterion, device)

    assert loss == 0.5
    assert model.train.called
    assert optimizer.zero_grad.call_count == 2
    assert optimizer.step.call_count == 2

@patch("experiment_runner.create_dataset")
@patch("experiment_runner.build_model")
@patch("experiment_runner.train_one_epoch")
@patch("experiment_runner.Adam")
def test_run_experiment(mock_adam, mock_train, mock_build, mock_create):
    mock_create.return_value = ([], 100)

    mock_model = MagicMock(spec=nn.Module)
    param_mock = torch.nn.Parameter(torch.zeros(1))
    mock_model.parameters.return_value = [param_mock]
    mock_build.return_value = mock_model

    mock_train.return_value = 1.5

    cfg = {"embed_size": 32, "num_layers": 2, "heads": 2, "max_length": 64}

    result = run_experiment("test_exp", cfg, "data.txt", "vocab.pt", torch.device("cpu"))

    assert result["exp_name"] == "test_exp"
    assert result["status"] == "ok"
    assert result["final_loss"] == 1.5

@patch("argparse.ArgumentParser.parse_args")
@patch("os.path.exists")
@patch("experiment_runner.run_experiment")
@patch("experiment_runner.json.dump")
def test_main(mock_dump, mock_run, mock_exists, mock_args):
    mock_args.return_value = argparse.Namespace(
        mode="e2e", data="dummy.txt", vocab="vocab.pt",
        output="out.json", exp_name=None, series=None, log_file=None, epochs=None,
        use_compression=False, embed_size=None, num_layers=None, heads=None, forward_expansion=4, dropout=0.1, max_length=128, batch_size=32, lr=0.0003, chunk_size=8, compressor_layers=2
    )
    mock_exists.return_value = True
    mock_run.return_value = {"status": "ok", "n_params": 100, "final_loss": 1.5, "elapsed_seconds": 10}

    with patch("experiment_runner.get_e2e_configs", return_value={"test_exp": {"epochs": 1}}):
        with patch("experiment_runner.open", mock_open()):
            main()

    mock_run.assert_called()
