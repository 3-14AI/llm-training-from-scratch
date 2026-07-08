import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.run_inference import main

@patch('scripts.run_inference.argparse.ArgumentParser.parse_args')
@patch('scripts.run_inference.torch.device')
@patch('scripts.run_inference.os.path.exists')
@patch('scripts.run_inference.BPETokenizer')
@patch('scripts.run_inference.Transformer')
@patch('scripts.run_inference.CompressedTransformer')
@patch('scripts.run_inference.generate_text')
@patch('scripts.run_inference.torch.load')
def test_run_inference_success(mock_load, mock_generate, mock_compressed, mock_transformer, mock_tokenizer, mock_exists, mock_device, mock_parse_args, capsys):
    mock_args = MagicMock()
    mock_args.prompt = "test prompt"
    mock_args.max_tokens = 10
    mock_args.temperature = 0.5
    mock_args.model_path = "test_model.pth"
    mock_args.use_compression = False
    mock_args.chunk_size = 8
    mock_parse_args.return_value = mock_args

    mock_exists.return_value = True

    mock_generate.return_value = "generated test output"

    main()

    captured = capsys.readouterr()
    output_json = json.loads(captured.out)

    assert output_json["status"] == "success"
    assert output_json["prompt"] == "test prompt"
    assert output_json["generated_text"] == "generated test output"

@patch('scripts.run_inference.argparse.ArgumentParser.parse_args')
@patch('scripts.run_inference.generate_text')
@patch('scripts.run_inference.os.path.exists')
@patch('scripts.run_inference.BPETokenizer')
def test_run_inference_error(mock_tokenizer, mock_exists, mock_generate, mock_parse_args, capsys):
    mock_args = MagicMock()
    mock_args.prompt = "test prompt"
    mock_args.max_tokens = 10
    mock_args.temperature = 0.5
    mock_args.model_path = ""
    mock_args.use_compression = False
    mock_args.chunk_size = 8
    mock_parse_args.return_value = mock_args

    mock_exists.return_value = False

    mock_generate.side_effect = Exception("Test generation error")

    main()

    captured = capsys.readouterr()
    output_json = json.loads(captured.out)

    assert output_json["status"] == "error"
    assert "Test generation error" in output_json["error"]

@patch('scripts.run_inference.argparse.ArgumentParser.parse_args')
@patch('scripts.run_inference.torch.device')
@patch('scripts.run_inference.os.path.exists')
@patch('scripts.run_inference.BPETokenizer')
@patch('scripts.run_inference.CompressedTransformer')
@patch('scripts.run_inference.generate_text')
def test_run_inference_compressed(mock_generate, mock_compressed, mock_tokenizer, mock_exists, mock_device, mock_parse_args, capsys):
    mock_args = MagicMock()
    mock_args.prompt = "test"
    mock_args.max_tokens = 5
    mock_args.temperature = 0.7
    mock_args.model_path = ""
    mock_args.use_compression = True
    mock_args.chunk_size = 4
    mock_parse_args.return_value = mock_args

    mock_exists.return_value = False
    mock_generate.return_value = "output"

    main()

    captured = capsys.readouterr()
    output_json = json.loads(captured.out)

    assert output_json["status"] == "success"
    mock_compressed.assert_called_once()
