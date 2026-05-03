import pytest
from unittest.mock import AsyncMock, patch, MagicMock, mock_open
from telegram import Update, User, Chat, Message
from telegram.ext import ContextTypes
import sys
import os

sys.path.insert(0, os.path.abspath("scripts"))
from telegram_bot import start_command, help_command, status_command
from telegram_bot import prepare_dataset_command, list_experiments_command, run_experiment_command
from telegram_bot import get_log_command, get_results_command, run_custom_command, monitor_task_log, main

@pytest.fixture
def update():
    mock_update = MagicMock(spec=Update)
    mock_user = MagicMock(spec=User)
    mock_user.id = 452526556
    mock_user.mention_html.return_value = "<a href='tg://user?id=452526556'>User</a>"
    mock_update.effective_user = mock_user

    mock_chat = MagicMock(spec=Chat)
    mock_chat.id = 12345
    mock_update.effective_chat = mock_chat

    mock_message = AsyncMock(spec=Message)
    mock_update.message = mock_message

    return mock_update

@pytest.fixture
def context():
    return MagicMock(spec=ContextTypes.DEFAULT_TYPE)

@pytest.mark.asyncio
async def test_start_command(update, context):
    await start_command(update, context)
    update.message.reply_html.assert_called_once()
    assert "Привет" in update.message.reply_html.call_args[0][0]

@pytest.mark.asyncio
async def test_help_command(update, context):
    await help_command(update, context)
    update.message.reply_text.assert_called_once()
    assert "Доступные команды" in update.message.reply_text.call_args[0][0]

@pytest.mark.asyncio
async def test_status_command_empty(update, context):
    with patch("telegram_bot.running_tasks", {}):
        await status_command(update, context)
        update.message.reply_text.assert_called_with("Нет активных задач.")

@pytest.mark.asyncio
async def test_status_command_with_tasks(update, context):
    from datetime import datetime, timedelta
    tasks = {
        "task_1": {
            "type": "custom",
            "name": "CustomTask",
            "status": "running",
            "start_time": datetime.now() - timedelta(minutes=5)
        }
    }
    with patch("telegram_bot.running_tasks", tasks):
        await status_command(update, context)
        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "task_1" in text
        assert "CustomTask" in text

@pytest.mark.asyncio
async def test_prepare_dataset_command(update, context):
    context.args = ["10", "20", "30"]
    mock_process = AsyncMock()
    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec, \
         patch("asyncio.create_task") as mock_task, \
         patch("telegram_bot.running_tasks", {}):

        await prepare_dataset_command(update, context)

        mock_exec.assert_called_once()
        assert "prepare_dataset.py" in mock_exec.call_args[0][1]
        assert "--ru" in mock_exec.call_args[0]
        assert "10" in mock_exec.call_args[0]
        update.message.reply_text.assert_called_once()
        mock_task.assert_called_once()

@pytest.mark.asyncio
async def test_list_experiments_command(update, context):
    await list_experiments_command(update, context)
    update.message.reply_text.assert_called()
    assert "Доступные эксперименты:" in update.message.reply_text.call_args_list[0][0][0]

@pytest.mark.asyncio
async def test_run_experiment_command_no_args(update, context):
    context.args = []
    await run_experiment_command(update, context)
    update.message.reply_text.assert_called_with("Пожалуйста, укажите имя эксперимента. Например: /run_experiment s1_small_baseline e2e")

@pytest.mark.asyncio
async def test_run_experiment_command_invalid_exp(update, context):
    context.args = ["invalid_exp"]
    await run_experiment_command(update, context)
    update.message.reply_text.assert_called_with("Эксперимент `invalid_exp` не найден в конфигурациях.")

@pytest.mark.asyncio
async def test_run_experiment_command_valid(update, context):
    context.args = ["s1_small_baseline"]
    mock_process = AsyncMock()
    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec, \
         patch("asyncio.create_task"), \
         patch("telegram_bot.running_tasks", {}), \
         patch("telegram_bot.ALL_EXPERIMENTS", {"s1_small_baseline": {}}):

        await run_experiment_command(update, context)

        mock_exec.assert_called_once()
        assert "experiment_runner.py" in mock_exec.call_args[0][1]
        assert "--exp_name" in mock_exec.call_args[0]
        assert "s1_small_baseline" in mock_exec.call_args[0]
        update.message.reply_text.assert_called_once()

@pytest.mark.asyncio
async def test_get_log_command_no_args(update, context):
    context.args = []
    await get_log_command(update, context)
    update.message.reply_text.assert_called_with("Пожалуйста, укажите ID задачи. Например: /get_log exp_1")

@pytest.mark.asyncio
async def test_get_log_command_not_found(update, context):
    context.args = ["invalid_id"]
    with patch("telegram_bot.running_tasks", {}):
        await get_log_command(update, context)
        update.message.reply_text.assert_called_with("Задача с ID `invalid_id` не найдена.")

@pytest.mark.asyncio
async def test_run_custom_command_missing_args(update, context):
    context.args = ["64", "2"]
    await run_custom_command(update, context)
    update.message.reply_text.assert_called_once()
    assert "Использование" in update.message.reply_text.call_args[0][0]

@pytest.mark.asyncio
async def test_run_custom_command_valid(update, context):
    context.args = ["64", "2", "4"]
    mock_process = AsyncMock()
    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec, \
         patch("asyncio.create_task"), \
         patch("telegram_bot.running_tasks", {}):

        await run_custom_command(update, context)
        mock_exec.assert_called_once()
        update.message.reply_text.assert_called_once()
        assert "Кастомный эксперимент" in update.message.reply_text.call_args[0][0]

@pytest.mark.asyncio
async def test_get_results_command_no_args(update, context):
    context.args = []
    await get_results_command(update, context)
    update.message.reply_text.assert_called_with("Пожалуйста, укажите ID эксперимента. Например: /get_results exp_1")

@pytest.mark.asyncio
async def test_get_results_command_not_found(update, context):
    context.args = ["invalid_id"]
    with patch("telegram_bot.running_tasks", {}):
        await get_results_command(update, context)
        update.message.reply_text.assert_called_with("Результаты для ID `invalid_id` не найдены.")
def test_main():
    with patch("telegram_bot.Application.builder") as mock_builder, \
         patch("os.environ.get", return_value="fake_token"):
        mock_app = MagicMock()
        mock_builder.return_value.token.return_value.build.return_value = mock_app
        main()
        mock_app.add_handler.assert_called()
        mock_app.run_polling.assert_called_once()

@pytest.mark.asyncio
async def test_monitor_task_log():
    from telegram_bot import monitor_task_log, running_tasks
    import asyncio

    mock_process = AsyncMock()
    mock_process.stdout = AsyncMock()
    mock_process.stdout.readline.side_effect = [b"Line 1\n", b"Saved line\n", b""]
    mock_process.wait.return_value = 0

    running_tasks["test_task_mon"] = {
        "process": mock_process,
        "chat_id": 123,
        "log_file": "dummy_log.txt",
        "name": "TestTask"
    }

    with patch("builtins.open", mock_open()):
        with patch("telegram_bot.Application") as mock_app:
            mock_bot = AsyncMock()
            mock_app.builder().token().build().bot = mock_bot
            await monitor_task_log("test_task_mon")

    assert running_tasks["test_task_mon"]["status"] == "completed"

@pytest.mark.asyncio
async def test_monitor_task_log_error():
    from telegram_bot import monitor_task_log, running_tasks
    import asyncio

    mock_process = AsyncMock()
    mock_process.stdout = AsyncMock()
    mock_process.stdout.readline.side_effect = [b""]
    mock_process.wait.return_value = 1

    running_tasks["test_task_err"] = {
        "process": mock_process,
        "chat_id": 123,
        "log_file": "dummy_log_err.txt",
        "name": "TestTask"
    }

    with patch("builtins.open", mock_open()):
        with patch("telegram_bot.Application") as mock_app:
            mock_bot = AsyncMock()
            mock_app.builder().token().build().bot = mock_bot
            await monitor_task_log("test_task_err")

    assert running_tasks["test_task_err"]["status"] == "error"


@pytest.mark.asyncio
async def test_get_log_command_truncate(update, context):
    context.args = ["task_1"]
    with patch("telegram_bot.running_tasks", {"task_1": {"log_file": "dummy.log"}}):
        with patch("os.path.exists", return_value=True):
            # simulate long log
            with patch("builtins.open", mock_open(read_data="A" * 5000)):
                await get_log_command(update, context)
                update.message.reply_document.assert_called_once()

@pytest.mark.asyncio
async def test_get_results_command_success(update, context):
    context.args = ["task_1"]
    tasks = {"task_1": {"results_file": "dummy.json"}}
    with patch("telegram_bot.running_tasks", tasks):
        with patch("os.path.exists", return_value=True):
            data = '{"task_1": {"status": "ok", "exp_name": "test", "n_params": 100, "final_loss": 0.5, "elapsed_seconds": 10, "epoch_losses": [0.8, 0.5]}}'
            with patch("builtins.open", mock_open(read_data=data)):
                await get_results_command(update, context)
                update.message.reply_text.assert_called_once()
                assert "Результаты `task_1` (test)" in update.message.reply_text.call_args[0][0]

@pytest.mark.asyncio
async def test_get_results_command_long_error(update, context):
    context.args = ["task_1"]
    tasks = {"task_1": {"results_file": "dummy.json"}}
    with patch("telegram_bot.running_tasks", tasks):
        with patch("os.path.exists", return_value=True):
            data = '{"task_1": {"status": "error", "error": "' + 'A' * 5000 + '"}}'
            with patch("builtins.open", mock_open(read_data=data)):
                await get_results_command(update, context)
                update.message.reply_document.assert_called_once()
