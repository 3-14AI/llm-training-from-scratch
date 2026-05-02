import logging
import os
import asyncio
import subprocess
import json
from pathlib import Path
from datetime import datetime

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Включить логирование
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Ваш Telegram User ID для авторизации
AUTHORIZED_USER_ID = 452526556  # Денис Пименов

# Словарь для хранения запущенных процессов (эксперименты и подготовка данных)
# task_id: {process: subprocess.Popen, chat_id: int, status: str, log_file: str, start_time: datetime, type: str}
running_tasks = {}
task_counter = 0

# Путь к корневой директории репозитория
REPO_ROOT = Path(__file__).parent

# Загрузка конфигов экспериментов для валидации
try:
    from experiment_configs import ALL_EXPERIMENTS
except ImportError:
    logger.error("Не удалось загрузить experiment_configs.py. Убедитесь, что файл существует.")
    ALL_EXPERIMENTS = {}


# Функция-декоратор для проверки авторизации
def restricted(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id != AUTHORIZED_USER_ID:
            logger.warning(f"Unauthorized access attempt by user {user_id}")
            await update.message.reply_text("У вас нет доступа к этому боту.")
            return
        return await func(update, context)
    return wrapper


@restricted
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветственное сообщение при команде /start."""
    user = update.effective_user
    await update.message.reply_html(
        f"Привет, {user.mention_html()}! Я бот для управления экспериментами LLM. "
        "Используйте /help для списка команд."
    )


@restricted
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет сообщение с помощью при команде /help."""
    await update.message.reply_text(
        "Доступные команды:\n"
        "/start - Начать взаимодействие\n"
        "/help - Показать это сообщение\n"
        "/prepare_dataset [ru] [en] [zh] - Подготовить датасет (аргументы: кол-во сэмплов для каждого языка)\n"
        "/list_experiments - Показать список доступных экспериментов\n"
        "/run_experiment <exp_name> [mode] [epochs] - Запустить эксперимент (mode: e2e/full, epochs: число)\n"
        "/status - Получить статус запущенных задач\n"
        "/get_log <task_id> - Получить лог задачи по ID\n"
        "/get_results <task_id> - Получить финальные результаты эксперимента по ID"
    )


@restricted
async def prepare_dataset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Запускает подготовку датасета."""
    global task_counter
    
    ru = context.args[0] if len(context.args) > 0 else "700"
    en = context.args[1] if len(context.args) > 1 else "200"
    zh = context.args[2] if len(context.args) > 2 else "100"
    
    task_counter += 1
    task_id = f"prep_{task_counter}"
    log_file_path = REPO_ROOT / f"logs/{task_id}.log"
    
    os.makedirs(REPO_ROOT / "logs", exist_ok=True)
    
    command = [
        str(REPO_ROOT / "venv/bin/python3"), str(REPO_ROOT / "prepare_dataset.py"),
        "--ru", ru,
        "--en", en,
        "--zh", zh,
        "--output", str(REPO_ROOT / "multilingual_corpus.txt")
    ]
    
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=REPO_ROOT
        )
        running_tasks[task_id] = {
            "process": process,
            "chat_id": update.effective_chat.id,
            "status": "running",
            "type": "preparation",
            "name": "Dataset Preparation",
            "log_file": str(log_file_path),
            "start_time": datetime.now(),
        }
        await update.message.reply_text(
            f"Запущена подготовка датасета (ID: `{task_id}`).\n"
            f"Параметры: RU={ru}, EN={en}, ZH={zh}\n"
            f"Лог: `{log_file_path}`"
        )
        asyncio.create_task(monitor_task_log(task_id))
        
    except Exception as e:
        logger.error(f"Ошибка при запуске prepare_dataset: {e}")
        await update.message.reply_text(f"Ошибка при запуске подготовки датасета: {e}")


@restricted
async def list_experiments_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показывает список доступных экспериментов."""
    if not ALL_EXPERIMENTS:
        await update.message.reply_text("Конфигурации экспериментов не загружены.")
        return

    message_text = "Доступные эксперименты:\n"
    for exp_name, cfg in ALL_EXPERIMENTS.items():
        message_text += f"- `{exp_name}` (Series {cfg.get('series', '?')}, {'Compressed' if cfg['use_compression'] else 'Baseline'})\n"
    await update.message.reply_text(message_text)


@restricted
async def run_experiment_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Запускает эксперимент в отдельном процессе."""
    global task_counter

    if not context.args:
        await update.message.reply_text("Пожалуйста, укажите имя эксперимента. Например: /run_experiment s1_small_baseline e2e")
        return

    exp_name = context.args[0]
    mode = context.args[1] if len(context.args) > 1 else "e2e"
    epochs = context.args[2] if len(context.args) > 2 else None

    if exp_name not in ALL_EXPERIMENTS:
        await update.message.reply_text(f"Эксперимент `{exp_name}` не найден в конфигурациях.")
        return

    task_counter += 1
    task_id = f"exp_{task_counter}"
    log_file_path = REPO_ROOT / f"logs/{task_id}.log"
    results_file_path = REPO_ROOT / f"results/{task_id}_results.json"

    os.makedirs(REPO_ROOT / "logs", exist_ok=True)
    os.makedirs(REPO_ROOT / "results", exist_ok=True)

    command = [
        str(REPO_ROOT / "venv/bin/python3"), str(REPO_ROOT / "experiment_runner.py"),
        "--exp_name", exp_name,
        "--mode", mode,
        "--data", str(REPO_ROOT / "multilingual_corpus.txt"),
        "--vocab", str(REPO_ROOT / "multilingual_vocab.pt"),
        "--output", str(results_file_path),
    ]
    if epochs:
        command.extend(["--epochs", epochs])

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=REPO_ROOT
        )
        running_tasks[task_id] = {
            "process": process,
            "chat_id": update.effective_chat.id,
            "status": "running",
            "type": "experiment",
            "name": exp_name,
            "mode": mode,
            "log_file": str(log_file_path),
            "results_file": str(results_file_path),
            "start_time": datetime.now(),
        }
        await update.message.reply_text(
            f"Эксперимент `{exp_name}` (ID: `{task_id}`) запущен.\n" 
            f"Режим: `{mode}`" + (f", Эпох: `{epochs}`" if epochs else "") + "\n"
            f"Лог: `{log_file_path}`\n" 
            f"Результаты: `{results_file_path}`"
        )
        asyncio.create_task(monitor_task_log(task_id))

    except Exception as e:
        logger.error(f"Ошибка при запуске эксперимента {exp_name}: {e}")
        await update.message.reply_text(f"Ошибка при запуске эксперимента `{exp_name}`: {e}")


@restricted
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Получает статус запущенных задач."""
    if not running_tasks:
        await update.message.reply_text("Нет активных задач.")
        return

    message_text = "Статус активных задач:\n"
    for task_id, info in running_tasks.items():
        elapsed = datetime.now() - info["start_time"]
        message_text += (
            f"ID: `{task_id}`, Тип: `{info['type']}`, Имя: `{info['name']}`, "
            f"Статус: `{info['status']}`, Время: {str(elapsed).split('.')[0]}\n"
        )
    await update.message.reply_text(message_text)


@restricted
async def get_log_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет лог задачи."""
    if not context.args:
        await update.message.reply_text("Пожалуйста, укажите ID задачи. Например: /get_log exp_1")
        return
    task_id = context.args[0]

    if task_id not in running_tasks:
        await update.message.reply_text(f"Задача с ID `{task_id}` не найдена.")
        return

    log_file = running_tasks[task_id]["log_file"]
    if not os.path.exists(log_file):
        await update.message.reply_text(f"Лог-файл для `{task_id}` не найден.")
        return

    with open(log_file, "r", encoding="utf-8") as f:
        log_content = f.read()

    if len(log_content) > 4000:
        await update.message.reply_document(log_file, caption=f"Лог задачи `{task_id}` (обрезан)")
    else:
        # Экранирование для MarkdownV2 может быть сложным, используем простой текст если не уверены
        await update.message.reply_text(f"Лог `{task_id}`:\n```\n{log_content[-3500:]}\n```")


@restricted
async def get_results_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет финальные результаты эксперимента."""
    if not context.args:
        await update.message.reply_text("Пожалуйста, укажите ID эксперимента. Например: /get_results exp_1")
        return
    task_id = context.args[0]

    if task_id not in running_tasks or "results_file" not in running_tasks[task_id]:
        await update.message.reply_text(f"Результаты для ID `{task_id}` не найдены.")
        return

    results_file = running_tasks[task_id]["results_file"]
    if not os.path.exists(results_file):
        await update.message.reply_text(f"Файл результатов для `{task_id}` не найден или задача ещё не завершена.")
        return

    with open(results_file, "r", encoding="utf-8") as f:
        results_data = json.load(f)
    
    # Формируем красивый вывод результатов
    if task_id in results_data:
        res = results_data[task_id]
    else:
        # Если в файле словарь с ключами-именами экспериментов
        res = next(iter(results_data.values())) if results_data else None

    if res and res.get("status") == "ok":
        epoch_losses = res.get("epoch_losses", [])
        losses_str = "\n".join([f"Epoch {i+1}: {loss:.4f}" for i, loss in enumerate(epoch_losses)])
        message = (
            f"📊 Результаты `{task_id}` ({res['exp_name']}):\n"
            f"Статус: `OK`\n"
            f"Параметры: {res['n_params']:,}\n"
            f"Финальный Loss: {res['final_loss']:.4f}\n"
            f"Время: {res['elapsed_seconds']} сек\n\n"
            f"📈 Loss по эпохам:\n{losses_str}"
        )
        await update.message.reply_text(message)
    else:
        results_content = json.dumps(results_data, indent=2, ensure_ascii=False)
        if len(results_content) > 4000:
            await update.message.reply_document(results_file, caption=f"Результаты эксперимента `{task_id}`")
        else:
            await update.message.reply_text(f"Результаты `{task_id}`:\n```json\n{results_content}\n```")


async def monitor_task_log(task_id: str) -> None:
    """Фоновая задача для мониторинга лога и отправки обновлений."""
    info = running_tasks[task_id]
    process = info["process"]
    chat_id = info["chat_id"]
    log_file = info["log_file"]
    task_name = info["name"]

    with open(log_file, "w", encoding="utf-8") as f_log:
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            decoded_line = line.decode().strip()
            f_log.write(decoded_line + "\n")
            f_log.flush()
            
            # Отправляем важные обновления в Telegram
            if any(x in decoded_line for x in ["Loss:", "Saved", "Corpus saved", "Fetching"]):
                try:
                    # Используем новый инстанс бота для отправки сообщения
                    bot = Application.builder().token(os.environ.get("TELEGRAM_BOT_TOKEN")).build().bot
                    await bot.send_message(chat_id=chat_id, text=f"[`{task_id}`] {decoded_line}")
                except Exception as e:
                    logger.error(f"Ошибка при отправке лога: {e}")

    returncode = await process.wait()
    if returncode == 0:
        info["status"] = "completed"
        message = f"Задача `{task_name}` (ID: `{task_id}`) успешно завершена!"
    else:
        info["status"] = "error"
        message = f"Задача `{task_name}` (ID: `{task_id}`) завершилась с ошибкой (код: {returncode})."
    
    try:
        bot = Application.builder().token(os.environ.get("TELEGRAM_BOT_TOKEN")).build().bot
        await bot.send_message(chat_id=chat_id, text=message)
    except Exception as e:
        logger.error(f"Ошибка при отправке финального сообщения: {e}")


def main() -> None:
    """Запускает бота."""
    BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

    if not BOT_TOKEN:
        logger.error("Telegram Bot Token не найден.")
        return

    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("prepare_dataset", prepare_dataset_command))
    application.add_handler(CommandHandler("list_experiments", list_experiments_command))
    application.add_handler(CommandHandler("run_experiment", run_experiment_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("get_log", get_log_command))
    application.add_handler(CommandHandler("get_results", get_results_command))

    logger.info("Бот запущен. Ожидание сообщений...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
