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

# Словарь для хранения запущенных экспериментов
# experiment_id: {process: subprocess.Popen, chat_id: int, status: str, log_file: str, start_time: datetime}
running_experiments = {}
experiment_counter = 0

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
        "/list_experiments - Показать список доступных экспериментов\n"
        "/run_experiment <exp_name> [mode] - Запустить эксперимент (mode: e2e или full, по умолчанию e2e)\n"
        "/status - Получить статус запущенных экспериментов\n"
        "/get_log <exp_id> - Получить лог эксперимента по ID\n"
        "/get_results <exp_id> - Получить финальные результаты эксперимента по ID"
    )


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
    global experiment_counter

    if not context.args:
        await update.message.reply_text("Пожалуйста, укажите имя эксперимента. Например: /run_experiment s1_small_baseline e2e")
        return

    exp_name = context.args[0]
    mode = context.args[1] if len(context.args) > 1 else "e2e"

    if exp_name not in ALL_EXPERIMENTS:
        await update.message.reply_text(f"Эксперимент `{exp_name}` не найден в конфигурациях.")
        return

    experiment_counter += 1
    exp_id = f"exp_{experiment_counter}"
    log_file_path = REPO_ROOT / f"logs/{exp_id}.log"
    results_file_path = REPO_ROOT / f"results/{exp_id}_results.json"

    os.makedirs(REPO_ROOT / "logs", exist_ok=True)
    os.makedirs(REPO_ROOT / "results", exist_ok=True)

    command = [
        "python3", str(REPO_ROOT / "experiment_runner.py"),
        "--exp_name", exp_name,
        "--mode", mode,
        "--data", str(REPO_ROOT / "multilingual_corpus.txt"),
        "--vocab", str(REPO_ROOT / "multilingual_vocab.pt"),
        "--output", str(results_file_path),
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=REPO_ROOT
        )
        running_experiments[exp_id] = {
            "process": process,
            "chat_id": update.effective_chat.id,
            "status": "running",
            "exp_name": exp_name,
            "mode": mode,
            "log_file": str(log_file_path),
            "results_file": str(results_file_path),
            "start_time": datetime.now(),
            "last_log_line": 0,
        }
        await update.message.reply_text(
            f"Эксперимент `{exp_name}` (ID: `{exp_id}`) запущен в режиме `{mode}`.\n" 
            f"Лог: `{log_file_path}`\n" 
            f"Результаты: `{results_file_path}`"
        )
        # Запускаем фоновую задачу для мониторинга логов
        asyncio.create_task(monitor_experiment_log(exp_id))

    except Exception as e:
        logger.error(f"Ошибка при запуске эксперимента {exp_name}: {e}")
        await update.message.reply_text(f"Ошибка при запуске эксперимента `{exp_name}`: {e}")


@restricted
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Получает статус запущенных экспериментов."""
    if not running_experiments:
        await update.message.reply_text("Нет запущенных экспериментов.")
        return

    message_text = "Статус запущенных экспериментов:\n"
    for exp_id, info in running_experiments.items():
        elapsed = datetime.now() - info["start_time"]
        message_text += (
            f"ID: `{exp_id}`, Имя: `{info['exp_name']}`, "
            f"Режим: `{info['mode']}`, Статус: `{info['status']}`, "
            f"Время: {str(elapsed).split('.')[0]}\n"
        )
    await update.message.reply_text(message_text)


@restricted
async def get_log_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет лог эксперимента."""
    if not context.args:
        await update.message.reply_text("Пожалуйста, укажите ID эксперимента. Например: /get_log exp_1")
        return
    exp_id = context.args[0]

    if exp_id not in running_experiments:
        await update.message.reply_text(f"Эксперимент с ID `{exp_id}` не найден.")
        return

    log_file = running_experiments[exp_id]["log_file"]
    if not os.path.exists(log_file):
        await update.message.reply_text(f"Лог-файл для `{exp_id}` не найден.")
        return

    with open(log_file, "r", encoding="utf-8") as f:
        log_content = f.read()

    if len(log_content) > 4000: # Telegram API limit
        await update.message.reply_document(log_file, caption=f"Лог эксперимента `{exp_id}` (обрезан)")
    else:
        await update.message.reply_text(f"```\n{log_content}\n```", parse_mode="MarkdownV2")


@restricted
async def get_results_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет финальные результаты эксперимента."""
    if not context.args:
        await update.message.reply_text("Пожалуйста, укажите ID эксперимента. Например: /get_results exp_1")
        return
    exp_id = context.args[0]

    if exp_id not in running_experiments:
        await update.message.reply_text(f"Эксперимент с ID `{exp_id}` не найден.")
        return

    results_file = running_experiments[exp_id]["results_file"]
    if not os.path.exists(results_file):
        await update.message.reply_text(f"Файл результатов для `{exp_id}` не найден или эксперимент ещё не завершен.")
        return

    with open(results_file, "r", encoding="utf-8") as f:
        results_content = f.read()
    
    if len(results_content) > 4000:
        await update.message.reply_document(results_file, caption=f"Результаты эксперимента `{exp_id}`")
    else:
        await update.message.reply_text(f"```json\n{results_content}\n```", parse_mode="MarkdownV2")


async def monitor_experiment_log(exp_id: str) -> None:
    """Фоновая задача для мониторинга лога эксперимента и отправки обновлений."""
    info = running_experiments[exp_id]
    process = info["process"]
    chat_id = info["chat_id"]
    log_file = info["log_file"]
    results_file = info["results_file"]
    exp_name = info["exp_name"]

    # Открываем лог-файл для записи вывода процесса
    with open(log_file, "w", encoding="utf-8") as f_log:
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            decoded_line = line.decode().strip()
            f_log.write(decoded_line + "\n")
            f_log.flush()
            # Отправляем последние 2-3 строки лога в Telegram для прогресса
            # Ограничиваем частоту отправки, чтобы не спамить
            if "Loss:" in decoded_line or "finished" in decoded_line:
                try:
                    await Application.builder().token(os.environ.get("TELEGRAM_BOT_TOKEN", "8626937727:AAEtndzq6a3osg0gM5JpPqBaGtdJpaD31Xk")).build().bot.send_message(
                        chat_id=chat_id, text=f"[`{exp_id}`] {decoded_line}", parse_mode="MarkdownV2"
                    )
                except Exception as e:
                    logger.error(f"Ошибка при отправке лога в Telegram: {e}")

    # Процесс завершился
    returncode = await process.wait()
    if returncode == 0:
        info["status"] = "completed"
        message = f"Эксперимент `{exp_name}` (ID: `{exp_id}`) успешно завершен!\n" \
                  f"Результаты доступны по команде /get_results `{exp_id}`"
    else:
        info["status"] = "error"
        message = f"Эксперимент `{exp_name}` (ID: `{exp_id}`) завершился с ошибкой (код: {returncode}).\n" \
                  f"Подробности в логе: /get_log `{exp_id}`"
    
    try:
        await Application.builder().token(os.environ.get("TELEGRAM_BOT_TOKEN", "8626937727:AAEtndzq6a3osg0gM5JpPqBaGtdJpaD31Xk")).build().bot.send_message(
            chat_id=chat_id, text=message, parse_mode="MarkdownV2"
        )
    except Exception as e:
        logger.error(f"Ошибка при отправке финального сообщения в Telegram: {e}")


def main() -> None:
    """Запускает бота."""
    BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8626937727:AAEtndzq6a3osg0gM5JpPqBaGtdJpaD31Xk")

    if not BOT_TOKEN:
        logger.error("Telegram Bot Token не найден. Установите переменную окружения TELEGRAM_BOT_TOKEN.")
        return

    application = Application.builder().token(BOT_TOKEN).build()

    # Регистрируем обработчики команд
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("list_experiments", list_experiments_command))
    application.add_handler(CommandHandler("run_experiment", run_experiment_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("get_log", get_log_command))
    application.add_handler(CommandHandler("get_results", get_results_command))

    # Запускаем бота
    logger.info("Бот запущен. Ожидание сообщений...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
