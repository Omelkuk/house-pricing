import logging
import sys
import os

class Logger:
    def __init__(self, log_dir="results"):
        os.makedirs(log_dir, exist_ok=True)
        
        # Настройка формата логирования
        log_format = "%(asctime)s | %(levelname)s | %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        
        # Файловый обработчик (записывает историю в файл)
        file_handler = logging.FileHandler(os.path.join(log_dir, "run.log"), encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        
        # Консольный обработчик (выводит информацию на экран)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        
        self.logger = logging.getLogger("ML_Logger")
        self.logger.setLevel(logging.INFO)
        
        # Защита от дублирования логов
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(f"[INFO] {message}")

    def success(self, message):
        self.logger.info(f"[OK] {message}")

    def warning(self, message):
        self.logger.warning(f"[WARN] {message}")

    def error(self, message):
        self.logger.error(f"[ERROR] {message}")
        
    def section(self, title):
        self.logger.info(f"\n{'='*60}\n{title.upper()}\n{'='*60}")

    def metric(self, name, value):
        self.logger.info(f"  {name:<15} : {value:.5f}")