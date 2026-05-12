import logging
from logging.handlers import BaseRotatingHandler
import os

class TruncatingFileHandler(BaseRotatingHandler):
    """
    Кастомный обработчик логов, который обрезает файл логов при
    превышении максимального размера, удаляя старые строки.
    """
    def __init__(self, filename, mode='a', maxBytes=0, encoding=None, delay=False):
        # maxBytes - максимальный размер файла в байтах
        # backupCount - не используется, так как мы не ротируем файлы, а обрезаем
        if maxBytes > 0:
            mode = 'a'
        super().__init__(filename, mode, encoding, delay)
        self.maxBytes = maxBytes

    def shouldRollover(self, record):
        """
        Определяет, нужно ли обрезать файл логов.
        """
        if self.maxBytes > 0:
            self.stream.seek(0, 2)  # Перемещаем указатель в конец файла
            if self.stream.tell() >= self.maxBytes:
                return True
        return False

    def doRollover(self):
        """
        Выполняет усечение файла.
        """
        self.stream.close()
        
        with open(self.baseFilename, 'r', encoding=self.encoding) as f:
            lines = f.readlines()
            
        # Удаляем старые строки, пока размер не станет меньше maxBytes
        total_size = sum(len(line.encode(self.encoding)) for line in lines)
        while total_size >= self.maxBytes and len(lines) > 0:
            total_size -= len(lines.pop(0).encode(self.encoding))
            
        with open(self.baseFilename, 'w', encoding=self.encoding) as f:
            f.writelines(lines)
            
        if not self.delay:
            self.stream = self._open()

def setup_logger(name, log_file, level=logging.INFO, max_bytes=5 * 1024 * 1024):
    """
    Настраивает логгер с кастомным обработчиком TruncatingFileHandler.

    Параметры:
    - name (str): Имя логгера.
    - log_file (str): Путь к файлу логов.
    - level (int): Уровень логирования.
    - max_bytes (int): Максимальный размер файла в байтах (по умолчанию 5MB).
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Создаем директорию, если она не существует
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    handler = TruncatingFileHandler(log_file, maxBytes=max_bytes, encoding='utf-8')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    logger.propagate = False 
    
    return logger