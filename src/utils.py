import pickle
import json
from src.logger import setup_logger
from src import config

utils_logger = setup_logger('utils', config.LOG_FILE_PATH, level=config.LOG_LEVEL)

def save_pickle(obj, path):
    """Сохраняет объект в файл pickle."""
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        utils_logger.info(f"Объект успешно сохранен в: {path}")
    except Exception as e:
        utils_logger.error(f"Ошибка при сохранении объекта в {path}: {e}")

def load_pickle(path):
    """Загружает объект из файла pickle."""
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        utils_logger.info(f"Объект успешно загружен из: {path}")
        return obj
    except FileNotFoundError:
        utils_logger.error(f"Файл не найден по пути: {path}")
        return None
    except Exception as e:
        utils_logger.error(f"Ошибка при загрузке объекта из {path}: {e}")
        return None

def save_json(data, filepath):
    """Сохраняет данные в JSON файл."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        utils_logger.info(f"Данные успешно сохранены в JSON файл: {filepath}")
    except Exception as e:
        utils_logger.error(f"Ошибка при сохранении данных в JSON файл {filepath}: {e}")

def load_json(filepath):
    """Загружает данные из JSON файла."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            utils_logger.info(f"Данные успешно загружены из JSON файла: {filepath}")
            return json.load(f)
    except FileNotFoundError:
        utils_logger.error(f"JSON файл не найден по пути: {filepath}")
        return None
    except Exception as e:
        utils_logger.error(f"Ошибка при загрузке данных из JSON файла {filepath}: {e}")
        return None