import subprocess
import time
import platform
from typing import Optional
import httpx
from codelens.config import OLLAMA_MODEL, OLLAMA_URL
from codelens.logger import get_logger

logger = get_logger(__name__)

class OllamaManager:
    def __init__(self, model_name: str = OLLAMA_MODEL, url: str = OLLAMA_URL):
        self.model_name = model_name
        self.url = url
        self._process: Optional[subprocess.Popen] = None

    def is_installed(self) -> bool:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def is_running(self) -> bool:
        try:
            resp = httpx.get(f"{self.url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def start(self) -> bool:
        if self.is_running():
            logger.info("Ollama уже запущена")
            return True

        if not self.is_installed():
            logger.error("Ollama не установлена")
            return False

        popen_kw: dict = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if platform.system() == "Windows":
            popen_kw["creationflags"] = 0x00000008

        try:
            self._process = subprocess.Popen(["ollama", "serve"], **popen_kw)
            time.sleep(3)
            if self.is_running():
                logger.info("Ollama запущена")
                return True
            logger.warning("Не удалось запустить Ollama")
            return False
        except Exception as e:
            logger.error("Не удалось запустить Ollama: %s", e)
            return False

    def ensure_model(self) -> bool:
        if not self.is_running():
            logger.warning("Ollama не запущена, модель не будет скачана")
            return False

        list_result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if self.model_name in list_result.stdout.lower():
            logger.info("%s уже скачана", self.model_name)
            return True

        logger.info("Скачивание %s", self.model_name)
        pull = subprocess.run(
            ["ollama", "pull", self.model_name],
            capture_output=True,
            text=True,
        )
        if pull.returncode == 0:
            logger.info("%s готова", self.model_name)
            return True
        logger.error("Не удалось скачать %s", self.model_name)
        return False
