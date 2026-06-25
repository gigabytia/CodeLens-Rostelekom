import platform
import subprocess
import time
from typing import Optional

import httpx

from codelens.config import OLLAMA_MODEL, OLLAMA_URL
from codelens.logger import get_logger

logger = get_logger(__name__)


class OllamaManager:
    def __init__(
        self,
        model_name: str = OLLAMA_MODEL,
        url: str = OLLAMA_URL,
    ) -> None:
        self.model_name = model_name
        self.url = url.rstrip("/")
        self._process: Optional[subprocess.Popen] = None

    def is_installed(self) -> bool:
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.debug("Ollama CLI: %s", result.stdout.strip())
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

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
            logger.error(
                "Ollama не установлена. "
                "Инструкция: https://ollama.com/download"
            )
            return False

        popen_kw: dict = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if platform.system() == "Windows":
            popen_kw["creationflags"] = subprocess.CREATE_NO_WINDOW

        try:
            self._process = subprocess.Popen(["ollama", "serve"], **popen_kw)
            for _ in range(10):
                time.sleep(1)
                if self.is_running():
                    logger.info("Ollama сервер запущен (PID=%d)", self._process.pid)
                    return True
            logger.warning("Ollama запущена, но не отвечает за 10 сек")
            return False
        except Exception as exc:
            logger.error("Не удалось запустить Ollama: %s", exc)
            return False

    def model_is_available(self) -> bool:
        if not self.is_running():
            return False
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return self.model_name.lower() in result.stdout.lower()
        except Exception:
            return False

    def ensure_model(self) -> bool:
        if not self.is_running():
            logger.warning("Ollama не запущена - модель не будет скачана")
            return False

        if self.model_is_available():
            logger.info("Модель %s уже доступна", self.model_name)
            return True

        logger.info("Скачивание модели %s ...", self.model_name)
        try:
            result = subprocess.run(
                ["ollama", "pull", self.model_name],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info("Модель %s успешно скачана", self.model_name)
                return True
            logger.error(
                "Не удалось скачать %s: %s",
                self.model_name,
                result.stderr.strip(),
            )
            return False
        except Exception as exc:
            logger.error("Ошибка при скачивании модели: %s", exc)
            return False

    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            logger.info("Ollama процесс остановлен")
