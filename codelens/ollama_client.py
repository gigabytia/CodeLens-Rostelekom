import asyncio

import httpx

from codelens.config import OLLAMA_URL
from codelens.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "Ты ассистент, помогающий разработчику разобраться в кодовой базе на Python.\n"
    "СТРОГИЕ ПРАВИЛА - нарушать нельзя:\n"
    "1. Отвечай на основе предоставленных фрагментов кода.\n"
    "2. Если спрашивают о функции или классе, которых НЕТ в фрагментах, "
    "скажи прямо: «Функция/класс [название] не найдена в кодовой базе.»\n"
    "3. Не придумывай, не угадывай, не используй общие знания о Python.\n"
    "4. Если вопрос не про код - скажи: «Я помогаю только с вопросами по кодовой базе.»\n"
    "5. Всегда отвечай на том же языке, на котором задан вопрос.\n"
    "6. Отвечай кратко и по делу; подробно - только если явно просят."
)


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=120, trust_env=False)

    async def is_available(self) -> bool:
        try:
            resp = await self._client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
    ) -> str:
        payload = {
            "model": model,
            "system": system or SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
        }
        try:
            resp = await self._client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json().get("response", "Нет ответа от модели.")
        except httpx.ConnectError:
            logger.warning("Нет соединения с Ollama (%s)", self.base_url)
            return "Ошибка подключения к Ollama. Проверьте, что сервер запущен."
        except httpx.TimeoutException:
            logger.warning("Таймаут при запросе к Ollama")
            return "Таймаут ответа Ollama. Попробуйте ещё раз."
        except Exception as exc:
            logger.exception("Неожиданная ошибка Ollama: %s", exc)
            return f"Ошибка Ollama: {exc}"

    async def close(self) -> None:
        await self._client.aclose()

    def _run(self, coro):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    def is_available_sync(self) -> bool:
        return self._run(self.is_available())

    def generate_sync(
        self,
        model: str,
        prompt: str,
        system: str = "",
    ) -> str:
        return self._run(self.generate(model, prompt, system))
