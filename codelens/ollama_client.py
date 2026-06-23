import asyncio

import httpx

from codelens.config import OLLAMA_URL
from codelens.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "Ты ассистент, помогающий разработчику разобраться в кодовой базе на Python.\n"
    "СТРОГИЕ ПРАВИЛА - нарушать нельзя:\n"
    "1. Отвечай на основе предоставленных фрагментов кода.\n"
    "2. Если спрашивают о функции или классе, которых НЕТ в фрагментах - скажи прямо: 'Функция/класс [название] не найдена в кодовой базе."
    "3. Не придумывай, не угадывай, не используй общие знания о Python.\n"
    "4. Если вопрос не про код - скажи: 'Я помогаю только с вопросами по кодовой базе.'\n"
    "5. Всегда отвечай на том же языке, на котором задан вопрос.\n"
    "6. Отвечай кратко и по делу, но если просят объяснить подробно рассказывай более развернуто."
)

class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL):
        self.base_url = base_url
        self._client = httpx.AsyncClient(timeout=120, trust_env=False)

    async def is_available(self) -> bool:
        try:
            resp = await self._client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    async def generate(
        self, model: str, prompt: str, system: str = ""
    ) -> str:
        payload = {
            "model": model,
            "system": system or SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
        }
        try:
            response = await self._client.post(
                f"{self.base_url}/api/generate", json=payload
            )
            response.raise_for_status()
            return response.json().get("response", "Нет ответа от модели.")
        except httpx.ConnectError:
            return "Ошибка подключения к Ollama. Проверьте, что сервер запущен."
        except httpx.TimeoutException:
            return "Таймаут ответа Ollama. Попробуйте ещё раз."
        except Exception as e:
            return f"Ошибка Ollama: {e}"

    async def close(self) -> None:
        await self._client.aclose()

    def generate_sync(self, model: str, prompt: str, system: str = "") -> str:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate(model, prompt, system))
        finally:
            loop.close()

    def is_available_sync(self) -> bool:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.is_available())
        finally:
            loop.close()