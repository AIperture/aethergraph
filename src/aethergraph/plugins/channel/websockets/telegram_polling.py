# telegram_polling.py
import asyncio
import json
from typing import Optional, Any, Dict, List
import aiohttp

from ..utils.telegram_utils import _http_session, _process_update 

class TelegramPollingRunner:
    def __init__(self, container, settings):
        self.container = container
        self.settings = settings
        self.bot_token: str = settings.telegram.bot_token.get_secret_value() or ""
        self._stop = False

    async def stop(self):
        self._stop = True

    async def _fetch_updates(self, offset: Optional[int]) -> List[Dict[str, Any]]:
        if not self.bot_token:
            self.container.logger.for_run().warning("[TelegramPolling] no bot token, skipping fetch")
            return []

        api = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        params: Dict[str, Any] = {"timeout": 30}
        if offset is not None:
            params["offset"] = offset

        try:
            sess = _http_session()
            async with sess.get(api, params=params) as r:
                status = r.status
                # Read raw text so we can log even if JSON parse fails
                raw = await r.text()

                if status != 200:
                    self.container.logger.for_run().warning(f"[TelegramPolling] non-200 status: {status}, returning empty list")
                    return []

                try:
                    data = json.loads(raw)
                except Exception as e:
                    self.container.logger.for_run().error(f"[TelegramPolling] JSON decode error: {e}")
                    return []

                if not data.get("ok"):
                    self.container.logger.for_run().warning(f"[TelegramPolling] ok=false in response: {data}")
                    return []

                result = data.get("result") or []
                if result:
                    first_id = result[0].get("update_id")
                    last_id = result[-1].get("update_id")
                    self.container.logger.for_run().info(f"[TelegramPolling] got {len(result)} updates, ids {first_id}..{last_id}")
                else:
                    self.container.logger.for_run().info("[TelegramPolling] got 0 updates in this poll")

                return result

        except asyncio.TimeoutError:
            self.container.logger.for_run().warning("[TelegramPolling] asyncio.TimeoutError while fetching updates")
            return []
        except aiohttp.ClientError as e:
            self.container.logger.for_run().error(f"[TelegramPolling] aiohttp.ClientError: {e}")
            return []
        except Exception as e:
            self.container.logger.for_run().error(f"[TelegramPolling] unexpected error in _fetch_updates: {e}")
            return []

        except asyncio.TimeoutError:
            self.container.logger.for_run().warning(f"{asyncio.TimeoutError} (client timed out waiting for response)")


        except aiohttp.ClientConnectionError as e:
            self.container.logger.for_run().warning(f"[TelegramPolling] ClientConnectionError: {e}")


        except Exception as e:
            self.container.logger.for_run().error(f"[TelegramPolling] unexpected error: {e}", exc_info=True)
            return []
        
    async def _fetch_updates_(self, offset: Optional[int]) -> List[Dict[str, Any]]:
        if not self.bot_token:
            return []
        api = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        params: Dict[str, Any] = {"timeout": 30}
        if offset is not None:
            params["offset"] = offset

        async with _http_session().get(api, params=params) as r:
            if r.status != 200:
                return []
            data = await r.json()
            if not data.get("ok"):
                return []
            return data.get("result") or []

    async def start(self):
        if not self.bot_token:
            self.container.logger.for_run().warning("[TelegramPolling] not started: missing bot token")
            return

        self.container.logger.for_run().info("[TelegramPolling] starting polling loop...")

        offset: Optional[int] = None

        # OPTIONAL: initial drain of old updates so we only react to new ones
        try:
            initial_updates = await self._fetch_updates(offset=None)
            if initial_updates:
                last_id = initial_updates[-1].get("update_id")
                if last_id is not None:
                    offset = last_id + 1
        except Exception as e:
            self.container.logger.for_run().error(f"[TelegramPolling] initial drain failed: {e}")

        while not self._stop:
            try:
                self.container.logger.for_run().info(f"[TelegramPolling] fetching updates with offset={offset}...")
                updates = await self._fetch_updates(offset)
                self.container.logger.for_run().info(f"[TelegramPolling] fetched {len(updates)} updates.")
                if updates:
                    # process each, then bump offset past the last one
                    for upd in updates:
                        await _process_update(self.container, upd, self.bot_token)

                    last_id = updates[-1].get("update_id")
                    if last_id is not None:
                        offset = last_id + 1
                # if no updates, loop back (Telegram held the connection up to timeout)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.container.logger.for_run().error(f"[TelegramPolling] error: {e}")
                await asyncio.sleep(5)

        self.container.logger.for_run().info("[TelegramPolling] stopped.")
