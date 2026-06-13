"""
tools/evidence.py — Tavily-powered evidence retrieval.
Called once before any agent sees the task.
Always grabs live web context so the debate is grounded in current facts.
"""
import asyncio
from tavily import TavilyClient
from core.models import EvidencePacket
from core.logger import get_logger

log = get_logger(__name__)


class EvidenceTool:
    def __init__(self, api_key: str, max_results: int = 5):
        self._client = TavilyClient(api_key=api_key)
        self._max_results = max_results

    async def fetch(self, query: str) -> EvidencePacket:
        """
        Run a Tavily search and return an EvidencePacket.
        Runs in a thread executor because the Tavily SDK is synchronous.
        """
        log.info(f"[bold]Evidence[/bold] → searching: {query[:80]}…")
        loop = asyncio.get_event_loop()

        try:
            result = await loop.run_in_executor(
                None,
                lambda: self._client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=self._max_results,
                    include_answer=True,
                ),
            )
        except Exception as exc:
            log.warning(f"Tavily search failed ({exc}); continuing without live evidence.")
            return EvidencePacket(query=query, snippets=[], urls=[])

        snippets: list[str] = []
        urls: list[str] = []
        for item in result.get("results", []):
            content = item.get("content", "").strip()
            url = item.get("url", "")
            if content:
                snippets.append(content)
            if url:
                urls.append(url)

        packet = EvidencePacket(
            query=query,
            snippets=snippets[:self._max_results],
            urls=urls[:self._max_results],
            raw_answer=result.get("answer"),
        )
        log.info(f"Evidence gathered: {len(snippets)} snippets, {len(urls)} sources.")
        return packet

    @staticmethod
    def format_for_prompt(packet: EvidencePacket) -> str:
        """Serialize evidence into a compact block suitable for injection into any prompt."""
        if not packet.snippets and not packet.raw_answer:
            return "[No live evidence retrieved — answer from model knowledge only.]"

        lines = ["=== Live evidence (Tavily) ==="]
        if packet.raw_answer:
            lines.append(f"Direct answer: {packet.raw_answer}")
        for i, (snippet, url) in enumerate(zip(packet.snippets, packet.urls), 1):
            lines.append(f"[{i}] {url}\n{snippet[:400]}")
        return "\n\n".join(lines)
