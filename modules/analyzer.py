"""LLM 内容分析器：摘要、要点、资源、行动建议（含重试）"""

import json
import time

from loguru import logger
from openai import OpenAI

INITIAL_DELAY = 5
MAX_DELAY = 120
BACKOFF_FACTOR = 2

_ANALYZE_PROMPT = """分析以下视频内容，生成知识笔记。

类别: {category}
标签: {tags}

转录文字:
{transcript}

画面文字:
{ocr_text}

输出 JSON:
{{
  "summary": "3-5句摘要",
  "key_points": ["要点1", "要点2"],
  "resources": ["工具/网站/书籍"],
  "action_items": {{
    "dev": ["开发相关建议"],
    "life": ["生活相关建议"],
    "tech_summary": ["技术总结建议"]
  }}
}}"""


def _llm_call_with_retry(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int = 8192,
    temperature: float = 0.3,
) -> str:
    delay = INITIAL_DELAY
    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=180.0,
            )
            msg = response.choices[0].message
            content = msg.content or msg.reasoning or ""
            if content.strip():
                return content
            logger.warning("LLM 分析返回空内容 (第{}次), {}s 后重试", attempt, delay)
        except Exception as e:
            logger.warning(
                "LLM 分析失败 (第{}次), {}s 后重试: {}", attempt, delay, str(e)[:120]
            )
        time.sleep(delay)
        delay = min(delay * BACKOFF_FACTOR, MAX_DELAY)
    return None


def _extract_json(text: str) -> dict | None:
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


class ContentAnalyzer:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=180.0)
        self.model = model

    def analyze(
        self,
        transcript: str,
        ocr_text: str,
        category: str,
        tags: list[str],
    ) -> dict:
        max_chars = 4000
        if len(transcript) > max_chars:
            transcript = transcript[:max_chars]

        prompt = _ANALYZE_PROMPT.format(
            category=category,
            tags=", ".join(tags),
            transcript=transcript or "（无转录文字）",
            ocr_text=ocr_text[:1000] if ocr_text else "（无画面文字）",
        )

        logger.info("调用 LLM 分析: category={}, model={}", category, self.model)

        content = _llm_call_with_retry(
            self.client,
            self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        result = _extract_json(content)
        if result is None:
            logger.error("分析 JSON 解析失败: {}", content[:200])
            return _empty_analysis(category, tags)

        action_items_raw = result.get("action_items", {}) or {}
        action_items: dict[str, list[str]] = {
            "dev": action_items_raw.get("dev", []),
            "life": action_items_raw.get("life", []),
            "tech_summary": action_items_raw.get("tech_summary", []),
        }
        analysis: dict = {
            "summary": result.get("summary", ""),
            "key_points": result.get("key_points", []),
            "resources": result.get("resources", []),
            "action_items": action_items,
            "category": category,
            "tags": tags,
        }

        logger.success(
            "分析完成: {} 个要点, {} 个资源, {} 条建议",
            len(analysis["key_points"]),
            len(analysis["resources"]),
            sum(len(v) for v in action_items.values()),
        )
        return analysis


def _empty_analysis(category: str, tags: list[str]) -> dict:
    return {
        "summary": "",
        "key_points": [],
        "resources": [],
        "action_items": {"dev": [], "life": [], "tech_summary": []},
        "category": category,
        "tags": tags,
    }
