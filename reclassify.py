#!/usr/bin/env python3
"""用 LLM 重新分类所有 261 篇笔记"""

import json
import os
import re
import time
from pathlib import Path

import yaml
from loguru import logger
from openai import OpenAI

NOTES_DIR = Path("output/notes")

LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")

INITIAL_DELAY = 5
MAX_DELAY = 120
BACKOFF_FACTOR = 2

NEW_CATEGORIES = [
    "RAG",
    "AGENT",
    "AI/ML",
    "前端",
    "后端",
    "DevOps",
    "开发",
    "创富",
    "文史哲",
    "感悟",
    "运动",
    "生活技巧",
    "其他",
]

CLASSIFY_PROMPT = """你是一个内容分类专家。请将以下视频笔记归入最合适的类别，并提取 2-5 个标签。

## 预设类别（含分类标准）

1. **RAG** — 检索增强生成、向量检索、知识库问答、Query改写、Embedding、文档解析(OCR/PDF)
2. **AGENT** — AI智能体开发、Agent框架、Skills、MCP、多智能体编排、Claude Code使用、Computer Use
3. **AI/ML** — 大模型通用(GPT/Claude/GLM等)、模型训练/微调、AI产品经理、AI行业趋势、AI工具推荐(非开发类)、AI学习方法论、TTS/语音识别/OCR模型
4. **前端** — 前端框架、UI/UX设计、CSS、React/Vue/TypeScript、页面交互、前端工程化
5. **后端** — 后端架构、数据库、API设计、服务器部署、Token认证、系统设计
6. **DevOps** — 运维、HTTPS/SSL、CI/CD、Docker/K8s、云服务、域名、监控、静态站点部署
7. **开发** — 通用开发工具(GitHub项目、开源工具)、编程效率、建站、VitePress/Hexo等建站工具、PDF处理工具、非AI专用的技术工具、文件格式转换
8. **创富** — 创业、赚钱、投资理财、金融知识、股票交易、商业案例、电商运营、流量运营、商业模式、超级个体、副业、IP打造
9. **文史哲** — 历史故事、文学作品、诗词、哲学思想(王阳明/国学/道/儒)、人文社科书籍、文化古迹
10. **感悟** — 纯个人成长感悟、人生哲理、心理疗愈、认知觉醒、自我提升(非技术非商业)
11. **运动** — 户外运动、徒步、骑行、自驾游攻略、旅行路线
12. **生活技巧** — 健康养生、职场技巧、失业金/社保、税收、科普知识、软件使用技巧、自媒体创作
13. **其他** — 无法归入以上任何类别的

## 分类关键区分规则
- 「开发」vs「DevOps」：开发=开发者工具/GitHub项目/效率工具；DevOps=部署/运维/HTTPS/服务器/云服务
- 「开发」vs「前端/后端」：明确涉及前端UI/框架→前端；明确涉及后端架构→后端；通用工具/建站→开发
- 「开发」vs「AI/ML」：与AI大模型强相关→AI/ML；通用技术工具(PDF解析/OCR引擎)→开发
- 「创富」vs「感悟」：涉及赚钱/商业/创业/投资→创富；纯心灵感悟/人生哲学→感悟
- 「文史哲」vs「感悟」：涉及具体历史/文学/哲学体系(王阳明/国学/诗词)→文史哲；个人情感/认知觉醒→感悟
- 「运动」vs「生活技巧」：户外活动/徒步/自驾路线→运动；健康知识/科普/社保→生活技巧

## 视频信息
标题: {title}
作者: {author}
原分类: {old_category}
原标签: {old_tags}

摘要: {summary}

只输出 JSON: {{"category": "类别", "tags": ["标签1", "标签2"]}}"""


def llm_call(prompt: str, max_tokens: int = 4096) -> str:
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=300.0)
    delay = INITIAL_DELAY
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            msg = resp.choices[0].message
            content = msg.content or msg.reasoning or ""
            if content.strip():
                return content
            logger.warning("LLM 返回空 (第{}次)", attempt)
        except Exception as e:
            logger.warning(
                "LLM 失败 (第{}次), {}s 后重试: {}", attempt, delay, str(e)[:120]
            )
        time.sleep(delay)
        delay = min(delay * BACKOFF_FACTOR, MAX_DELAY)


def extract_json(text: str) -> dict | None:
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def main():
    notes = []
    for f in sorted(NOTES_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        parts = text.split("---", 2)
        if len(parts) < 3:
            continue
        try:
            fm = yaml.safe_load(parts[1]) or {}
        except yaml.YAMLError:
            continue

        title = fm.get("title", f.stem)
        author = fm.get("author", "")
        old_cat = fm.get("category", "其他")
        old_tags = fm.get("tags") or []

        m = re.search(
            r"> \[!summary\] 摘要\n> (.+?)(?=\n##|\n#|\Z)", parts[2], re.DOTALL
        )
        summary = m.group(1).strip()[:200] if m else ""

        notes.append(
            {
                "file": f,
                "title": title,
                "author": author,
                "old_cat": old_cat,
                "old_tags": old_tags,
                "summary": summary,
                "frontmatter": fm,
                "content": parts[2],
            }
        )

    logger.info("共 {} 篇笔记需要重新分类", len(notes))

    changed = 0
    for i, note in enumerate(notes):
        prompt = CLASSIFY_PROMPT.format(
            title=note["title"][:80],
            author=note["author"],
            old_category=note["old_cat"],
            old_tags=", ".join(note["old_tags"]),
            summary=note["summary"][:300] if note["summary"] else "（无摘要）",
        )

        result_text = llm_call(prompt)
        result = extract_json(result_text)

        if result is None:
            logger.error(
                "[{}/{}] JSON 解析失败: {}", i + 1, len(notes), note["title"][:40]
            )
            continue

        new_cat = result.get("category", "其他")
        new_tags = result.get("tags", note["old_tags"])

        if new_cat not in NEW_CATEGORIES:
            logger.warning(
                "未知类别 '{}'，回退到 其他: {}", new_cat, note["title"][:40]
            )
            new_cat = "其他"

        if new_cat != note["old_cat"]:
            changed += 1
            logger.info(
                "[{}/{}] {} → {} | {}",
                i + 1,
                len(notes),
                note["old_cat"],
                new_cat,
                note["title"][:50],
            )

        # 写回 frontmatter
        note["frontmatter"]["category"] = new_cat
        note["frontmatter"]["tags"] = new_tags

        new_fm = yaml.dump(
            note["frontmatter"], allow_unicode=True, default_flow_style=False
        ).strip()
        new_content = f"---\n{new_fm}\n---{note['content']}"
        note["file"].write_text(new_content, encoding="utf-8")

    logger.success("重新分类完成: {} 篇, {} 篇变更", len(notes), changed)

    # 统计新分类
    from collections import Counter

    cats = []
    for f in sorted(NOTES_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                fm = yaml.safe_load(parts[1]) or {}
                cats.append(fm.get("category", "其他"))
            except:
                cats.append("其他")

    for cat, count in Counter(cats).most_common():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
