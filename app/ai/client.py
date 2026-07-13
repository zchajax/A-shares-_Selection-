"""OpenAI 兼容的极简模型客户端(零第三方依赖,仅用标准库)。

配置文件: 项目根目录 config.json (已在 .gitignore 中,不入库)。
格式:
{
  "ai": {
    "base_url": "https://api.deepseek.com/v1",
    "api_key":  "sk-xxxxxxxx",
    "model":    "deepseek-chat",
    "timeout":  40
  }
}

兼容各家(只要提供 OpenAI 风格的 /chat/completions):
- DeepSeek : base_url = https://api.deepseek.com/v1        model = deepseek-chat
- 通义千问  : base_url = https://dashscope.aliyuncs.com/compatible-mode/v1  model = qwen-plus
- 智谱GLM   : base_url = https://open.bigmodel.cn/api/paas/v4  model = glm-4-flash
- OpenAI    : base_url = https://api.openai.com/v1          model = gpt-4o-mini
"""

import json
import os
import urllib.request
import urllib.error

# 项目根目录(app/ai/client.py 往上三级)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONFIG_PATH = os.path.join(_ROOT, "config.json")


def _load_config() -> dict:
    """读取 config.json 里的 ai 段;不存在或损坏时返回空 dict。"""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("ai", {}) or {}
    except Exception:  # noqa
        return {}


def is_configured() -> bool:
    """是否已正确填写 base_url / api_key / model。"""
    c = _load_config()
    return bool(c.get("base_url") and c.get("api_key") and c.get("model"))


def config_hint() -> str:
    """未配置时给用户的引导文案。"""
    return (
        "尚未配置 AI 模型。请在项目根目录新建 config.json:\n"
        '{\n'
        '  "ai": {\n'
        '    "base_url": "https://api.deepseek.com/v1",\n'
        '    "api_key": "你的key",\n'
        '    "model": "deepseek-chat"\n'
        '  }\n'
        '}\n'
        "支持 DeepSeek / 通义 / 智谱 / OpenAI 等 OpenAI 兼容接口。\n"
        "(config.json 已在 .gitignore 中,不会上传)"
    )


class AIError(Exception):
    """AI 调用相关异常(配置缺失、网络、接口错误统一抛出)。"""


def chat(messages, temperature=0.4, max_tokens=600) -> str:
    """
    调用 OpenAI 兼容的 /chat/completions,返回第一条回复的文本内容。

    messages: [{"role": "system"/"user"/"assistant", "content": "..."}]
    仅用标准库 urllib,无需安装 openai SDK。异常统一抛 AIError。
    """
    c = _load_config()
    if not (c.get("base_url") and c.get("api_key") and c.get("model")):
        raise AIError(config_hint())

    base_url = c["base_url"].rstrip("/")
    url = base_url + "/chat/completions"
    timeout = float(c.get("timeout", 40))

    payload = {
        "model": c["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {c['api_key']}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:  # noqa
        detail = ""
        try:
            detail = e.read().decode("utf-8")[:300]
        except Exception:  # noqa
            pass
        raise AIError(f"接口返回错误 HTTP {e.code}: {detail or e.reason}")
    except urllib.error.URLError as e:  # noqa
        raise AIError(f"网络连接失败: {getattr(e, 'reason', e)}")
    except Exception as e:  # noqa
        raise AIError(f"调用异常: {e}")

    try:
        obj = json.loads(body)
        return obj["choices"][0]["message"]["content"].strip()
    except Exception as e:  # noqa
        raise AIError(f"解析响应失败: {e}; 原文前200字: {body[:200]}")
