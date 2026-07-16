"""AI 辅助分析模块。

设计原则(务必遵守):
- AI 不替代量化引擎,只做"翻译官+体检医生":把我们本地算好的结构化指标
  翻译成人话点评、提示消息面风险。
- 绝不让 AI 直接给"买/卖"结论或目标价 —— 避免幻觉与合规风险。
- 所有 AI 输出统一挂"仅供参考,不构成投资建议"。
- 模型走 OpenAI 兼容接口,base_url/key/model 由本地 config.json 配置,
  可自由切换 DeepSeek / 通义 / 智谱 / OpenAI 等,key 绝不进 git。
"""

from .client import chat, chat_stream, is_configured, config_hint  # noqa: F401
from .commentary import comment_stock, build_facts          # noqa: F401
