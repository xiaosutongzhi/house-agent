import os

from langchain.chat_models import init_chat_model


def _clean_env_value(value: str | None) -> str:
	if not value:
		return ""
	return value.strip().strip('"').strip("'")


def _normalize_base_url(value: str | None) -> str:
	url = _clean_env_value(value)
	if not url:
		return ""
	if not url.startswith(("http://", "https://")):
		url = f"https://{url}"
	return url.rstrip("/")


def _resolve_model_name() -> str:
	# 显式配置优先
	env_model = _clean_env_value(
		os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or os.getenv("CHAT_MODEL")
	)
	if env_model:
		return env_model

	# 针对第三方兼容 OpenAI 网关给出更稳妥默认值
	api_base = _normalize_base_url(os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")).lower()
	if "siliconflow" in api_base:
		return "deepseek-ai/DeepSeek-V3"

	# OpenAI 官方默认
	return "gpt-4o-mini"


# 关键：即使模型名是 deepseek-ai/DeepSeek-V3，也按 OpenAI 兼容协议调用，
# 避免 LangChain 自动识别为 deepseek provider 并要求额外安装 langchain-deepseek。
_resolved_base_url = _normalize_base_url(os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"))
if _resolved_base_url:
	# 统一环境变量，兼容不同组件读取 OPENAI_BASE_URL / OPENAI_API_BASE 的行为。
	os.environ["OPENAI_BASE_URL"] = _resolved_base_url
	os.environ["OPENAI_API_BASE"] = _resolved_base_url

_api_key = _clean_env_value(os.getenv("OPENAI_API_KEY"))

_init_kwargs: dict[str, object] = {
	"model_provider": "openai",
	"temperature": 0,
}
if _resolved_base_url:
	_init_kwargs["base_url"] = _resolved_base_url
if _api_key:
	_init_kwargs["api_key"] = _api_key

model = init_chat_model(
	_resolve_model_name(),
	**_init_kwargs,
)