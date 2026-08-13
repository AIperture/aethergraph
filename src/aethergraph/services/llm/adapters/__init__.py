"""Exact endpoint-adapter runtime implementations."""

from .anthropic import AnthropicMessagesAdapter
from .azure import AzureChatAdapter
from .chat import ChatAdapterInvocation, invoke_chat_adapter
from .embedding import (
    EmbeddingAdapterInvocation,
    invoke_embedding_adapter,
    registered_embedding_adapter_ids,
)
from .gemini import GeminiGenerateContentAdapter
from .image import ImageAdapterInvocation, invoke_image_adapter, registered_image_adapter_ids
from .openai_compatible import OpenAICompatibleChatAdapter
from .openai_responses import OpenAIResponsesAdapter
from .stream import (
    ChatStreamInvocation,
    invoke_chat_stream_adapter,
    registered_chat_stream_adapter_ids,
)

__all__ = [
    "AnthropicMessagesAdapter",
    "AzureChatAdapter",
    "ChatAdapterInvocation",
    "ChatStreamInvocation",
    "EmbeddingAdapterInvocation",
    "GeminiGenerateContentAdapter",
    "ImageAdapterInvocation",
    "OpenAICompatibleChatAdapter",
    "OpenAIResponsesAdapter",
    "invoke_chat_adapter",
    "invoke_chat_stream_adapter",
    "invoke_embedding_adapter",
    "invoke_image_adapter",
    "registered_image_adapter_ids",
    "registered_embedding_adapter_ids",
    "registered_chat_stream_adapter_ids",
]
