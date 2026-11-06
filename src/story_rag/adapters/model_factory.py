from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

from story_rag.config import ConfigError, Settings
from story_rag.logging_config import silence_third_party_console
from story_rag.pipelines.token_budget import TokenBudget


class BudgetedOpenAILike(OpenAILike):
    """Honor request-level limits instead of replacing them with model defaults."""

    def _get_model_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        thinking = kwargs.pop("thinking", None)
        model_kwargs = super()._get_model_kwargs(**kwargs)
        if "max_tokens" in kwargs:
            model_kwargs["max_tokens"] = kwargs["max_tokens"]
        if thinking is not None:
            model_kwargs["extra_body"] = {
                **model_kwargs.get("extra_body", {}),
                "thinking": thinking,
            }
        return model_kwargs


def build_embedding_model(settings: Settings) -> OpenAIEmbedding:
    silence_third_party_console()
    return OpenAIEmbedding(
        # model_name bypasses the built-in OpenAI model enum so any LM Studio
        # embedding model id can be used through the OpenAI-compatible endpoint.
        model_name=settings.embedding_model,
        api_base=settings.lmstudio_base_url,
        api_key=settings.lmstudio_api_key,
        embed_batch_size=16,
        timeout=settings.request_timeout_seconds,
    )


def build_llm(settings: Settings) -> OpenAILike:
    return OpenAILike(
        model=settings.llm_model,
        api_base=settings.deepseek_base_url,
        api_key=settings.llm_api_key,
        is_chat_model=True,
        is_function_calling_model=False,
        context_window=1_000_000,
        max_tokens=8192,
        temperature=0.1,
        timeout=settings.request_timeout_seconds,
        max_retries=2,
    )


def build_conversation_llm(settings: Settings) -> BudgetedOpenAILike:
    return BudgetedOpenAILike(
        model=settings.llm_model,
        api_base=settings.deepseek_base_url,
        api_key=settings.llm_api_key,
        is_chat_model=True,
        is_function_calling_model=False,
        context_window=TokenBudget().context_limit,
        max_tokens=8_192,
        temperature=0.1,
        timeout=settings.request_timeout_seconds,
        max_retries=0,
    )


def build_reranker(settings: Settings, *, device: str | None = None) -> SentenceTransformerRerank:
    if not settings.reranker_model_path.is_dir():
        raise ConfigError(f"Reranker model path does not exist: {settings.reranker_model_path}")
    silence_third_party_console()
    return SentenceTransformerRerank(
        model=str(settings.reranker_model_path),
        top_n=settings.rerank_top_n,
        device=device,
        keep_retrieval_score=True,
        cross_encoder_kwargs={"max_length": settings.reranker_max_length},
    )


@dataclass(slots=True)
class ModelBundle:
    """Swap-in point for tests: real models in production, fakes in tests."""

    llm: Any
    embedding: Any
    reranker: Any
    conversation_llm: Any | None = None


def model_bundle(settings: Settings) -> ModelBundle:
    return ModelBundle(
        llm=build_llm(settings),
        embedding=build_embedding_model(settings),
        reranker=build_reranker(settings),
        conversation_llm=build_conversation_llm(settings),
    )
