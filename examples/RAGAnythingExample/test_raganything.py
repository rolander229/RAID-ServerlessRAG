import asyncio
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

"""
RAGAnything Qwen3 Interface Test
"""

async def main():
    # Set up API configuration
    api_key = "sk-vllm-api-key-123456"
    base_url = "http://192.168.5.65:30100/v1"  # Optional

    # Define LLM model function
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        # Await the async call so we return the final response instead of a coroutine
        return await openai_complete_if_cache(
            "qwen3-1.7b",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    llm_response = await llm_model_func("Hello, how are you?", system_prompt="You are a helpful assistant.")
    print(llm_response)


if __name__ == "__main__":
    asyncio.run(main())