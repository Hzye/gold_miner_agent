from langchain_ollama import ChatOllama
from typing import List


def get_ollama_llm(
    model: str = "qwen3:8b-q8_0",
    temperature: float = 0
):
    """Init ollama llm"""
    return ChatOllama(
        model=model,
        temperature=temperature
    )

def get_llm_with_tools(tools: List):
    """Get llm configured with tools"""
    llm = get_ollama_llm()
    # bind tools
    return llm.bind_tools(tools)