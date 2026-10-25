from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from . import config as C


def get_llm():
    if C.PROVIDER == "ollama":
        return ChatOllama(model=C.OLLAMA_CHAT_MODEL, base_url=C.OLLAMA_BASE_URL, temperature=0)
    return ChatOpenAI(model=C.OPENAI_CHAT_MODEL, temperature=0)


def get_embeddings():
    if C.PROVIDER == "ollama":
        return OllamaEmbeddings(model=C.OLLAMA_EMBED_MODEL, base_url=C.OLLAMA_BASE_URL)
    return OpenAIEmbeddings(model=C.OPENAI_EMBED_MODEL)
