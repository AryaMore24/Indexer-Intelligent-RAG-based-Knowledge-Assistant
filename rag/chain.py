from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from . import config as C
from .models import get_llm
from .retriever import HybridRetriever

SYSTEM_PROMPT = f"""You are a knowledge assistant. Answer ONLY using the context below.
Rules:
- Use no outside knowledge and make no guesses.
- If the context does not contain the answer, reply with exactly:
  "{C.FALLBACK}"
- Keep answers concise and factual."""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "Context:\n{context}\n\nQuestion: {question}")]
)


class Assistant:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.chain = PROMPT | get_llm() | StrOutputParser()

    def ask(self, question: str) -> tuple[str, list[str]]:
        docs = self.retriever.invoke(question)
        if not docs:
            return C.FALLBACK, []
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        answer = self.chain.invoke({"context": context, "question": question}).strip()
        sources = sorted({d.metadata.get("source", "unknown") for d in docs})
        return answer, sources
