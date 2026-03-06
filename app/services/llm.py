"""Optional LLM integration for generating answers from retrieved chunks."""

from typing import Optional


def generate_answer(question: str, context_chunks: list[str], api_key: Optional[str], model: str = "gpt-4o-mini") -> Optional[str]:
    """
    If OPENAI_API_KEY is set, call OpenAI to generate an answer given the question and context.
    Otherwise return None (API will only return retrieved chunks).
    """
    if not api_key or not context_chunks:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        context = "\n\n---\n\n".join(context_chunks)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer the user's question using only the provided context. If the context does not contain enough information, say so briefly.",
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            max_tokens=500,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return None
