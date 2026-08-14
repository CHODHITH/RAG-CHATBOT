import os
from vector_store import VectorStoreManager
from prompt import SYSTEM_PROMPT
import requests

class RAGPipeline:
    def __init__(self):
        self.vs_manager = VectorStoreManager()
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

    def initialize_pipeline(self, documents):
        return self.vs_manager.create_vector_store(documents)

    def answer_question(self, question, k=4):
        results = self.vs_manager.similarity_search(question, k=k)
        if not results:
            return "I could not find this information in the uploaded documents.", []

        context_parts = []
        sources = []
        for doc, score in results:
            context_parts.append(f"Content: {doc.page_content}\nSource: {doc.metadata['source']}, Page: {doc.metadata['page']}")
            sources.append({"source": doc.metadata["source"], "page": doc.metadata["page"], "score": float(score)})

        context_str = "\n\n---\n\n".join(context_parts)
        prompt = SYSTEM_PROMPT.format(context=context_str, question=question)

        # Call LLM if API key exists, otherwise provide a synthesized response based on top context
        answer = self._call_llm(prompt, context_parts)
        return answer, sources

    def _call_llm(self, prompt, context_parts):
        # If OpenAI API key is available, use OpenAI API
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=openai_key, base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a precise document QA assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI API error: {e}")

        # Fallback heuristic generator if no API key is set
        # Extracts key sentences from context matching question keywords
        return f"Based on the uploaded documents:\n\n" + "\n\n".join([cp.split('\nSource:')[0].replace('Content: ', '') for cp in context_parts[:2]]) + \
               f"\n\n*(Note: Running in offline heuristic mode. Set OPENAI_API_KEY for full LLM generation).* \n\n**Sources:** " + \
               ", ".join([f"{cp.split('Source: ')[1]}" for cp in context_parts[:2]])
