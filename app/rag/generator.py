from app.core.config import settings
import json
import requests
import os
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception

def is_transient_error(e):
    if isinstance(e, requests.exceptions.HTTPError):
        if e.response is not None:
            return e.response.status_code in (429, 500, 502, 503, 504)
        return False
    if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    return False

def log_retry(retry_state):
    print(f"Retrying LLM API call: attempt {retry_state.attempt_number} failed with error: {retry_state.outcome.exception()}")

class Generator:
    def generate(self, query: str, context: str) -> dict:
        max_context_length = 24000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "... [context truncated]"

        prompt = f"""
You are an expert AI assistant answering questions based strictly on the provided context.

Context:
{context}

Question:
{query}

Instructions:
1. Answer only from the retrieved context.
2. Do not invent facts.
3. If the answer is not in the context, say "The information is unavailable."
4. When stating facts from the context, cite the supporting sources using bracket notation like [1], [2] corresponding to the source numbers provided in the context.

Answer:"""
        
        provider = settings.llm_provider.lower()
        
        try:
            if provider == "ollama":
                res = self._call_ollama(prompt)
            elif provider == "gemini":
                res = self._call_gemini(prompt)
            elif provider == "groq":
                res = self._call_groq(prompt)
            else:
                return {
                    "answer": "LLM integration currently only supports ollama, gemini, or groq.",
                    "success": False,
                    "error": f"Unsupported provider: {provider}",
                    "llm_calls": 0,
                    "token_usage": None
                }

            answer_text = res.get("answer", "")
            if answer_text.startswith("Failed to generate answer:"):
                return {
                    "answer": answer_text,
                    "success": False,
                    "error": answer_text,
                    "llm_calls": 0,
                    "token_usage": None
                }

            return {
                "answer": answer_text,
                "success": True,
                "error": None,
                "llm_calls": res.get("llm_calls", 1),
                "token_usage": res.get("token_usage", None)
            }
        except Exception as e:
            print(f"Error calling {provider}: {e}")
            return {
                "answer": "Failed to generate answer due to LLM error.",
                "success": False,
                "error": str(e),
                "llm_calls": 1,
                "token_usage": None
            }
            
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(4),
        retry=retry_if_exception(is_transient_error),
        before_sleep=log_retry,
        reraise=True
    )
    def _call_ollama(self, prompt: str) -> dict:
        url = f"http://localhost:11434/api/generate"
        payload = {
            "model": os.getenv("OLLAMA_MODEL", "llama3.1"),
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return {"answer": data.get("response", ""), "llm_calls": 1, "token_usage": data.get("eval_count")}

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(4),
        retry=retry_if_exception(is_transient_error),
        before_sleep=log_retry,
        reraise=True
    )
    def _call_gemini(self, prompt: str) -> dict:
        api_key = settings.gemini_api_key
        if not api_key:
            return {"answer": "Failed to generate answer: GEMINI_API_KEY is missing in .env", "llm_calls": 0, "token_usage": None}
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
        response.raise_for_status()
        data = response.json()
        usage = data.get("usageMetadata", {}).get("totalTokenCount", None)
        return {"answer": data["candidates"][0]["content"]["parts"][0]["text"], "llm_calls": 1, "token_usage": usage}
            
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(4),
        retry=retry_if_exception(is_transient_error),
        before_sleep=log_retry,
        reraise=True
    )
    def _call_groq(self, prompt: str) -> dict:
        api_key = settings.groq_api_key
        if not api_key:
            return {"answer": "Failed to generate answer: GROQ_API_KEY is missing in .env", "llm_calls": 0, "token_usage": None}
            
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
            "messages": [{"role": "user", "content": prompt}]
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {}).get("total_tokens", None)
        return {"answer": data["choices"][0]["message"]["content"], "llm_calls": 1, "token_usage": usage}

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(4),
        retry=retry_if_exception(is_transient_error),
        before_sleep=log_retry,
        reraise=True
    )
    def _call_groq_json(self, json_prompt: str) -> str:
        api_key = settings.groq_api_key
        if not api_key:
            return "{}"
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
            "messages": [{"role": "user", "content": json_prompt}],
            "response_format": {"type": "json_object"}
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def generate_json(self, prompt: str) -> str:
        json_prompt = prompt + "\n\nIMPORTANT: You must respond with ONLY valid JSON. Do not include markdown blocks or any other text."
        
        provider = settings.llm_provider.lower()
        try:
            if provider == "ollama":
                return self._call_ollama(json_prompt)["answer"]
            elif provider == "gemini":
                return self._call_gemini(json_prompt)["answer"]
            elif provider == "groq":
                return self._call_groq_json(json_prompt)
        except Exception as e:
            print(f"Error calling {provider} JSON: {e}")
            return "{}"
        return "{}"

generator = Generator()

