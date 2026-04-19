import os
from dotenv import load_dotenv

load_dotenv()

llm_provider = os.getenv("LLM_PROVIDER", "google").lower()

if llm_provider == "ollama":
    from langchain_ollama import ChatOllama

    model_name = os.getenv("OLLAMA_MODEL", "llama3")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = ChatOllama(model=model_name, base_url=base_url)
else:
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

    model_name = os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-flash")
    llm = ChatGoogleGenerativeAI(model=model_name)
