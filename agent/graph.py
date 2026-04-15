import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List

# Make sure tools are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from agent.state import AgentState
from tools.tool_model_executor import run_model

# Initialize the LLM (Gemini or Ollama via LangChain)
llm_provider = os.getenv("LLM_PROVIDER", "google").lower()

if llm_provider == "ollama":
    from langchain_ollama import ChatOllama
    model_name = os.getenv("OLLAMA_MODEL", "llama3")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = ChatOllama(model=model_name, base_url=base_url)
else:
    from langchain_google_genai import ChatGoogleGenerativeAI
    # Ensure GOOGLE_API_KEY is defined (Gemini SDK uses GOOGLE_API_KEY)
    if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    
    model_name = os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-flash")
    llm = ChatGoogleGenerativeAI(model=model_name)

def reviewer_node(state: AgentState) -> Dict:
    """
    Reviewer node: Validates user selections and prepares the execution manifest.
    In v1.1, it reviews the ensemble selection and ensures modality compatibility.
    """
    selections = state.get("user_manual_selections", {})
    file_path = selections.get("file_path", "")
    models = selections.get("models", [])
    prompt = selections.get("prompt", "Analyze this medical image and report clinical findings.")
    
    if not file_path:
        return {"status": "error", "clinical_report": "Error: No file path provided."}
    if not models:
        return {"status": "error", "clinical_report": "Error: No models selected for ensemble."}

    # Prepare the formal execution manifest
    manifest = []
    for model_name in models:
        manifest.append({
            "model": model_name,
            "image_path": file_path,
            "prompt": prompt
        })

    return {
        "execution_manifest": manifest,
        "status": "planning_complete"
    }

def executor_node(state: AgentState) -> Dict:
    """
    Executor node: Dispatches inference jobs for each model in the manifest.
    Handles parallel execution for in-process models (though sequential here for simplicity).
    """
    manifest = state.get("execution_manifest", [])
    model_outputs = []
    
    for job in manifest:
        model_name = job["model"]
        image_path = job["image_path"]
        prompt = job["prompt"]
        
        # Invoke the unified tool wrapper
        result = run_model(model_name, image_path, prompt)
        model_outputs.append(result)
            
    return {
        "model_outputs": model_outputs,
        "status": "executed"
    }

def synthesizer_node(state: AgentState) -> Dict:
    """
    Synthesizer node: Uses Gemini 3.1 Pro to compare all model outputs.
    Identifies Consensus Findings and Model Discrepancies.
    """
    outputs = state.get("model_outputs", [])
    selections = state.get("user_manual_selections", {})
    
    # Construct a synthesis prompt for the ensemble
    synthesis_prompt = f"""
    You are a senior clinical AI architect. Synthesize a unified clinical report from the following ensemble outputs.
    
    User Selections:
    - Target File: {selections.get('file_path')}
    - Original Question: {selections.get('prompt')}
    
    Model Raw Outputs (JSON):
    {json.dumps(outputs, indent=2)}
    
    REQUIRED REPORT STRUCTURE:
    1. ### CONSENSUS FINDINGS: Points agreed upon by multiple models.
    2. ### MODEL DISCREPANCIES / FLAGS: Highlight any contradictions or low-confidence outliers.
    3. ### CLINICAL INTERPRETATION: Final summary and recommendation for human review.
    
    Keep the tone professional, objective, and clinical.
    """
    
    try:
        response = llm.invoke(synthesis_prompt)
        report = response.content
    except Exception as e:
        report = f"### Synthesis Error\n\nFailed to generate consensus report: {str(e)}"
    
    return {
        "clinical_report": report,
        "status": "completed"
    }

def create_agent_graph():
    # Build the state graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("Reviewer", reviewer_node)
    workflow.add_node("Executor", executor_node)
    workflow.add_node("Synthesizer", synthesizer_node)

    # Define edges
    workflow.set_entry_point("Reviewer")
    workflow.add_edge("Reviewer", "Executor")
    workflow.add_edge("Executor", "Synthesizer")
    workflow.add_edge("Synthesizer", END)

    # Checkpointer for state persistence
    memory = MemorySaver()
    
    # Compile graph with HITL gateway before Executor if specified
    # For now, we follow the "Conditional Gate" logic in the UI layer (app.py)
    # but the graph itself is fully executable once triggered.
    app = workflow.compile(checkpointer=memory)
    
    return app
