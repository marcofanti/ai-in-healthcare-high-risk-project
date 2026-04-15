from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """
    LangGraph state schema for MedHuggingGPT Interactive Ensemble.
    """
    user_manual_selections: Dict[str, Any] # {'file_path': '...', 'models': [...], 'prompt': '...'}
    execution_manifest: List[Dict[str, Any]] # List of {model, path, prompt}
    model_outputs: List[Dict[str, Any]]      # Raw JSON from tools
    clinical_report: str                    # Final synthesized markdown
    status: str                             # Current status
    images_to_render: List[str]             # UI helper
