from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """
    LangGraph state schema for MedHuggingGPT routing and execution.
    """
    user_query: str
    manifest: str          # JSON string of manifest metadata
    agent_plan: str        # JSON string containing the execution plan
    execution_steps: List[Dict[str, Any]] # Intermediate steps and tool outputs
    report_summary: str    # Final synthesized clinical report text
    images_to_render: List[str] # Local paths to output images for UI
    status: str            # Current status of the agent
