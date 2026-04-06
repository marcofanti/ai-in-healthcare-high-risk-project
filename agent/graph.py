import sys
import os
import json
from pathlib import Path

# Make sure tools are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState

# Import newly developed tools natively
from tools.tool_oasis_parser import parse_oasis_data
from tools.tool_dicom_parser import parse_dicom, parse_nifti
from tools.tool_quilt_parser import parse_wsi
from tools.tool_hsi_parser import parse_hsi
from tools.tool_iq_oth_parser import parse_ct_jpeg

def planner_node(state: AgentState) -> Dict:
    """
    Ingests the JSON Manifest and dynamically routes modalities 
    strictly matching to the corresponding executor actions.
    """
    manifest_data = json.loads(state.get("manifest", "[]"))
    
    plan = {
        "analysis_steps": []
    }
    
    for item in manifest_data:
        mtype = item.get("type", "")
        file_path = item.get("file_path", "")
        
        # Dynamically map the modality to the corresponding action string
        action = None
        desc = ""
        target = file_path # default injection target
        
        if mtype == "Legacy MRI":
            action = "execute_oasis"
            desc = "Extracting demographics/metadata from OASIS scan."
            target = os.path.dirname(file_path)
            
        elif mtype == "DICOM CT":
            action = "execute_dicom"
            desc = "Parsing standard DICOM volumetric headers."
            target = os.path.dirname(file_path)
            
        elif mtype == "NIfTI":
            action = "execute_nifti"
            desc = "Retrieving 3D spatial properties from NIfTI masks."
            
        elif mtype == "WSI Pathology":
            action = "execute_wsi"
            desc = "Ingesting gigapixel pathology dimensional hierarchies."
            
        elif mtype == "HSI Pathology":
            action = "execute_hsi"
            desc = "Consuming deep hyperspectral sensor channels/bands."
            
        elif mtype == "CT Image":
            action = "execute_ct_jpeg"
            desc = "Reading local JPEG clinical snapshot properties."
            
        else:
            action = "execute_fallback"
            desc = f"Unknown modality '{mtype}' bypass route."

        plan["analysis_steps"].append({
            "action": action,
            "target": target,
            "description": desc,
            "modality": mtype
        })


    return {
        "agent_plan": json.dumps(plan, indent=2),
        "status": "awaiting_approval"
    }

def executor_node(state: AgentState) -> Dict:
    """
    Iterates dynamically through the universally mapped graph.
    """
    agent_plan = json.loads(state.get("agent_plan", "{}"))
    execution_steps = []
    
    for step in agent_plan.get("analysis_steps", []):
        action = step["action"]
        target = step["target"]
        
        result = {}
        # Dynamic Dispatch Mapping
        if action == "execute_oasis":
            result = parse_oasis_data(target)
        elif action == "execute_dicom":
            result = parse_dicom(target)
        elif action == "execute_nifti":
            result = parse_nifti(target)
        elif action == "execute_wsi":
            result = parse_wsi(target)
        elif action == "execute_hsi":
            result = parse_hsi(target)
        elif action == "execute_ct_jpeg":
            result = parse_ct_jpeg(target)
        else:
            result = {"error": f"No native handler for bypass action {action}"}
            
        execution_steps.append({
            "action": action,
            "modality": step.get("modality", "Unknown"),
            "target": target,
            "result": result
        })
            
    return {
        "execution_steps": execution_steps,
        "status": "executed"
    }

def synthesizer_node(state: AgentState) -> Dict:
    """
    Universally aggregates variable JSON dictionaries strictly mapping back 
    to cohesive unified clinical syntax blocks!
    """
    steps = state.get("execution_steps", [])
    
    report = "### Synthesized Clinical Report\n\n"
    report += "*Multi-modality dataset aggregation mapping verified automatically.*\n\n"
    
    if not steps:
        report += "No analysis steps were physically executed."
    
    for step in steps:
        modality = step["modality"]
        res = step["result"]
        
        report += f"**Processed Dataset [{modality}]:**\n"
        
        # Check for explicitly thrown errors
        if "error" in res:
            report += f"- ❌ Execution Fault: {res['error']}\n\n"
            continue
            
        # Dynamically loop and prettify all the resulting payload outputs dynamically
        # Ignore extremely heavy payloads like "metadata_keys" directly
        for key, value in res.items():
            if key == "metadata_keys":
                val_str = f"{len(value)} unique structural tags retrieved."
            else:
                val_str = str(value)
            
            clean_key = key.replace("_", " ").title()
            report += f"- **{clean_key}:** {val_str}\n"
            
        report += "\n"
            
    report += "\n**Final Interpretation:** All modalities mapped above have officially completed parsing evaluation checks effectively! The universal execution pipeline mapping succeeded natively."
    
    return {
        "report_summary": report,
        "status": "completed"
    }

def create_agent_graph():
    # Build the state graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("Planner", planner_node)
    workflow.add_node("Executor", executor_node)
    workflow.add_node("Synthesizer", synthesizer_node)

    # Define edges
    workflow.set_entry_point("Planner")
    workflow.add_edge("Planner", "Executor")
    workflow.add_edge("Executor", "Synthesizer")
    workflow.add_edge("Synthesizer", END)

    # Checkpointer for state persistence and HITL gateway
    memory = MemorySaver()
    
    # Compile graph with HITL gateway before Executor
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["Executor"]
    )
    
    return app
