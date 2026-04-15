import streamlit as st
import json
import os
import sys

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.manifest_generator import generate_manifest
from agent.graph import create_agent_graph

# Initialize LangGraph app globally so memory checkpointer persists across Streamlit reruns
if "agent_app" not in st.session_state:
    st.session_state.agent_app = create_agent_graph()

# Model mapping by modality
MODALITY_MODEL_MAPPING = {
    "Legacy MRI": ["vit_alzheimer", "biomedclip", "medgemma", "llava_med"],
    "2D Histopathology": ["conch", "musk", "biomedclip", "llava_med"],
    "3D Volumetric": ["biomedclip", "medgemma", "llava_med", "chexagent"],
    "Hyperspectral": ["biomedclip", "musk"],
    "CT Image": ["chexagent", "biomedclip", "medgemma", "llava_med"],
    "DICOM CT": ["chexagent", "biomedclip", "medgemma", "llava_med"],
    "NIfTI": ["biomedclip", "medgemma", "llava_med"],
    "WSI Pathology": ["conch", "musk", "biomedclip"],
    "HSI Pathology": ["biomedclip", "musk"],
    "Unknown": ["biomedclip", "llava_med", "medgemma"]
}

# Default prompts by modality
MODALITY_PROMPT_MAPPING = {
    "Legacy MRI": "Analyze this brain MRI for signs of Alzheimer's disease or structural atrophy.",
    "2D Histopathology": "Identify the primary tissue type and detect any signs of malignancy or adenocarcinoma.",
    "3D Volumetric": "Segment the key anatomical structures and report any pathological findings.",
    "Hyperspectral": "Analyze the spectral signatures to differentiate between tumor and non-tumor regions.",
    "CT Image": "Describe the findings in this chest CT slice, specifically looking for lung nodules or masses.",
    "DICOM CT": "Perform a comprehensive radiological analysis of this CT volume.",
    "Unknown": "Perform a general clinical analysis of this medical image."
}

def init_app():
    st.set_page_config(page_title="MedHuggingGPT (Interactive Ensemble)", layout="wide")
    st.title("🧩 MedHuggingGPT - Interactive Medical AI Ensemble")
    st.markdown("Phase 3: Interactive Ensemble - Multi-model clinical analysis with LLM consensus.")
    
    # Session state for thread ID
    if "thread_id" not in st.session_state:
         st.session_state.thread_id = "sess_" + str(os.urandom(4).hex())

    # Sidebar for File Selection & Manifest
    with st.sidebar:
        st.header("1. Ingestion")
        workspace_dir = st.text_input("Local Staging Directory", value="./workspace/mock_oasis")
        
        if st.button("Scan Directory", type="primary"):
             if not os.path.exists(workspace_dir):
                 st.error(f"Directory not found: {workspace_dir}")
             else:
                 manifest_str = generate_manifest(workspace_dir)
                 st.session_state.manifest_data = json.loads(manifest_str)
                 st.session_state.workspace_dir = workspace_dir
                 st.success("Manifest generated!")

    # Main Panel: Interactive Setup
    st.header("2. Ensemble Configuration")
    
    if "manifest_data" in st.session_state:
        manifest_data = st.session_state.manifest_data
        file_options = [f"{item['file_path']} ({item['type']})" for item in manifest_data]
        selected_file_str = st.selectbox("Select File for Analysis", options=file_options)
        
        # Extract file path and type
        selected_index = file_options.index(selected_file_str)
        selected_item = manifest_data[selected_index]
        file_path = selected_item["file_path"]
        modality = selected_item["type"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Select Experts")
            suggested_models = MODALITY_MODEL_MAPPING.get(modality, MODALITY_MODEL_MAPPING["Unknown"])
            selected_models = st.multiselect("Ensemble Models", 
                                            options=["biomedclip", "conch", "musk", "medgemma", "vit_alzheimer", "chexagent", "llava_med"],
                                            default=suggested_models[:2])
            
        with col2:
            st.subheader("Clinical Query")
            default_prompt = MODALITY_PROMPT_MAPPING.get(modality, MODALITY_PROMPT_MAPPING["Unknown"])
            user_prompt = st.text_area("Your Question", value=default_prompt)
            
        # Analysis Trigger with Conditional Approval
        if st.button("Run Ensemble Analysis 🚀", use_container_width=True):
            if not selected_models:
                st.error("Please select at least one model.")
            else:
                # Conditional Gate
                is_heavy = modality == "Hyperspectral" or len(selected_models) > 3
                if is_heavy:
                    st.warning("⚠️ High Latency Warning: This ensemble may take >30 seconds to execute.")
                
                # Setup selections for the agent
                st.session_state.user_manual_selections = {
                    "file_path": file_path,
                    "models": selected_models,
                    "prompt": user_prompt
                }
                
                # Invoke LangGraph
                run_langgraph(st.session_state.user_manual_selections)

    # Result Display
    if "agent_state" in st.session_state:
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        current_state = st.session_state.agent_app.get_state(config)
        agent_status = current_state.values.get("status", "unknown")
        
        if agent_status == "completed":
             st.success("✅ Analysis Completed")
             
             tab1, tab2 = st.tabs(["Synthesized Report", "Model Evidence (Audit Trail)"])
             
             with tab1:
                  st.markdown(current_state.values.get("clinical_report", "No report generated."))
                  
             with tab2:
                  st.markdown("### Raw Model Outputs")
                  outputs = current_state.values.get("model_outputs", [])
                  for out in outputs:
                      with st.expander(f"Model: {out.get('model', 'Unknown')}"):
                          st.json(out)
                  
             if st.button("Start New Analysis"):
                  st.session_state.pop("agent_state")
                  st.rerun()
        else:
             st.info(f"Current Agent Status: {agent_status}")

def run_langgraph(selections):
    # Setup initial state
    initial_state = {
        "user_manual_selections": selections,
        "status": "started"
    }
    
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    # Run the graph
    app = st.session_state.agent_app
    with st.spinner("Agent Orchestrating Ensemble..."):
        # Run until completion
        for output in app.stream(initial_state, config):
            st.session_state.agent_state = output
        
if __name__ == "__main__":
    init_app()
