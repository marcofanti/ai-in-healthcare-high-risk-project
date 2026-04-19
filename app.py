import streamlit as st
import json
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# --- DIAGNOSTIC CHECK ---
if os.getenv("STREAMLIT_SERVER_HEADLESS") != "true":
    print(f"\n[DIAGNOSTIC] Python Executable: {sys.executable}")
    print(f"[DIAGNOSTIC] Python Version: {sys.version}")
    print(f"[DIAGNOSTIC] Current Directory: {os.getcwd()}")
    try:
        import nibabel
        print(f"[DIAGNOSTIC] nibabel found at: {nibabel.__file__}")
    except ImportError:
        print("[DIAGNOSTIC] nibabel NOT FOUND in this environment")
# ------------------------

# Ensure GOOGLE_API_KEY is defined for the agent (Gemini SDK uses GOOGLE_API_KEY)
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.manifest_generator import generate_manifest
from agent.graph import create_agent_graph
from utils.viz_utils import create_medical_viz, get_image_metadata
from utils.query_generator import generate_clinical_questions, improve_clinical_prompt
from utils.file_picker import pick_directory

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
        default_dir = os.getenv("LOCAL_STAGING_DIR", "./workspace/mock_oasis")

        # Apply any pending update from the Browse button BEFORE the
        # text_input widget is instantiated this run.
        if "_pending_staging_dir" in st.session_state:
            st.session_state.staging_dir_input = st.session_state._pending_staging_dir
            del st.session_state._pending_staging_dir

        if "staging_dir_input" not in st.session_state:
            st.session_state.staging_dir_input = default_dir

        workspace_dir = st.text_input("Local Staging Directory", key="staging_dir_input")

        if st.button("📁 Browse for Directory", use_container_width=True):
            picked = pick_directory()
            if picked:
                st.session_state._pending_staging_dir = picked
                st.rerun()
            else:
                st.info("No directory selected.")

        if st.button("Scan Directory", type="primary", use_container_width=True):
             if not os.path.exists(workspace_dir):
                 st.error(f"Directory not found: {workspace_dir}")
             else:
                 manifest_str = generate_manifest(workspace_dir)
                 st.session_state.manifest_data = json.loads(manifest_str)
                 st.success("Manifest generated!")

    # Main Panel: Interactive Setup
    if "manifest_data" in st.session_state:
        manifest_data = st.session_state.manifest_data
        file_options = [f"{item['file_path']} ({item['type']})" for item in manifest_data]
        selected_file_str = st.selectbox("Select File for Analysis", options=file_options)
        
        # Extract file path and type
        selected_index = file_options.index(selected_file_str)
        selected_item = manifest_data[selected_index]
        file_path = selected_item["file_path"]
        modality = selected_item["type"]
        
        # --- MEDIA GALLERY (PREVIEW) ---
        st.header("2. Media Gallery")
        with st.container(border=True):
            col_viz, col_meta = st.columns([3, 1])
            
            with col_viz:
                viz_buf = create_medical_viz(file_path, modality)
                if viz_buf:
                    st.image(viz_buf, width=512, caption=f"Visualization for {file_path}")
                else:
                    st.warning("Could not generate visualization for this format.")
            
            with col_meta:
                st.subheader("Technical Metadata")
                metadata = get_image_metadata(file_path, modality)
                for k, v in metadata.items():
                    st.markdown(f"**{k}:** `{v}`")

        # --- ENSEMBLE CONFIGURATION ---
        st.header("3. Ensemble Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Select Experts")
            suggested_models = MODALITY_MODEL_MAPPING.get(modality, MODALITY_MODEL_MAPPING["Unknown"])
            selected_models = st.multiselect("Ensemble Models", 
                                            options=["biomedclip", "conch", "musk", "medgemma", "vit_alzheimer", "chexagent", "llava_med"],
                                            default=suggested_models[:2])
            
        with col2:
            st.subheader("Clinical Query")

            # Reset query state whenever the user picks a different file
            if st.session_state.get("_last_selected_file") != file_path:
                st.session_state._last_selected_file = file_path
                st.session_state.query_mode = None
                st.session_state.generated_questions = []
                st.session_state.generated_questions_for_file = None
                st.session_state.clinical_query_widget = ""
                st.session_state.pop("generated_question_select", None)

            # Callbacks (run pre-widget-render on the next rerun, so they can
            # safely mutate clinical_query_widget even when fired from a
            # widget that sits below the text area).
            def _on_select_generated():
                st.session_state.clinical_query_widget = st.session_state.generated_question_select

            def _improve_current():
                draft = st.session_state.get("clinical_query_widget", "")
                if not draft.strip():
                    st.session_state._improve_warning = "Write a question first, then click Improve."
                    return
                try:
                    improved = improve_clinical_prompt(draft, file_path, modality)
                    st.session_state.clinical_query_widget = improved
                except Exception as e:
                    st.session_state._improve_warning = f"Failed to improve prompt: {e}"

            # Mode toggle buttons (rendered above the text area so direct
            # session_state mutation in their handlers takes effect this run).
            btn_cols = st.columns(2)
            gen_clicked = btn_cols[0].button(
                "✨ Pre-generated Questions", use_container_width=True, key="mode_generated_btn"
            )
            custom_clicked = btn_cols[1].button(
                "✍️ Custom Question", use_container_width=True, key="mode_custom_btn"
            )

            if gen_clicked and st.session_state.query_mode != "generated":
                st.session_state.query_mode = "generated"
                if st.session_state.generated_questions_for_file != file_path:
                    try:
                        with st.spinner("Generating clinical questions..."):
                            st.session_state.generated_questions = generate_clinical_questions(
                                file_path, modality
                            )
                        st.session_state.generated_questions_for_file = file_path
                        st.session_state.pop("generated_question_select", None)
                    except Exception as e:
                        st.error(f"Failed to generate questions: {e}")
                        st.session_state.generated_questions = []
                if st.session_state.generated_questions:
                    st.session_state.clinical_query_widget = st.session_state.generated_questions[0]

            if custom_clicked and st.session_state.query_mode != "custom":
                st.session_state.query_mode = "custom"
                st.session_state.clinical_query_widget = ""

            # Generated mode: dropdown renders above the text area so its
            # on_change callback can update the text area this run.
            if st.session_state.query_mode == "generated" and st.session_state.generated_questions:
                # Expand each dropdown option on hover so long questions are
                # fully readable before selecting. Targets BaseWeb's popover
                # markup that Streamlit's selectbox renders into.
                st.markdown(
                    """
                    <style>
                    /* Default: truncate each option and its inner text wrapper */
                    div[data-baseweb="popover"] [role="option"],
                    div[data-baseweb="popover"] [role="option"] * {
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                    }
                    /* Hover: expand the option AND every nested child so the
                       full question is visible before selection */
                    div[data-baseweb="popover"] [role="option"]:hover,
                    div[data-baseweb="popover"] [role="option"]:hover * {
                        white-space: normal !important;
                        word-break: break-word !important;
                        overflow: visible !important;
                        text-overflow: clip !important;
                        height: auto !important;
                        max-height: none !important;
                        min-height: fit-content !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                st.selectbox(
                    "Select a generated question",
                    options=st.session_state.generated_questions,
                    key="generated_question_select",
                    on_change=_on_select_generated,
                )

            user_prompt = st.text_area("Your Question", key="clinical_query_widget", height=140)

            # Custom mode: Improve button sits below the text area; callback
            # fires on the next rerun, pre-widget, so mutation is safe.
            if st.session_state.query_mode == "custom":
                st.button(
                    "🤖 Improve with AI",
                    on_click=_improve_current,
                    use_container_width=True,
                    key="improve_prompt_btn",
                )
                if "_improve_warning" in st.session_state:
                    st.warning(st.session_state._improve_warning)
                    del st.session_state._improve_warning
            
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
