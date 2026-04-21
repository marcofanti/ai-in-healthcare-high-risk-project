import warnings
warnings.filterwarnings("ignore")

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

from agent.graph import create_agent_graph
from utils.viz_utils import create_medical_viz, get_image_metadata
from utils.query_generator import (
    generate_clinical_questions,
    improve_clinical_prompt,
    validate_query_compatibility,
)

DATASETS_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "datasets_config.json")

def load_datasets_config() -> dict:
    if os.path.exists(DATASETS_CONFIG_PATH):
        with open(DATASETS_CONFIG_PATH) as f:
            return json.load(f)
    return {}

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

    # Sidebar: Dataset selection from config
    file_path = None
    modality = None

    with st.sidebar:
        st.header("1. Dataset Selection")
        datasets = load_datasets_config()

        if not datasets:
            st.warning("No datasets configured. Add entries to `datasets_config.json`.")
        else:
            selected_datasets = []
            for ds_name, ds_info in datasets.items():
                if st.checkbox(f"{ds_name}  —  *{ds_info['modality']}*", key=f"ds_{ds_name}"):
                    selected_datasets.append(ds_name)

            # Flatten files from all checked datasets into a single selectbox
            file_options = []
            for ds_name in selected_datasets:
                ds = datasets[ds_name]
                for f in ds["files"]:
                    file_options.append({
                        "label": f"{ds_name} / {os.path.basename(f['path'])}  ({f['type']})",
                        "path": f["path"],
                        "modality": ds["modality"],
                    })

            if file_options:
                labels = [o["label"] for o in file_options]
                chosen_label = st.selectbox("Select File for Analysis", options=labels)
                chosen = next(o for o in file_options if o["label"] == chosen_label)
                file_path = chosen["path"]
                modality = chosen["modality"]

                # Clear visualization and analysis when file selection changes
                if file_path != st.session_state.get("visualized_file"):
                    st.session_state.pop("visualized_file", None)
                    st.session_state.pop("agent_state", None)

                if st.button("Visualize", type="primary", use_container_width=True):
                    st.session_state.visualized_file = file_path
                    st.session_state.pop("agent_state", None)
                    st.session_state.thread_id = "sess_" + str(os.urandom(4).hex())

            elif selected_datasets:
                st.info("No files found in selected datasets.")

    is_visualized = file_path and st.session_state.get("visualized_file") == file_path

    # Main Panel: Interactive Setup
    if file_path and modality:

        # --- MEDIA GALLERY (PREVIEW) ---
        st.header("2. Media Gallery")
        with st.container(border=True):
            col_viz, col_meta = st.columns([3, 1])

            with col_meta:
                st.subheader("Technical Metadata")
                metadata = get_image_metadata(file_path, modality)
                for k, v in metadata.items():
                    st.markdown(f"**{k}:** `{v}`")

            with col_viz:
                if is_visualized:
                    viz_buf = create_medical_viz(file_path, modality)
                    if viz_buf:
                        st.image(viz_buf, width=512, caption=f"Visualization for {file_path}")
                    else:
                        st.warning("Could not generate visualization for this format.")
                else:
                    st.info("Click **Visualize** in the sidebar to load the image preview.")

    # --- ENSEMBLE CONFIGURATION (only after visualization) ---
    if is_visualized:
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

            default_prompt = MODALITY_PROMPT_MAPPING.get(modality, MODALITY_PROMPT_MAPPING["Unknown"])

            # Reset to default question whenever the user picks a different file
            if st.session_state.get("_last_selected_file") != file_path:
                st.session_state._last_selected_file = file_path
                st.session_state.generated_questions = []
                st.session_state.generated_questions_for_file = None
                st.session_state.clinical_query_widget = default_prompt
                st.session_state.pop("generated_question_select", None)

            if "clinical_query_widget" not in st.session_state:
                st.session_state.clinical_query_widget = default_prompt

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

            user_prompt = st.text_area("Your Question", key="clinical_query_widget", height=140)

            btn_cols = st.columns(2)
            gen_clicked = btn_cols[0].button(
                "✨ Generate More Questions", use_container_width=True, key="gen_questions_btn"
            )
            btn_cols[1].button(
                "🤖 Improve with AI",
                on_click=_improve_current,
                use_container_width=True,
                key="improve_prompt_btn",
            )

            if "_improve_warning" in st.session_state:
                st.warning(st.session_state._improve_warning)
                del st.session_state._improve_warning

            if gen_clicked:
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

            if st.session_state.get("generated_questions"):
                st.markdown(
                    """
                    <style>
                    div[data-baseweb="popover"] [role="option"],
                    div[data-baseweb="popover"] [role="option"] * {
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                    }
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
                    "Select a generated question to use",
                    options=st.session_state.generated_questions,
                    key="generated_question_select",
                    on_change=_on_select_generated,
                )
            
        # Analysis Trigger with Conditional Approval
        run_triggered = st.button("Run Ensemble Analysis 🚀", use_container_width=True)
        override_engaged = bool(st.session_state.pop("_force_run_override", False))

        if run_triggered or override_engaged:
            if not selected_models:
                st.error("Please select at least one model.")
                st.stop()
            if not user_prompt.strip():
                st.error("Please enter or select a clinical query before running.")
                st.stop()

            if not override_engaged:
                # Normal path: pre-flight query/model compatibility validation
                validation = None
                with st.status("Validating Query...", expanded=True) as status:
                    try:
                        validation = validate_query_compatibility(
                            user_prompt, selected_models, modality, file_path
                        )
                        if validation.is_valid:
                            status.update(label="✅ Query Valid", state="complete", expanded=False)
                        else:
                            status.update(label="❌ Query Not Valid", state="error", expanded=True)
                            st.markdown(f"**Reason:** {validation.reasoning}")
                            if validation.incompatible_models:
                                st.markdown(
                                    "**Selected models unsuitable for this query:** "
                                    + ", ".join(f"`{m}`" for m in validation.incompatible_models)
                                )
                            if validation.recommended_models:
                                st.markdown(
                                    "**Recommended models from the available pool:** "
                                    + ", ".join(f"`{m}`" for m in validation.recommended_models)
                                )
                            st.info(
                                "Update your query or change the selected models, then click "
                                "**Run Ensemble Analysis** again — or override this check with "
                                "**Proceed Anyway** below."
                            )
                    except Exception as e:
                        status.update(label=f"Validation failed: {e}", state="error", expanded=True)

                if validation is None:
                    st.stop()

                if not validation.is_valid:
                    # Offer override — click sets a session flag and reruns so
                    # the next pass enters the override branch above and skips
                    # validation entirely.
                    if st.button(
                        "⚠️ Proceed Anyway",
                        key="proceed_anyway_btn",
                        use_container_width=True,
                    ):
                        st.session_state._force_run_override = True
                        st.rerun()
                    st.stop()

                # Soft warning for compatible-but-imperfect selections
                if validation.incompatible_models:
                    st.warning(
                        "Some selected models may be a weak fit: "
                        + ", ".join(f"`{m}`" for m in validation.incompatible_models)
                        + ". Running anyway because at least one selected model is capable."
                    )
            else:
                st.info(
                    "⚠️ Validation bypassed — running ensemble with **Proceed Anyway** override. "
                    "Results may be unreliable for the selected model set."
                )

            # Conditional Gate (runs on both normal-valid and override paths)
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
