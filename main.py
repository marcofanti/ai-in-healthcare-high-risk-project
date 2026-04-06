import json
import os
from agent.graph import create_agent_graph

def main():
    # Construct a theoretical clinical manifest spanning the fully mapped scope
    manifest = [
        {
            "id": "item1",
            "type": "Legacy MRI",
            "file_path": "week1/data/Oasis1/OAS1_0001_MR1/RAW/OAS1_0001_MR1_mpr-1_anon.hdr"
        },
        {
            "id": "item2",
            "type": "DICOM CT",
            "file_path": "week1/data/Spinal_DICOM/Myel_001/MonoE_80keVHU/1-0001.dcm"
        },
        {
            "id": "item3",
            "type": "WSI Pathology",
            "file_path": "week1/data/Quilt1M_quilt/dTr3MNl1FxE_image_c54e9a8d-9348-456a-9645-3b8921eb0b79.jpg"
        },
        {
            "id": "item4",
            "type": "HSI Pathology",
            "file_path": "week1/data/PKG_HistologyHSI_GB/P1/ROI_01_C01_T/raw.hdr"
        },
        {
            "id": "item5",
            "type": "CT Image",
            "file_path": "week1/data/IQ-OTH_NCCD/archive/Bengin case (85).jpg"
        }
    ]
    
    app = create_agent_graph()
    initial_state = {"manifest": json.dumps(manifest)}
    config = {"configurable": {"thread_id": "test_thread_multimodal"}}
    
    print("\n[MOCK] Invoking Graph -> Planner...")
    for event in app.stream(initial_state, config):
        for key, value in event.items():
            print(f"Node [{key}] completed successfully!")

    print("\n[MOCK] State should now be heavily awaited internally due to HITL interrupt!")
    state = app.get_state(config)
    print(f"Current Next node target: {state.next}")
    
    plan = json.loads(state.values.get("agent_plan", "{}"))
    print("\nGenerated Evaluation Plan:")
    for step in plan.get("analysis_steps", []):
         print(f" - {step['action']}: {step['description']}")
            
    print("\n[MOCK] Resuming graph execution (Agent HITL Authorization Triggered)...")
    for event in app.stream(None, config):
        for key, value in event.items():
             print(f"Node [{key}] completed successfully!")
             
    final_state = app.get_state(config)
    print("\n================== FINAL REPORT ==================\n")
    print(final_state.values.get("report_summary", "No Report Returned"))
    print("\n==================================================\n")

if __name__ == "__main__":
    main()
