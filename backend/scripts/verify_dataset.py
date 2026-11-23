import os
import sys

# Ensure backend root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.training.dataset import AgriPulseDataset
from app.training.data_cleaner import DataCleaner

def run_verification():
    print("=== AgriPulse Dataset Verification ===")
    
    # 1. Test the Data Cleaner Logic
    cleaner = DataCleaner(
        target_classes=["Healthy", "Rice Blast (Magnaporthe oryzae)", "Brown Spot (Bipolaris oryzae)"]
    )
    
    print("\n[Testing Label Harmonization]")
    test_labels = ["healthy", "rice_blast", "brown_spot", "early blight", "random_nonsense"]
    for lbl in test_labels:
        print(f"Raw: '{lbl}' -> Harmonized: '{cleaner.harmonize_label(lbl)}'")
        
    print("\n[Testing Severity Heuristics]")
    test_sev_labels = ["healthy", "early blight", "late blight", "rice_blast"]
    for lbl in test_sev_labels:
        print(f"Raw: '{lbl}' -> Severity: {cleaner.estimate_severity(lbl)}%")

    print("\n[Testing Unified Dataset Initialization]")
    # We pass dummy directories that don't exist; the class handles it gracefully
    dummy_dirs = {
        "paddy_doctor": "/tmp/nonexistent/paddy",
        "plant_village": "/tmp/nonexistent/pv",
        "plantdoc": "/tmp/nonexistent/pdoc",
        "uav_rice_field": "/tmp/nonexistent/uav"
    }
    
    dataset = AgriPulseDataset(data_dirs=dummy_dirs, is_train=True)
    print(f"Dataset successfully initialized. Total valid samples loaded: {len(dataset)}")
    print("Schema is ready to output: {'image', 'label', 'severity', 'dataset', 'source', 'metadata'}")

if __name__ == "__main__":
    run_verification()
