import os
import sys
from collections import Counter

# Ensure backend root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.training.dataset import AgriPulseDataset, DISEASE_CLASSES

def analyze_class_distribution():
    print("=== Analyzing Class Distribution ===\n")
    
    # We pass dummy directories that don't exist; the class handles it gracefully
    dummy_dirs = {
        "paddy_doctor": "/tmp/nonexistent/paddy",
        "plant_village": "/tmp/nonexistent/pv",
        "plantdoc": "/tmp/nonexistent/pdoc",
        "uav_rice_field": "/tmp/nonexistent/uav"
    }
    
    # Create dataset without triggering the full download/processing of actual files
    # For a real analysis, point to actual paths
    dataset = AgriPulseDataset(data_dirs=dummy_dirs, is_train=True)
    
    # Normally we would iterate through dataset.samples
    # Here we mock it for demonstration if the dataset is empty due to dummy paths
    samples = dataset.samples
    if not samples:
        print("Note: Dataset is empty (dummy paths used). Generating mock distribution for display purposes.")
        samples = [
            {"label": "Healthy"} for _ in range(10407)
        ] + [
            {"label": "Rice Blast (Magnaporthe oryzae)"} for _ in range(5430)
        ] + [
            {"label": "Brown Spot (Bipolaris oryzae)"} for _ in range(2150)
        ] + [
            {"label": "Leaf Scald (Gerlachia oryzae)"} for _ in range(950)
        ] + [
            {"label": "Bacterial Blight (Xanthomonas oryzicola)"} for _ in range(1200)
        ]

    # Count occurrences
    counts = Counter(s["label"] for s in samples)
    total = sum(counts.values())
    
    if total == 0:
        print("No valid samples found to analyze.")
        return
        
    print(f"Total Samples: {total}\n")
    
    # Find the max count for scaling the bar chart
    max_count = max(counts.values())
    max_bar_length = 50
    
    for cls in DISEASE_CLASSES:
        count = counts.get(cls, 0)
        pct = (count / total) * 100 if total > 0 else 0
        
        # Scale bar length relative to the maximum class count
        bar_len = int((count / max_count) * max_bar_length) if max_count > 0 else 0
        bar = "█" * bar_len
        
        # Format output to align columns
        cls_name = cls.split(" (")[0] # Shorten name for display
        print(f"{cls_name:<20} | {count:>6} | {pct:>5.1f}% | {bar}")
        
    print("\n====================================")
    print("Recommendation:")
    
    min_count = min(counts.values())
    if max_count > min_count * 5:
        print("⚠️ SEVERE IMBALANCE DETECTED.")
        print(f"The largest class is {max_count/max(min_count, 1):.1f}x bigger than the smallest class.")
        print("Consider applying Focal Loss or WeightedRandomSampler in datamodule.py.")
    else:
        print("✅ Class distribution is relatively balanced.")

if __name__ == "__main__":
    analyze_class_distribution()
