from PIL import Image
import sys

try:
    img_path = r'e:\projects\agri new\model_weights\publication_results\figures\fig7_per_class_metrics.png'
    img = Image.open(img_path)
    print(f"Original size: {img.size}")
    
    # Crop the top 45 pixels to remove the title
    img_cropped = img.crop((0, 45, img.width, img.height))
    img_cropped.save(img_path)
    
    print(f"Cropped size: {img_cropped.size}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
