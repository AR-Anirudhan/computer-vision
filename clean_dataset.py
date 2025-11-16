import os
import cv2
import shutil

def clean_dataset(input_folder='dataset/train',
                 output_folder='dataset/cleaned_dataset',
                 min_size=(100, 100)):
    """
    Clean and validate dataset by removing corrupted/invalid images
    """
    
    shapes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
    
    stats = {
        'total': 0,
        'valid': 0,
        'invalid': 0,
        'by_shape': {shape: 0 for shape in shapes}
    }
    
    print("Cleaning dataset...")
    
    for shape in shapes:
        input_path = os.path.join(input_folder, shape)
        output_path = os.path.join(output_folder, shape)
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        if not os.path.exists(input_path):
            print(f"Skipping {shape} - folder not found")
            continue
        
        files = os.listdir(input_path)
        print(f"\nProcessing {shape}: {len(files)} files")
        
        for filename in files:
            stats['total'] += 1
            file_path = os.path.join(input_path, filename)
            
            # Try to load image
            img = cv2.imread(file_path)
            
            if img is None:
                print(f"  Invalid: {filename}")
                stats['invalid'] += 1
                continue
            
            # Check minimum size
            h, w = img.shape[:2]
            if h < min_size[0] or w < min_size[1]:
                print(f"  Too small: {filename} ({w}x{h})")
                stats['invalid'] += 1
                continue
            
            # Copy valid file
            shutil.copy(file_path, os.path.join(output_path, filename))
            stats['valid'] += 1
            stats['by_shape'][shape] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("CLEANING SUMMARY")
    print("="*60)
    print(f"Total files processed: {stats['total']}")
    print(f"Valid files: {stats['valid']}")
    print(f"Invalid files: {stats['invalid']}")
    print("\nBy shape:")
    for shape, count in stats['by_shape'].items():
        print(f"  {shape:12s}: {count:3d} images")
    print("="*60)

if __name__ == "__main__":
    clean_dataset()
