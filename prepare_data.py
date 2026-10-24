import os
import shutil
from tqdm import tqdm

RAW_DIR = "raw_dataset/leapGestRecog" # Extracted Kaggle data
PROCESSED_DIR = "processed_dataset"

# The 10 subjects are folders '00' to '09'
# Let's use 00-07 for training, 08 for validation, and 09 for the unseen test subject
SPLITS = {
    'train': ['00', '01', '02', '03', '04', '05', '06', '07'],
    'val': ['08'],
    'test_unseen': ['09']
}

def setup_directories():
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)

    # Get class names from the first subject folder
    classes = os.listdir(os.path.join(RAW_DIR, '00'))

    for split in SPLITS.keys():
        for cls in classes:
            os.makedirs(os.path.join(PROCESSED_DIR, split, cls), exist_ok=True)
    return classes

def process_data():
    classes = setup_directories()
    print("Reorganizing dataset for strict subject-wise splitting...")

    for split_name, subjects in SPLITS.items():
        for subject in subjects:
            subject_path = os.path.join(RAW_DIR, subject)
            if not os.path.exists(subject_path):
                continue

            for cls in classes:
                src_dir = os.path.join(subject_path, cls)
                dest_dir = os.path.join(PROCESSED_DIR, split_name, cls)

                for img in os.listdir(src_dir):
                    src_file = os.path.join(src_dir, img)
                    dest_file = os.path.join(dest_dir, f"{subject}_{img}")
                    shutil.copy2(src_file, dest_file)

        print(f"Populated {split_name} with subjects: {subjects}")

if __name__ == "__main__":
    process_data()
