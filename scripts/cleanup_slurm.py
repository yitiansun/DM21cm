import os
import re
import shutil
from datetime import datetime

def group_slurm_files(slurm_folder):
    # Regular expression pattern to match the filenames
    pattern = re.compile(r'^(.+)_\d{1,2}\.(out|err)$')

    # Dictionary to hold run_name as keys and their corresponding files as values
    run_name_dict = {}

    # List all files in the slurm directory
    for filename in os.listdir(slurm_folder):
        # Construct full file path
        file_path = os.path.join(slurm_folder, filename)

        # Skip directories
        if not os.path.isfile(file_path):
            continue

        # Match the filename with the pattern
        match = pattern.match(filename)
        if match:
            run_name = match.group(1)

            # Add the file to the corresponding run_name key
            run_name_dict.setdefault(run_name, []).append(filename)

    # Create folders and move files
    for run_name, files in run_name_dict.items():
        # Create a new directory for the run_name if it doesn't exist
        run_dir = os.path.join(slurm_folder, run_name)
        # Append today's date MMDD if directory already exists
        while os.path.exists(run_dir):
            run_dir += datetime.now().strftime("%m%d")
        os.makedirs(run_dir, exist_ok=True)

        # Move each file into the corresponding run_name directory
        for file in files:
            src = os.path.join(slurm_folder, file)
            dst = os.path.join(run_dir, file)
            shutil.move(src, dst)


if __name__ == "__main__":

    group_slurm_files("../outputs/slurm")
