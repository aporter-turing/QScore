import os
import uuid
import logging
import argparse
from pathlib import Path
from permutations.permutations import QScore

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='Run permutation analysis on a design file')
    parser.add_argument('--path', type=str, help='Path to data file')
    parser.add_argument('--task_type', default="object_naming", type=str, help='Type of task to run (motor or object_naming)')
    parser.add_argument('--condition', default="CON1", type=str, help='Condition to run')
    
    # Parse arguments
    args = parser.parse_args()

    # Input file is a nifti file
    nifti_file = Path(args.path)

    # Set logging level
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # This output directory will need to have a randomly generated name
    session_uuid = uuid.uuid4()
    output_file = nifti_file.parent 
    output_folder = nifti_file.stem
    output_directory = Path(f"/tmp/{session_uuid}")
    output_directory.mkdir(parents=True, exist_ok=True)
    logging.info(f"Set output directory: {output_directory}")

    # base folder is current directory
    base_folder = Path(os.environ.get("QSCORE_PATH", "/app/q_score"))
    
    # Start permutation analysis
    try:
        data = QScore(
                        base_folder = base_folder,
                        output_path = output_directory,
                        original_nifti = nifti_file
                    )

        data.get_q_score()

    except Exception as e:
        logging.error(f"Error setting up permutations: {e}")
