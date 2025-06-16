import os
import json
import time
import queue
import logging
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Define a queue for file paths to be analyzed
analysis_queue = queue.Queue()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define a custom event handler to watch for file creation in /incoming
class NiftiFileHandler(FileSystemEventHandler):
    """
    Watchdog class for handling events in nifti listener directory
    """            
    def on_closed(self, event):
        """
        Handle file creation events for nifti files
        """

        if event.is_directory:
            return None
        if event.src_path.endswith('.nii'):
            logging.info("New nifti file detected")
            logging.info(event.src_path)
            analysis_queue.put(event.src_path)

def start_listener(nifti_directory: Path):
    """
    Start listening for nifti files in the incoming directory and add them to the analysis queue

    Args:
        nifti_directory: Path contaning to where nifti files will be written
    """

    event_handler = NiftiFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=nifti_directory, recursive=False)
    observer.start()
    
    logging.info(f"Started listening for nifti files in {nifti_directory}")

    try:
        while True:
            # Check if there are files in the analysis queue to process
            if not analysis_queue.empty():  

                # Get the nifti path to analyze
                file_to_analyze = analysis_queue.get()

                logging.info(f"Analyzing nifti files in {file_to_analyze}")

                # Make call to analysis script
                call_analysis_script(file_to_analyze)

                logging.info(f"Finished analysis for {file_to_analyze}")

            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def call_analysis_script(nifti_file_path):
    """
    Call the analysis script to analyze a nifti file

    Args:
        nifti_file_path: Path to nifti file to be analyzed
    """

    q_score_path = str(os.environ.get("QSCORE_PATH", '/app/qscore'))
    path_to_script = f'{q_score_path}/run_analysis.py'

    command = ["python3", path_to_script, "--path", nifti_file_path]

    # Run command
    try:
        subprocess.run(command, text=True)
    except:
        raise Exception(f"Error running command {command}")
