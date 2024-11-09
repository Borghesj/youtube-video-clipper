import os
import logging

def safe_remove(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Successfully removed: {file_path}")
        else:
            logging.info(f"File not found, skipping removal: {file_path}")
    except Exception as e:
        logging.error(f"Error removing {file_path}: {str(e)}")