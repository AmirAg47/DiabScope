import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
import logging
import pandas as pd
from .utils import (
    load_encoder, load_model, load_scaler, predict_data, spectrum_processor
    
)
from pathlib import Path

@shared_task(bind=True, max_retries=3, default_retry_delay=5)
def run_prediction_task(self, data_dict, results_file_path):
    """
    Celery task to perform prediction and save results to CSV file.
    
    Parameters:
        data_dict: dictionary version of DataFrame (e.g., from .to_dict())
        results_file_path: str - path to save results
    """
    try:
        logging.info("Prediction task started.")

        # بازیابی دیتافریم از دیکشنری
        new_df = pd.DataFrame.from_dict(data_dict)

        # Load tools
        gender_encoder = load_encoder()
        scaler = load_scaler()
        model_path = os.path.join('models', 'dnn_model.tflite')

        # Predict
        predictions_df = predict_data(model_path, new_df, scaler, gender_encoder)

        # Save
        predictions_df.to_csv(results_file_path, index=False, encoding='utf-8-sig')
        logging.info(f"Prediction written to {results_file_path}")

    except Exception as e:
        logging.error(f"Prediction task failed: {str(e)}")
        raise self.retry(exc=e)


def delete_file_later(file_path):
    """Delete a file after 2 minutes."""
    try:
        time.sleep(10)  # Wait 2 minutes
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
        else:
            logger.warning(f"Tried to delete missing file: {file_path}")
    except Exception as e:
        logger.error(f"Error deleting file: {file_path}, error: {e}")