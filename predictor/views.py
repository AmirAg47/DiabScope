import pandas as pd
import numpy as np
import uuid
import os
import time
import logging
import threading
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from pathlib import Path
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.core.files.uploadedfile import TemporaryUploadedFile
from django.shortcuts import redirect
from .forms import CSVUploadForm
from .utils import load_encoder, load_model, predict_data, load_scaler, spectrum_processor
import tflite_runtime.interpreter as tflite
from celery.result import AsyncResult
from django.http import JsonResponse
from pathlib import Path  # Add this import at the top if not already imported
from .tasks import run_prediction_task
from .tasks import delete_file_later



def delete_file_after_delay(file_path, delay):
    def delete_file():
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} deleted after {delay} seconds.")
    threading.Timer(delay, delete_file).start()



def check_task_status(request, task_id):
    result = AsyncResult(str(task_id))
    if result.ready():
        if result.successful():
            return JsonResponse({
                'ready': True,
                'status': 'success',
                'file': f"{settings.MEDIA_URL}predictions_{task_id}.csv"
            })
        elif result.state == 'FAILURE':
            return JsonResponse({
                'ready': True,
                'status': 'error',
                'message': str(result.result)
            })
    return JsonResponse({'ready': False})

# Background prediction function
def run_prediction_in_background(new_df, results_file_path):
    try:
        import logging
        from .utils import load_model, load_scaler, load_encoder  # Assuming you have these
        from .utils import spectrum_processor, predict_data  # Assuming you wrote these

        logging.info("Prediction started.")

        # Load encoders and scalers
        gender_encoder = load_encoder()
        scaler = load_scaler()
        model = load_model()

        # Predict using the updated predict_data function
        predictions_df = predict_data(model, new_df, scaler, gender_encoder)

        # Save predictions to file
        predictions_df.to_csv(results_file_path, index=False, encoding='utf-8-sig')
        logging.info(f"Prediction written to {results_file_path}")

    except Exception as e:
        logging.error(f"Prediction failed: {e}")

def predict(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']

            media_root_path = Path(settings.MEDIA_ROOT)
            file_path = default_storage.save(uploaded_file.name, uploaded_file)
            full_path = media_root_path / Path(file_path)

            try:
                new_df = pd.read_csv(full_path)
                if len(new_df) > 1:
                    form.add_error('file', 'The uploaded file should not have more than 1 sample.')
                    return render(request, 'predictor/upload.html', {'form': form})
                    
                unique_filename = f'predictions_{uuid.uuid4().hex}.csv'
                results_file_path = media_root_path / Path(unique_filename)
                results_file_url = settings.MEDIA_URL + unique_filename

                # ارسال تسک به celery
                run_prediction_task.delay(new_df.to_dict(orient='records'), str(results_file_path))

        

                # نمایش پیام پردازش در حال انجام است
                return render(request, 'predictor/results_pending.html', {
                    'results_file_url': results_file_url,
                    'message': 'Prediction is being processed. The download link will be available shortly.'
                })

            except Exception as e:
                # لاگ یا مدیریت خطا
                form.add_error(None, 'An error occurred while processing your file.')

            finally:
                if os.path.exists(full_path):
                    os.remove(full_path)
    else:
        form = CSVUploadForm()

    return render(request, 'predictor/upload.html', {'form': form})

# Other views
def redirect_to_predict(request):
    return redirect('predict', permanent=True)

def about(request):
    return render(request, 'predictor/about.html')

def contact(request):
    return render(request, 'predictor/contact.html')

def upload(request):
    return render(request, 'predictor/upload.html')
