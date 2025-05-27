import pandas as pd
import numpy as np
import uuid
import re
from typing import List, Optional
import pickle
from pathlib import Path  
from django.conf import settings 
import tflite_runtime.interpreter as tflite 
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d



def spectrum_processor(df, target_points=3736):
    """
    تبدیل طیف‌های هر ردیف به بردار ۳۷۳۶‌تایی با بازنمونه‌گیری.
    فرض می‌شود فقط دو ستون اول دیتا (GENDER و AGE) غیرداده‌ی طیفی هستند.

    Parameters:
        df (pd.DataFrame): دیتافریم شامل ستون‌های GENDER، AGE و سپس داده‌های طیفی.
        target_points (int): تعداد ویژگی‌هایی که طیف باید به آن تبدیل شود (پیش‌فرض: 3736)

    Returns:
        pd.DataFrame: دیتافریمی شامل ۳۷۳۶ ستون طیفی برای هر ردیف.
    """

    spectra_raw = df.iloc[:, 3:]  # همه ستون‌ها به جز GENDER و AGE
    x_orig = np.arange(spectra_raw.shape[1])  # موقعیت اصلی ستون‌ها
    x_target = np.linspace(x_orig.min(), x_orig.max(), target_points)  # موقعیت هدف بازنمونه‌گیری

    interpolated = []

    for i, row in spectra_raw.iterrows():
        y = row.values.astype(float)
        valid = ~np.isnan(y)
        if valid.sum() < 2:
            interpolated.append(np.zeros(target_points))
        else:
            f = interp1d(x_orig[valid], y[valid], kind='linear', fill_value="extrapolate", bounds_error=False)
            y_interp = f(x_target)
            interpolated.append(y_interp)

    interpolated = np.array(interpolated)
    return pd.DataFrame(interpolated, columns=[f'spec_{i}' for i in range(target_points)])



def load_encoder():
    with open(Path(settings.BASE_DIR) / 'models/one_hot_encoder.pkl', 'rb') as f:
        return pickle.load(f)


def load_model():
    # Path to your .keras model (update the path as necessary)
    model_path = Path(settings.BASE_DIR) / 'models/dnn_model.tflite'  # Update to your model's path

    # Load the model using TensorFlow's Keras API
    model = tf.keras.models.load_model(model_path)
    
    return model

def load_scaler():
    """
    Loads the scaler from a pickle file located in the 'models' directory.
  
    """
    # Load the scaler from the pickle file using the path to 'models'
    with open(Path(settings.BASE_DIR) / 'models/scaler.pkl', 'rb') as f:
        return pickle.load(f)





def predict_data(tflite_model_path, data, scaler, gender_encoder, feature_indices=None) -> pd.DataFrame:
    """
    Predict binary class (positive/negative) using a trained model.
    
    Parameters:
        model: Trained TensorFlow/Keras model
        data: DataFrame including 'GENDER', 'AGE', and spectral columns
        scaler: Pre-fitted scaler (e.g., MinMaxScaler)
        gender_encoder: Encoder for 'GENDER' column (LabelEncoder or OneHotEncoder)
        feature_indices: Optional list of selected feature indices
    
    Returns:
        pd.DataFrame with columns: 'ID' (if exists), 'GENDER', 'AGE', 'Prediction'
    """

    df = data.copy()
    
    # Optional: extract ID if exists
    id_col = df['ID'] if 'ID' in df.columns else None
    if 'ID' in df.columns:
        df = df.drop(columns='ID')
    
    # Encode 'GENDER'
    if hasattr(gender_encoder, 'categories_'):  # OneHotEncoder
        gender_encoded = gender_encoder.transform(df[['GENDER']]).toarray()
    else:
        gender_encoded = gender_encoder.transform(df['GENDER']).reshape(-1, 1)
    
    # Extract 'AGE'
    age = df['AGE'].values.reshape(-1, 1)

    # Process spectra
    spectra = spectrum_processor(df)

    # Combine all inputs
    X = np.hstack([age, spectra, gender_encoded])
    X_scaled = scaler.transform(X)

    if feature_indices is not None:
        X_scaled = X_scaled[:, feature_indices]
    
    # Load TFLite model
    interpreter = tflite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    preds = []
    for row in X_scaled:
        row = row.reshape(1, -1).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], row)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        preds.append(output[0][0])

    class_preds = (np.array(preds) >= 0.5).astype(int).flatten()
    labels = np.where(class_preds == 1, 'positive', 'negative')

    # Assemble result
    result = pd.DataFrame({
        'GENDER': data['GENDER'],
        'AGE': data['AGE'],
        'Prediction': labels
    })

    if id_col is not None:
        result.insert(0, 'ID', id_col)

    return result