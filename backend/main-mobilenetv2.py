from fastapi import FastAPI, File, UploadFile
import kagglehub
import numpy as np
import keras
import os
import io
import httpx
import tensorflow as tf
from PIL import Image
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from keras.preprocessing import image
# from keras.applications import MobileNetV2
# from tensorflow.python.keras import  layers, models # datasets
# from tensorflow.python.keras.application.mobile
# from tensorflow.python.keras #import MobileNetV2, preprocess_input
# from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.preprocessing import image


img_path = os.path.expanduser("~/REPO1/assignment2/.localenv/resized/apple.png")
appleBoolean = False

# model = MobileNetV2(weights='imagenet')

model=keras.applications.MobileNetV2(
    input_shape=None,
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name=None,
)


def check_if_apple(img_path):
     try:
          img =image.load_img(img_path)
          img_array = image.img_to_array(img)
          img_array = np.expand_dims(img_array, axis=0)
          img_array = preprocess_input(img_array)
        
        # Make prediction
          predictions = model.predict(img_array)
          decoded = decode_predictions(predictions, top=3)[0]
        
        # Check if any of the top 3 predictions are apple-related
        # ImageNet classes: 'Granny_Smith', 'eating_apple', etc.
          apple_keywords = ['banana', 'apple', 'potato']
     
          for _, class_name, probability in decoded:
            class_lower = class_name.lower()
            if any(keyword in class_lower for keyword in apple_keywords):
                print(f"Detected: {class_name} with probability {probability:.2%}")
                return True
        
          print(f"Top prediction: {decoded[0][1]} with probability {decoded[0][2]:.2%}")
          return False
     except Exception as e :
          print(f"ERR--CRASHED {e}" )
          return False
     
# apple = check_if_apple(img_path)


     
app = FastAPI()
# path = kagglehub.dataset_download("moltean/fruits")

@app.post("/hello")
async def read_root():
     return {"message": "hello world"}

@app.post("/test1")
async def test1():
     return {"message": tf.version.VERSION + tf.teras.version()}
     
@app.post("/test3")
async def test3():
     is_apple = check_if_apple(img_path)
     return {"message": is_apple}

@app.post("/analyze")
async def analyze_image(image_url: str):
    # Your n8n webhook URL
     n8n_webhook = "https://your-n8n-instance.com/webhook/analyze-image"
     async with httpx.AsyncClient() as client:
          response = await client.post(
               n8n_webhook,
               json={"image_url": image_url}
          )
    
     return response.json()

@app.post("/appleornot")
async def decide():
     return {"message": appleBoolean}

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
     """
     Returns the top 3 predictions for the image.
     """
     try:
          contents = await file.read()
          
          img = Image.open(io.BytesIO(contents)).resize((224, 224))
          
          img_array = image.img_to_array(img)
          img_array = np.expand_dims(img_array, axis=0)
          img_array = preprocess_input(img_array)
          
          predictions = model.predict(img_array)
          decoded = decode_predictions(predictions, top=3)[0]
          
          results = [
               {
                    "class": class_name,
                    "probability": float(probability)
               }
               for _, class_name, probability in decoded
          ]
          
          return {"predictions": results, "image_path": img_path}
          
     except Exception as e:
          return {"error": str(e)}