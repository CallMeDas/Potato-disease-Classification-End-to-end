from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI()

MODEL = tf.keras.models.load_model('../models/1.keras')
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

@app.get('/ping')
def ping():
    return 'hello! working'


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open (BytesIO(data)))
    return image



@app.post('/predict')
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        'Class': predicted_class,
        'Confidence': float(confidence)
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8001)
