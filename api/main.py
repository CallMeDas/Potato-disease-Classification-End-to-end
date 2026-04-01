from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image


app = FastAPI()

@app.get('/ping')
def ping():
    return 'hello! working'


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open (BytesIO(data)))
    return image



@app.get('/predict')
async def predict():
    file : UploadFile = File(...)
    image = read_file_as_image(await file.read())
    return 'ok'


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)
