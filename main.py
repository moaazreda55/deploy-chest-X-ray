from fastapi import FastAPI
import io
from PIL import Image
from pydantic import BaseModel
import base64
from infer import infer, infer2, infer3


app = FastAPI()


class RequestData(BaseModel):
    img: str


@app.post("/predict_model")
async def predict(req: RequestData):

   img = io.BytesIO(base64.b64decode(req.img)) 
   img = Image.open(img)
   pred = infer(img)
   return {"pred": pred}


@app.post("/predict_model2")
async def predict2(req: RequestData):

   img = io.BytesIO(base64.b64decode(req.img)) 
   img = Image.open(img)
   pred = infer2(img)
   return {"pred": pred}


@app.post("/predict_model3")
async def predict3(req: RequestData):

   img = io.BytesIO(base64.b64decode(req.img)) 
   img = Image.open(img)
   pred = infer3(img)
   return {"pred": pred}