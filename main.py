import io
import torch
import uvicorn
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import gesture

Gesture_dict = gesture.Gesture_dict

path = "./model/"
state_dict = torch.load(path+"pytorch_model.bin")
model = ViTForImageClassification.from_pretrained(path)

model.load_state_dict(state_dict)
model.eval()
feature_extractor = ViTFeatureExtractor.from_pretrained(path)

app = FastAPI()

def prepare_image(image):
    image = Image.open(io.BytesIO(image))
    inputs = feature_extractor(image, return_tensors='pt')
    inputs = inputs['pixel_values']
    return inputs


# Define the endpoint

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    inputs = prepare_image(contents)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.logits, dim=1)
    #print(predicted.item())
    result = Gesture_dict[predicted.item()]
    return {"predicted_class": result}

