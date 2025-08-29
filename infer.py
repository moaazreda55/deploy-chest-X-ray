from PIL import Image
import torch
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, mobilenet_v2
from torch import nn
import pickle


model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
model.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
model.fc = nn.Linear(in_features=512,out_features=2,bias=True)

model2 = efficientnet_b0(pretrained=True)
model2.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
model2.classifier[1] = nn.Linear(model2.classifier[1].in_features, 2)

model3 = mobilenet_v2(pretrained=True)
model3.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
model3.classifier[1] = nn.Linear(model3.classifier[1].in_features, 2)

model.load_state_dict(torch.load(r"./model.pth"))
model2.load_state_dict(torch.load(r"./model2.pth"))
model3.load_state_dict(torch.load(r"./model3.pth"))

with open("./lbl_encoder.pkl","rb") as f:
    lbl_encoder = pickle.load(f)


def infer(img):
    img_gray = img.convert("L")
    img_trans = transforms.ToTensor()(img_gray).unsqueeze(0)
    preds = model(img_trans)
    clas_index = torch.argmax(preds).item()
    final_pred = lbl_encoder.inverse_transform([clas_index])[0]
    return final_pred

def infer2(img):
    img_gray = img.convert("L")
    img_trans = transforms.ToTensor()(img_gray).unsqueeze(0)
    preds = model2(img_trans)
    clas_index = torch.argmax(preds).item()
    final_pred = lbl_encoder.inverse_transform([clas_index])[0]
    return final_pred

def infer3(img):
    img_gray = img.convert("L")
    img_trans = transforms.ToTensor()(img_gray).unsqueeze(0)
    preds = model3(img_trans)
    clas_index = torch.argmax(preds).item()
    final_pred = lbl_encoder.inverse_transform([clas_index])[0]
    return final_pred

img_path = r"data\test\PNEUMONIA\person1_virus_12.jpeg"
img = Image.open(img_path)
pred = infer(img)
print(pred)
