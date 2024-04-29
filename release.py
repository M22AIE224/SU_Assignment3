import torch
from sklearn.metrics import roc_auc_score
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from utils import *
from scipy.interpolate import interp1d
from torch.nn.parallel import DataParallel
from model import *

# path of model and data
model_path = 'model\Best_LA_model_for_DF.pth'
data_path ='data\Dataset_Speech_Assignment'

print(model_path)
print(data_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(device=device)

model.to(device)
dictModel = torch.load(model_path,map_location=device)
model = DataParallel(model)
model.load_state_dict(dictModel)


# Assuming you have a function to load and preprocess the dataset
audio_wav, labels = Load_CustomData(data_path)

# Make predictions
with torch.no_grad():
    predictions = model(audio_wav)

# Compute AUC
auc = roc_auc_score(labels, predictions)

# Compute EER
fpr, tpr, thresholds = roc_curve(labels, predictions)
eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

# Report results
print("AUC:", auc)
print("EER:", eer)




