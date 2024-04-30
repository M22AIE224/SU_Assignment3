import torch
from sklearn.metrics import roc_auc_score
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from utils import *
from scipy.interpolate import interp1d
from torch.nn.parallel import DataParallel

from model import ModelNM
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('--runtime',type=str,help='mode name')
args=parser.parse_args()

if __name__ == "__main__":
    print("Runtime Mode ", args.runtime)
    
    mode = args.runtime
    # path of model and data
    
    evalpretrain_data_path ='data/Dataset_Speech_Assignment'
    finetune_data_path = 'data/for-2seconds/training'
    eval_data_path = 'data/for-2seconds/validation'
    finetune_model_path = "models/fine_tuned.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelNM(device=device)
    model.to(device)
    if mode == 'pretrained':
        model_path = 'models\Best_LA_model_for_DF.pth'
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

        #print("checkpoint", checkpoint)

        # Assuming 'model' is wrapped with DataParallel, unwrap it
        model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        model.load_state_dict(checkpoint)
        #print("model", model)
        #print("Loaded state dictionary")

       
    else : 
        train_data = For2_Data(finetune_data_path)
        eval_data = For2_Data(finetune_data_path)


        train_loader = DataLoader(train_data, batch_size=14, shuffle=True)
        eval_loader = DataLoader(eval_data, batch_size=14, shuffle=True)
              
        model = finetune(train_loader,eval_loader, model, device,lr=0.001, epochs=5)
        # save fine-tuned model
        torch.save(model.state_dict(), finetune_model_path)
        


    # Preprocess data
    dataset = Load_CustomData(evalpretrain_data_path)
    audio_wav, labels = DataLoader(dataset, batch_size=32, shuffle=True)

    # Make predictions
    with torch.no_grad():
        predictions = model(audio_wav)

    predicted_labels = model.predict(audio_wav)

    # calculate AUC
    auc = roc_auc_score(labels, predictions)

    # calculate Accuracy
    accuracy = calculate_accuracy(labels, predicted_labels)

    # ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, predicted_labels)
    auc = roc_auc_score(labels, predicted_labels)

    # call Equal Error Rate (EER) function
    eer = calculate_eer(labels, predicted_labels)

    #Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

