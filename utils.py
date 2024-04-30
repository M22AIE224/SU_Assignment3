import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import librosa

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from sklearn.metrics import roc_curve, auc , roc_auc_score
#from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder

def Load_CustomData(data_path):
    #audio_waves = [] # Audio List
    audio_paths = []
    labels = [] # Labels assuming 'Real' is class 0 and 'Fake' is class 1

    
    classes = ['Real', 'Fake']
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)

    # Read all files and mark their labels
    for class_name in classes:
        class_dir = os.path.join(data_path, class_name)
        class_label = 0 if class_name == 'Real' else 1
        for audio_file in os.listdir(class_dir):
            if audio_file.endswith('.wav'):  # Assuming audio files are in WAV format
                # Load audio file
                audio_path = os.path.join(class_dir, audio_file)
                #waveform, sample_rate = librosa.load(audio_path, sr=None)

                
                #mean_waveform = np.mean(waveform)

                # Append features and labels
                #audio_waves.append(mean_waveform)
                audio_paths.append(audio_path)

                labels.append(class_label)

    
    #labels = label_encoder.transform(labels)

    # Convert lists to numpy arrays
    audio_waves = np.array(audio_waves)
    labels = np.array(labels)

    return audio_waves, labels

def For2_Data(data_path):
    #audio_waves = [] # Audio List
    audio_paths = []
    labels = [] # Labels assuming 'Real' is class 0 and 'Fake' is class 1

    
    classes = ['Real', 'Fake']
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)

    # Read all files and mark their labels
    for class_name in classes:
        class_dir = os.path.join(data_path, class_name)
        class_label = 0 if class_name == 'Real' else 1
        for audio_file in os.listdir(class_dir):
            if audio_file.endswith('.wav'):  # Assuming audio files are in WAV format
                # Load audio file
                audio_path = os.path.join(class_dir, audio_file)
                #waveform, sample_rate = librosa.load(audio_path, sr=None)

                
                #mean_waveform = np.mean(waveform)

                # Append features and labels
                #audio_waves.append(mean_waveform)
                audio_paths.append(audio_path)

                labels.append(class_label)

    
    #labels = label_encoder.transform(labels)

    # Convert lists to numpy arrays
    audio_waves = np.array(audio_waves)
    labels = np.array(labels)

    return audio_waves, labels

def find_auc(all_embeddings, labels, accuracy,eer,eer_threshold,model_name="pre-trained"):
   
    embeddings_np = all_embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()
    positive_class_index = 0 

    scores_positive = embeddings_np[:, positive_class_index]
    labels_positive = labels_np[:, positive_class_index]

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels_positive, scores_positive)

    auc_score = roc_auc_score(labels, all_embeddings)
   
    #auc_score = auc(fpr, tpr)
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.suptitle(f"Accuracy: {accuracy}, EER: {eer}, EER Threshold: {eer_threshold}")
    plt.legend(loc="lower right")
    plt.savefig(f"roc_curve_{model_name}.png")
    plt.show()
    return auc

import numpy as np

def calculate_accuracy(predicted_labels, true_labels):
    
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)
    
   
    correct_predictions = np.sum(predicted_labels == true_labels)
    
    # Calculate accuracy
    accuracy = correct_predictions / len(true_labels)
    
    return accuracy

def calculate_eer(labels, predicted_labels):

    genuine_scores = [score for i, score in enumerate(predicted_labels) if labels[i] == 1]
    impostor_scores = [score for i, score in enumerate(predicted_labels) if labels[i] == 0]

    
    thresholds = sorted(predicted_labels)
    eer = None
    min_diff = float('inf')

    for threshold in thresholds:
        far = sum(score >= threshold for score in impostor_scores) / len(impostor_scores)
        frr = sum(score < threshold for score in genuine_scores) / len(genuine_scores)
        diff = abs(far - frr)

        
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2

    return eer


def finetune(train_loader, eval_loader, model, device, lr=0.001, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Evaluation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            eval_loss = 0.0

            for inputs, labels in eval_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                eval_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            eval_loss /= len(eval_loader)
            eval_acc = 100. * correct / total

            print(f"Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.2f}%")

        model.train()

    return model
