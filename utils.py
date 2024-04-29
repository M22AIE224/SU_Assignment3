import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import librosa


import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder

def Load_CustomData(data_path):
    audio_waves = [] # Feature List
    labels = [] # Labels assuming 'Real' is class 0 and 'Fake' is class 1

    
    classes = ['Real', 'Fake']
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)

    # Read all files and mark their labels
    for class_name in classes:
        class_dir = os.path.join(data_path, class_name)
        for audio_file in os.listdir(class_dir):
            if audio_file.endswith('.wav'):  # Assuming audio files are in WAV format
                # Load audio file
                audio_path = os.path.join(class_dir, audio_file)
                waveform, sample_rate = librosa.load(audio_path, sr=None)

                # Extract features (e.g., using librosa)
                # Here we'll just use a simple feature: the mean of the waveform
                mean_waveform = np.mean(waveform)

                # Append features and labels
                audio_waves.append(mean_waveform)
                labels.append(class_name)

    
    labels = label_encoder.transform(labels)

    # Convert lists to numpy arrays
    audio_waves = np.array(audio_waves)
    labels = np.array(labels)

    return audio_waves, labels

class Load_CustomDatabackup(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_paths, self.labels = self.load_data()
        self.cut=64600 


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        features = self.loads_audio(file_path)
        return features, label

    def load_data(self):
        file_paths = []
        labels = []
        label_encoded = []
        for label, folder in enumerate(['Real', 'Fake']):
            folder_path = os.path.join(self.data_dir, folder)
            for file in os.listdir(folder_path):
                file_paths.append(os.path.join(folder_path, file))
                if folder == 'Real':
                    labels.append(1)
                else:
                    labels.append(0)
        for label in labels:
            one_hot_label = torch.zeros(2)  # 2 classes: Real and Fake
            one_hot_label[label] = 1
            label_encoded.append(one_hot_label)

        return file_paths, label_encoded
    
    def pad(self, x, max_len=64600):
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        # need to pad
        num_repeats = int(max_len / x_len)+1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
        return padded_x	
    
    def loads_audio(self, file_path):
        audio, sr = librosa.load(file_path,sr=16000)
        X_pad = self.pad(audio,self.cut)
        x_inp= torch.tensor(X_pad)
        return x_inp
    
class Load_FORData(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_paths, self.labels = self.load_data()
        self.cut=64600
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        features = self.loads_audio(file_path)
        return features, label

    def load_data(self):
        file_paths = []
        labels = []
        label_encoded = []
        for label, folder in enumerate(['real', 'fake']):
            folder_path = os.path.join(self.data_dir, folder)
            for file in os.listdir(folder_path):
                file_paths.append(os.path.join(folder_path, file))
                if folder == 'real':
                    labels.append(1)
                else:
                    labels.append(0)
        for label in labels:
            one_hot_label = torch.zeros(2)  # 2 classes: Real and Fake
            one_hot_label[label] = 1
            label_encoded.append(one_hot_label)
        return file_paths, label_encoded
    def pad(self, x, max_len=64600):
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        # need to pad
        num_repeats = int(max_len / x_len)+1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
        return padded_x	
    
    def loads_audio(self, file_path):
        audio, sr = librosa.load(file_path,sr=16000)
        X_pad = self.pad(audio,self.cut)
        x_inp= torch.tensor(X_pad)
        return x_inp
