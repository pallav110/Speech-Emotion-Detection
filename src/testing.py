import os
import numpy as np
import torch
import librosa
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 256, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Get the last time step
        out = self.dropout1(out)
        out = nn.ReLU()(self.fc1(out))
        out = self.dropout2(out)
        out = nn.ReLU()(self.fc2(out))
        out = self.dropout3(out)
        out = self.fc3(out)
        return out
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_loaded = LSTMModel(input_size=40, num_classes=7).to(device)  # Update num_classes based on your labels
model_loaded.load_state_dict(torch.load('models/emotion_recognition_model.pth'))
model_loaded.eval()  # Set the model to evaluation mode
print("Model loaded successfully!")

# Define emotion labels based on your training
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Pleasant Surprise",
    6: "Sad"
}

# Function to extract MFCC features from an audio file
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Path to your test audio files
test_audio_files = [
    'audio_files/03-01-01-01-02-02-11.wav',
    'audio_files/03-01-07-01-02-01-11.wav',
    'audio_files/03-01-03-01-01-02-20.wav',
    'audio_files/test_1.wav',
    'audio_files/test_audio.wav',
    'audio_files/recorded_audio.wav',\
    "D:\Downloads\EMOTIONAL SPEECH DATASETS\TESS Toronto emotional speech set data\TESS Toronto emotional speech set data\YAF_disgust\YAF_boat_disgust.wav"
]

# Process each audio file and make predictions
for audio_path in test_audio_files:
    mfcc_features = extract_mfcc(audio_path)
    
    # Prepare input tensor
    input_tensor = torch.tensor(mfcc_features, dtype=torch.float32).view(1, 1, -1).to(device)  # Reshape for LSTM
    
    # Make prediction
    with torch.no_grad():
        output = model_loaded(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    
    # Map predicted class to emotion label
    emotion = emotion_labels.get(predicted_class, "Unknown Emotion")
    print(f'Audio file: {audio_path} - Predicted Emotion: {emotion}')
