import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch
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
# Set the directory containing the TESS dataset
base_path = r'D:/Downloads/EMOTIONAL SPEECH DATASETS/TESS Toronto emotional speech set data/TESS Toronto emotional speech set data'
emotion_folders = ['OAF', 'YAF']  # Male and female folders
emotions = ['angry', 'happy', 'disgust', 'neutral', 'Pleasant_surprised', 'Sad', 'Fear']  # Emotion labels

# Create paths and labels for the dataset
paths = []
labels = []
for folder in emotion_folders:
    for emotion in emotions:
        emotion_folder = f"{folder}_{emotion}"
        emotion_path = os.path.join(base_path, emotion_folder)
        for filename in os.listdir(emotion_path):
            if filename.endswith('.wav'):
                paths.append(os.path.join(emotion_path, filename))
                labels.append(emotion)

# Create a DataFrame
df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels

# Function to extract MFCC features
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Extract features for each audio file
X = np.array([extract_mfcc(file) for file in df['speech']])
y = OneHotEncoder().fit_transform(df[['label']]).toarray()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Visualize MFCC distribution by emotion
def plot_mfcc_distribution(X, labels):
    plt.figure(figsize=(10, 6))
    for i, emotion in enumerate(emotions):
        plt.hist(X[labels == emotion], bins=30, alpha=0.5, label=emotion)
    plt.xlabel('MFCC Coefficients')
    plt.ylabel('Frequency')
    plt.title('MFCC Distribution by Emotion')
    plt.legend()
    plt.show()

# Plot MFCC distribution for the first 100 samples for visualization
sample_labels = np.array(df['label'])[:100]  # Adjust sample size if necessary
sample_mfcc = X[:100]
plot_mfcc_distribution(sample_mfcc, sample_labels)

# Load the trained model and evaluate on the test set
# Assuming the model is already defined and loaded
input_size = X.shape[1]  # Number of MFCC features (40 in this case)
num_classes = len(emotions)  # Number of emotions to classify

model_loaded = LSTMModel(input_size=input_size, num_classes=num_classes).to(device="cuda")
model_loaded.load_state_dict(torch.load('models/emotion_recognition_model.pth'))
model_loaded.eval()

# Perform prediction on the test set (You can uncomment this once you have the model)
with torch.no_grad():
    test_outputs = model_loaded(torch.tensor(X_test, dtype=torch.float32).view(X_test.shape[0], 1, X_test.shape[1]).to(device="cuda"))
    _, predicted = torch.max(test_outputs, 1)
    y_true = y_test.argmax(axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_true, predicted.cpu().numpy())
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print(classification_report(y_true, predicted.cpu().numpy(), target_names=emotions))

if __name__ == '__main__':
    print("Visualizations are generated successfully.")
