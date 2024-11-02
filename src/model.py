import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm  # Import tqdm for progress bars

# Set the directory containing the TESS dataset
base_path = r'D:/Downloads/EMOTIONAL SPEECH DATASETS/TESS Toronto emotional speech set data/TESS Toronto emotional speech set data'
emotion_folders = ['OAF', 'YAF']  # male and female folders
emotions = ['angry', 'happy', 'disgust', 'neutral', 'Pleasant_surprised', 'Sad', 'Fear']  # emotion labels

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
print('Dataset is Loaded')
print(df.head())

# Function to extract MFCC features
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Extract features for each audio file
X = np.array([extract_mfcc(file) for file in df['speech']])
y = OneHotEncoder().fit_transform(df[['label']]).toarray()

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Reshape X for LSTM input: (samples, sequence_length, features)
X = X.view(X.shape[0], 1, X.shape[1])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
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

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size=40, num_classes=y.shape[1]).to(device)  # Use 40 as input_size

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize lists to store metrics
train_accuracies = []
val_accuracies = []
val_losses = []

# Fit the model with tqdm progress bar
# Fit the model with tqdm progress bar
epochs = 50
for epoch in tqdm(range(epochs), desc='Training Progress'):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train.to(device))
    loss = criterion(outputs, y_train.argmax(dim=1).to(device))  # Use argmax to get class labels
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test.to(device))
        val_loss = criterion(val_outputs, y_test.argmax(dim=1).to(device))
        _, predicted = torch.max(val_outputs, 1)
        val_accuracy = (predicted == y_test.argmax(dim=1).to(device)).float().mean().item()
    
    # Store metrics
    train_accuracies.append((outputs.argmax(dim=1) == y_train.argmax(dim=1).to(device)).float().mean().item())
    val_accuracies.append(val_accuracy)
    val_losses.append(val_loss.item())
    
    # Print metrics
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}')

# Calculate average metrics
avg_train_accuracy = np.mean(train_accuracies)
avg_val_accuracy = np.mean(val_accuracies)
avg_val_loss = np.mean(val_losses)

print(f"\nAverage Training Accuracy: {avg_train_accuracy:.4f}")
print(f"Average Validation Accuracy: {avg_val_accuracy:.4f}")
print(f"Average Validation Loss: {avg_val_loss:.4f}")

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test.to(device))
    _, predicted = torch.max(test_outputs, 1)
    y_true = y_test.argmax(dim=1)

# Generate a classification report
report = classification_report(y_true.numpy(), predicted.cpu().numpy(), target_names=OneHotEncoder().fit(df[['label']]).categories_[0])
print(report)

# After training, save the model
torch.save(model.state_dict(), 'emotion_recognition_model.pth')  # Save model
print("Model saved successfully!")

# Loading the model
model_loaded = LSTMModel(input_size=40, num_classes=y.shape[1]).to(device)
model_loaded.load_state_dict(torch.load('emotion_recognition_model.pth'))
model_loaded.eval()  # Set the model to evaluation mode
print("Model loaded successfully!")
