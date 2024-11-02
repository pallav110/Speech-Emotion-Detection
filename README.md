# Emotion Recognition in Speech Using LSTM

This project implements an emotion recognition system that classifies audio recordings of speech into different emotional categories using a Long Short-Term Memory (LSTM) neural network. The model is trained on the Toronto Emotional Speech Set (TESS) dataset, which contains various emotions expressed in speech.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Usage](#usage)
- [Installation](#installation)
- [License](#license)

## Project Overview

The goal of this project is to build a robust emotion recognition system that can accurately classify audio samples into different emotional categories. The model utilizes LSTM architecture to capture temporal dependencies in the audio data, leveraging Mel-frequency cepstral coefficients (MFCCs) as features.

## Dataset

The dataset used for training and testing is the **Toronto Emotional Speech Set (TESS)**, which includes recordings of various emotions spoken by male and female actors. The emotions included are:

- Angry
- Happy
- Disgust
- Neutral
- Pleasant Surprise
- Sad
- Fear

The audio files are organized in folders based on gender and emotion.

## Features

- **MFCC Extraction**: Extracts MFCC features from audio files for model input.
- **LSTM Model**: Implements a multi-layer LSTM for sequential data processing.
- **Data Visualization**: Visualizes the distribution of MFCC features and confusion matrix of predictions.
- **High Accuracy**: Achieved up to 99% accuracy on validation data.

## Model Architecture

The LSTM model consists of:

- An LSTM layer with 256 units.
- Several fully connected layers with dropout for regularization.
- A final output layer using softmax activation for multi-class classification.

### Sample Code

```python
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
        out = out[:, -1, :]
        out = self.dropout1(out)
        out = nn.ReLU()(self.fc1(out))
        out = self.dropout2(out)
        out = nn.ReLU()(self.fc2(out))
        out = self.dropout3(out)
        out = self.fc3(out)
        return out
