from concurrent.futures import ProcessPoolExecutor
import os
import pandas as pd
import speech_recognition as sr
from tqdm import tqdm

CREMA_PATH = "D:\\Downloads\\EMOTIONAL SPEECH DATASETS\\CREMA-Ddataset"

def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"API error: {e}")
        return None

def process_audio_file(file_path):
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    actor_id, sentence, emotion_code, emotion_level = parts[0], parts[1], parts[2], parts[3].replace('.wav', '')
    emotion_mapping = {"ANG": "Anger", "DIS": "Disgust", "FEA": "Fear", "HAP": "Happy", "NEU": "Neutral", "SAD": "Sad"}
    emotion = emotion_mapping.get(emotion_code, "Unknown")
    text = audio_to_text(file_path)
    return {'actor_id': actor_id, 'sentence': sentence, 'emotion': emotion, 'emotion_level': emotion_level, 'text': text} if text else None

def load_crema_dataset(CREMA_PATH):
    audio_files = [os.path.join(CREMA_PATH, f) for f in os.listdir(CREMA_PATH) if f.endswith('.wav')]
    data = []
    with ProcessPoolExecutor(max_workers=2) as executor:  # Control the number of workers
        results = list(tqdm(executor.map(process_audio_file, audio_files), total=len(audio_files), desc="Processing Files"))
    data = [result for result in results if result is not None]
    return pd.DataFrame(data)

def run_preprocessing():
    crema_df = load_crema_dataset(CREMA_PATH)
    crema_df.to_csv("crema_data.csv", index=False)
    print("Dataset saved as crema_data.csv")

if __name__ == "__main__":
    run_preprocessing()
