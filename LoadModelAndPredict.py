import json
import sys
import time
import numpy as np
import tensorflow as tf
import keyboard

from KKT_Module.ksoc_global import kgl
from KKT_Module.Configs import SettingConfigs
from KKT_Module.SettingProcess.SettingProccess import SettingProc, ConnectDevice, ResetDevice
from KKT_Module.DataReceive.DataReciever import RawDataReceiver

import requests

import os
import pygame
from gtts import gTTS

def speak(sentence, lang='en'):

    pygame.mixer.init()

    tts = gTTS(text=sentence, lang=lang)
    filename = "temp.mp3"
    tts.save(filename)

    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.music.stop()
    pygame.mixer.quit()

    os.remove(filename)


def call_translator(prompt):
    # Get an openAI api key by yourself
    api_key = ''

    if prompt != '':
        model = 'gpt-3.5-turbo'
        system_content = (
            'return correct sentence and fix grammer do not change the meaning, (period) means the punctuation.'
            'If only one word just simply return')

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        data = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': system_content},
                {'role': 'user', 'content': prompt}
            ]
        }

        try:
            response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)

            if response.status_code == 200:
                result = response.json()
                translated_text = result['choices'][0]['message']['content'].strip()
                print(f"Translated text: {translated_text}")  # Debugging info
                return translated_text
            else:
                error_message = f"Request failed with error code: {response.status_code}\n{response.text}"
                print(error_message)  # Debugging info
                return error_message
        except Exception as e:
            error_message = f"An error occurred: {e}"
            print(error_message)  # Debugging info
            return error_message

    return prompt


def connect():
    connect = ConnectDevice()
    connect.startUp()                       # Connect to the device
    reset = ResetDevice()
    reset.startUp()                         # Reset hardware register

def startSetting():
    SettingConfigs.setScriptDir("K60168-Test-00256-008-v0.0.8-20230717_480cm")  # Set the setting folder name
    ksp = SettingProc()                 # Object for setting process to setup the Hardware AI and RF before receive data
    ksp.startUp(SettingConfigs)             # Start the setting process

def load_model_and_metadata(model_path='gesture_recognition_model.h5', metadata_path='gesture_metadata.json'):
    # Ensure the model exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Loading the model
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    # Ensure the json file exists
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Using utf-8 to read it
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            gesture_ids = json.load(f)
        print(f"Metadata loaded successfully from {metadata_path}")
    except json.JSONDecodeError as json_error:
        raise RuntimeError(f"JSON decode error: {json_error}")
    except Exception as e:
        raise RuntimeError(f"Error reading metadata file: {e}")

    id_to_gesture = {v: k for k, v in gesture_ids.items()}
    return model, id_to_gesture

def preprocess_data_for_model(gesture_data_ch1, gesture_data_ch2):
    # Normalize data
    gesture_data_ch1 = gesture_data_ch1 / np.max(gesture_data_ch1)
    gesture_data_ch2 = gesture_data_ch2 / np.max(gesture_data_ch2)

    gesture_data_ch1 = gesture_data_ch1[..., np.newaxis, np.newaxis]
    gesture_data_ch2 = gesture_data_ch2[..., np.newaxis, np.newaxis]

    X = np.concatenate((gesture_data_ch1, gesture_data_ch2), axis=-1)

    X = np.expand_dims(X, axis=0)

    return X

def startLoop(model, id_to_gesture, frameCount=20):
    R = RawDataReceiver(chirps=32)
    R.trigger(chirps=32)  # Trigger receiver before getting the data
    time.sleep(0.5)
    print('# ======== Ready to recognize gestures ===========')

    my_sentence = ''
    start_train_time = 0
    while True:

        gesture_data_ch1 = []
        gesture_data_ch2 = []

        print('\nPress "Space" to start recording a gesture.')

        # Waiting for pressing Space
        while not keyboard.is_pressed('Space'):
            time.sleep(0.1)

        print('Recording gesture...')

        for frame in range(frameCount):
            while True:
                res = R.getResults()
                if res is not None and len(res) > 1 and res[0] is not None and res[1] is not None:
                    break
                print(f'Failed to get data. Retrying...')
                time.sleep(0.05)

            if frame < 8:
                time.sleep(0.05)
                continue

            gesture_data_ch1.append(res[0])
            gesture_data_ch2.append(res[1])


            progress = (frame + 1) / frameCount
            progress_bar = '[' + '#' * int(progress * 50) + ' ' * (50 - int(progress * 50)) + ']'
            #sys.stdout.write(f'\rProgress: {progress_bar} {progress * 100:.2f}%')
            #sys.stdout.flush()

            time.sleep(0.05)

        gesture_data_ch1 = np.array(gesture_data_ch1)
        gesture_data_ch2 = np.array(gesture_data_ch2)

        X = preprocess_data_for_model(gesture_data_ch1, gesture_data_ch2)

        prediction = model.predict(X)
        predicted_gesture_id = np.argmax(prediction, axis=1)[0]
        predicted_gesture = id_to_gesture[predicted_gesture_id]



        print(f'\nPredicted Gesture: {predicted_gesture}')

        my_sentence += ' ' + predicted_gesture
        if predicted_gesture == '(period)':
            new_sentence = call_translator(my_sentence)
            speak(new_sentence)
            my_sentence = ''
            time.sleep(0.5)




def main():
    kgl.setLib()

    connect()                               # First you have to connect to the device

    startSetting()                         # Second you have to set the setting configs

    try:
        model, id_to_gesture = load_model_and_metadata()  # Load model and metadata
        startLoop(model, id_to_gesture)        # Start the loop to recognize gestures

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
