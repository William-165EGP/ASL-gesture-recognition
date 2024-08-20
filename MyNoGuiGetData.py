import csv
import os.path
import shutil
import sys

import keyboard
import numpy as np

from KKT_Module.ksoc_global import kgl
from KKT_Module.Configs import SettingConfigs
from KKT_Module.SettingProcess.SettingProccess import SettingProc, ConnectDevice, ResetDevice
from KKT_Module.DataReceive.DataReciever import RawDataReceiver, HWResultReceiver, FeatureMapReceiver
import time

def connect():
    connect = ConnectDevice()
    connect.startUp()                       # Connect to the device
    reset = ResetDevice()
    reset.startUp()                         # Reset hardware register

def startSetting():
    SettingConfigs.setScriptDir("K60168-Test-00256-008-v0.0.8-20230717_480cm")  # Set the setting folder name
    ksp = SettingProc()                 # Object for setting process to setup the Hardware AI and RF before receive data
    ksp.startUp(SettingConfigs)             # Start the setting process
    # ksp.startSetting(SettingConfigs)        # Start the setting process in sub_thread

def save_data(gesture_id, train_time, data_ch1, data_ch2):
    Dirpath = os.path.normpath(os.path.join(os.getcwd(), './OutputFile'))
    if not os.path.isdir(Dirpath):
        os.mkdir(Dirpath)

    np.save(os.path.join(Dirpath, f'gesture_{gesture_id}_train_{train_time}_data_ch1.npy'), data_ch1)
    np.save(os.path.join(Dirpath, f'gesture_{gesture_id}_train_{train_time}_data_ch2.npy'), data_ch2)


def startLoop(name_gestures, frameCount=20, trainTimes=300):
    # kgl.ksoclib.switchLogMode(True)
    R = RawDataReceiver(chirps=32)

    # Receiver for getting Raw data
    # R = FeatureMapReceiver(chirps=32)  # Receiver for getting RDI PHD map
    # R = HWResultReceiver()                  # Receiver for getting hardware results (gestures, Axes, exponential)
    # buffer = DataBuffer(100)                # Buffer for saving latest frames of data
    R.trigger(chirps=32)  # Trigger receiver before getting the data
    time.sleep(0.5)
    print('# ======== Ready to record gestures ===========')

    Dirpath = os.path.normpath(os.path.join(os.getcwd(), './OutputFile'))
    if not os.path.isdir(Dirpath):
        os.mkdir(Dirpath)


    for gesture_id in name_gestures:
        for train_time in range(trainTimes):
            start_train_time = 0
            new_train_time = start_train_time + train_time
            print(f'\nPreparing to record gesture {gesture_id}, training {new_train_time + 1}/{trainTimes}. Press "Space" to start recording.')

            # Waiting for user pressing Space
            while not keyboard.is_pressed('Space'):
                time.sleep(0.1)

            print(f' Starting to record gesture {gesture_id}, training {new_train_time + 1}/{trainTimes}.')
            gesture_data_ch1 = []
            gesture_data_ch2 = []

            for frame in range(frameCount): # read in 20 frames
                while True:
                    res = R.getResults()
                    if res is not None and len(res) > 1 and res[0] is not None and res[1] is not None:
                        break
                    print(f'Failed to get data for gesture {gesture_id}, Train {new_train_time}, Frame {frame}. Retrying...')
                    time.sleep(0.05)

                print(f'Gesture {gesture_id}, Train {new_train_time}, Frame {frame}, data', res)


                if frame < 8: # discard 0.4s
                    time.sleep(0.05)
                    continue
                #print(res)
                gesture_data_ch1.append(res[0])
                gesture_data_ch2.append(res[1])

                # Updating the progress bar
                progress = (frame + 1) / frameCount
                progress_bar = '[' + '#' * int(progress * 50) + ' ' * (50 - int(progress * 50)) + ']'
                sys.stdout.write(f'\rProgress: {progress_bar} {progress * 100:.2f}%')
                sys.stdout.flush()

                time.sleep(0.05)

            gesture_data_ch1 = np.array(gesture_data_ch1)
            gesture_data_ch2 = np.array(gesture_data_ch2)

            save_data(gesture_id, new_train_time, gesture_data_ch1, gesture_data_ch2)




def main():
    name_gestures = ['(period)']

    kgl.setLib()

    # kgl.ksoclib.switchLogMode(True)

    connect()                               # First you have to connect to the device

    startSetting()                         # Second you have to set the setting configs

    startLoop(name_gestures=name_gestures)                             # Last you can continue to get the data in the loop

if __name__ == '__main__':
    main()
