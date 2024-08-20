import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir='./OutputFile'):
    X_ch1, X_ch2, y = [], [], []
    gesture_ids = {}
    dir = os.listdir(data_dir)

    for file_name in dir:
        if file_name.endswith('_data_ch1.npy'):
            parts = file_name.split('_')
            gesture_name = parts[1]
            train_time = int(parts[3].split('.')[0])

            if gesture_name not in gesture_ids:
                gesture_ids[gesture_name] = len(gesture_ids)

            gesture_id = gesture_ids[gesture_name]
            ch1_path = os.path.join(data_dir, file_name)
            # To read ch2 file
            ch2_path = os.path.join(data_dir, file_name.replace('_data_ch1.npy', '_data_ch2.npy'))

            if os.path.exists(ch1_path) and os.path.exists(ch2_path):
                data_ch1 = np.load(ch1_path)
                data_ch2 = np.load(ch2_path)
                # If ch1.shape!=ch2.shape, we'll discard it
                if data_ch1.shape == data_ch2.shape:
                    X_ch1.append(data_ch1)
                    X_ch2.append(data_ch2)
                    y.append(gesture_id)

    X_ch1 = np.array(X_ch1)
    X_ch2 = np.array(X_ch2)
    y = np.array(y)

    return X_ch1, X_ch2, y, gesture_ids


def preprocess_data(X_ch1, X_ch2):
    # Ensure the shape is the same to avoid error during training
    assert X_ch1.shape == X_ch2.shape, "The shape of ch1 and ch2 data must be the same."

    # Normalize the data to avoid overly depending on the features with larger range than others
    X_ch1 = X_ch1 / np.max(X_ch1)
    X_ch2 = X_ch2 / np.max(X_ch2)

    # Add a channel dimension and concatenate along the last axis
    X_ch1 = X_ch1[..., np.newaxis, np.newaxis]  # Shape: (5100, 12, 4096, 1, 1)
    X_ch2 = X_ch2[..., np.newaxis, np.newaxis]  # Shape: (5100, 12, 4096, 1, 1)

    X = np.concatenate((X_ch1, X_ch2), axis=-1)  # Shape: (5100, 12, 4096, 1, 2)

    print(f"Shape after concatenation: {X.shape}")

    return X


def create_model(input_shape, num_classes):
    model = Sequential([
        # 32 filters with a 3x3x3 kernel size (frames, height, width)
        # Padding is set to 'same' to ensure output dimensions match the input dimensions
        # L2 regularization is used to prevent overfitting
        Conv3D(32, (3, 3, 3), input_shape=input_shape, padding='same', kernel_regularizer=l2(0.01)),
        # MaxPooling operation reduces the spatial dimensions
        MaxPooling3D((2, 2, 2), padding='same'),
        # Dropout layer to avoid overfitting by dropping 40% of the units
        Dropout(0.1),

        # Second convolutional layer with 64 filters
        Conv3D(64, (3, 3, 3), padding='same', kernel_regularizer=l2(0.01)),
        MaxPooling3D((2, 2, 2), padding='same'),
        Dropout(0.1),

        # Third convolutional layer with 128 filters
        Conv3D(128, (3, 3, 3), padding='same', kernel_regularizer=l2(0.01)),
        MaxPooling3D((2, 2, 2), padding='same'),
        Dropout(0.1),

        # Flatten the 3D output to a 1D vector for the fully connected layer
        Flatten(),
        # Dense layer with 128 units and L2 regularization
        Dense(128, kernel_regularizer=l2(0.01)),
        Dropout(0.1),

        # Output layer with softmax activation for multi-class classification
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model with Adam optimizer and a learning rate of 0.00005
    # Sparse categorical crossentropy is used for multi-class classification
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



def save_model_and_metadata(model, gesture_ids, model_path='gesture_recognition_model.h5',
                            metadata_path='gesture_metadata.json'):
    model.save(model_path)

    with open(metadata_path, 'w') as f:
        json.dump(gesture_ids, f)

def main():
    data_dir = './OutputFile'

    X_ch1, X_ch2, y, gesture_ids = load_data(data_dir)

    if len(X_ch1) == 0:
        print("No data loaded. Please check your data directory and files.")
        return

    X = preprocess_data(X_ch1, X_ch2)
    print(f"Data shape after preprocessing: {X.shape}")  # Add this line to check the shape

    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = X_train.shape[1:]
    print(f"Input shape for the model: {input_shape}")  # Add this line to check the input shape

    num_classes = len(gesture_ids)

    model = create_model(input_shape, num_classes)

    batch_size = 16

    # Stop training after 15 epochs if no val_loss, and return back to the optimized model
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    steps_per_epoch = len(X_train) // batch_size

    model.fit(X_train, y_train,
              epochs=200,
              steps_per_epoch=steps_per_epoch,
              validation_data=(X_test, y_test),
              callbacks=[early_stopping])

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

    save_model_and_metadata(model, gesture_ids)
    print("Model and metadata saved successfully.")

    print("Gesture categories:")
    for gesture_name, gesture_id in gesture_ids.items():
        print(f"{gesture_name}: {gesture_id}")

if __name__ == "__main__":
    main()

