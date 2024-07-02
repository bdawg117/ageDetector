#!/home/bdawg/anaconda3/envs/aik/bin/python

import tensorflow as tf
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os
import warnings


def load_audio_data(folder, file_formats=('wav', 'flac', 'mp3'), max_length=None):
    data = []
    labels = []
    sampling_rates = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            file_format = file.split('.')[-1]
            if file_format in file_formats:
                file_path = os.path.join(root, file)
                label = os.path.basename(root)
                labels.append(label)

                # Convert MP3 to WAV if needed
                if file_format == 'mp3':
                    wav_path = file_path[:-3] + 'wav'
                    mp3_to_wav(file_path, wav_path)
                    file_path = wav_path

                try:
                    # Try loading with librosa
                    audio_data, sr = librosa.load(file_path, sr=None)
                except Exception as e:
                    # Handle the exception and print a warning
                    warnings.warn(f"Librosa failed to load {file_path}. Trying with audioread. Error: {e}")
                    # Try loading with audioread as a fallback
                    with audioread.audio_open(file_path) as input_file:
                        audio_data = input_file.read(dtype='float32')
                        sr = input_file.samplerate

                # Apply max_length if specified
                if max_length:
                    audio_data = np.pad(audio_data[:max_length], (0, max_length - len(audio_data[:max_length])))

                data.append((audio_data, sr))
                sampling_rates.append(sr)

    return data, np.array(labels), np.array(sampling_rates)


# Define paths to your data folders
adults_folder = r'C:\Users\brima\OneDrive\Desktop\HCI\en\clips_wav'
children_folder = r"C:\Users\brima\OneDrive\Desktop\HCI\sam_child"

max_length = 3 * 44100

# Load audio data for adults and children
adults_data, adults_labels, adults_sampling_rates = load_audio_data(adults_folder, max_length=max_length)
children_data, children_labels, children_sampling_rates = load_audio_data(children_folder, max_length=max_length)
print(len(adults_data), len(children_data))

# Convert lists to NumPy arrays
adults_data = np.array([audio_data for audio_data, _ in adults_data])
children_data = np.array([audio_data for audio_data, _ in children_data])

# Extract MFCC features
num_mfcc_coefficients = 13
adults_mfcc = np.array([librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=num_mfcc_coefficients) for audio_data, sr in zip(adults_data, adults_sampling_rates)])
children_mfcc = np.array([librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=num_mfcc_coefficients) for audio_data, sr in zip(children_data, children_sampling_rates)])

# Ensure that the features have the same number of dimensions
#adults_features = np.vstack((adults_mfcc.reshape(-1, num_mfcc_coefficients)))
#children_features = np.vstack((children_mfcc.reshape(-1, num_mfcc_coefficients)))

#children_features = np.vstack((children_mfcc[:, :num_mfcc_coefficients]))
# Check if arrays are not empty before stacking
if adults_mfcc.size > 0:
    adults_features = np.vstack((adults_mfcc.reshape(-1, num_mfcc_coefficients)))
else:
    adults_features = np.array([])

if children_mfcc.size > 0:
    children_features = np.vstack((children_mfcc.reshape(-1, num_mfcc_coefficients)))
else:
    children_features = np.array([])



# Concatenate the features
X = np.concatenate([adults_features, children_features], axis=0)
y = np.concatenate([np.zeros(len(adults_features)), np.ones(len(children_features))], axis=0)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model
model = Sequential([
    Flatten(input_shape=(X_train_scaled.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:\n', classification_rep)

# Save the model
model_filename_tf = 'audio_classification_model_tf.h5'
model.save(model_filename_tf)

# Save the scaler
scaler_filename_tf = 'scaler_tf.joblib'
joblib.dump(scaler, scaler_filename_tf)

# Save training information
training_info_tf = {'num_features': X_train_scaled.shape[1]}
joblib.dump(training_info_tf, 'training_info_tf.joblib')

print(f"TensorFlow Model saved to {model_filename_tf}, Scaler saved to {scaler_filename_tf}")
