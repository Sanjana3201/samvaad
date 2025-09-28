import cv2
import tensorflow as tf
import numpy as np
import pyttsx3
import os

# Step 1: Load the trained CNN model and class labels
model = tf.keras.models.load_model('word_model.h5')

# The class labels are the names of the folders in your prepared data
prepared_data_path = 'prepared_frames'
class_labels = sorted(os.listdir(prepared_data_path))

# Step 2: Initialize the webcam and TTS engine
cap = cv2.VideoCapture(0)
engine = pyttsx3.init()

print("Word translator is active. Show a sign for a word.")
last_predicted_word = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a more natural feel
    frame = cv2.flip(frame, 1)

    # Pre-process the frame
    img = cv2.resize(frame, (64, 64))
    img = np.expand_dims(img, axis=0) / 255.0

    # Make a prediction
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    predicted_word = class_labels[predicted_class_index]

    # Display the predicted word on the screen
    cv2.putText(frame, predicted_word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Speak the word if it's a new prediction
    if predicted_word != last_predicted_word:
        print(f"Predicted: {predicted_word}")
        engine.say(predicted_word)
        engine.runAndWait()
        last_predicted_word = predicted_word

    cv2.imshow('CNN Word Translator', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()