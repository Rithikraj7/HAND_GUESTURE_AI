# import tensorflow as tf
# import cv2
# from tensorflow.keras.models import load_model
# import numpy as np
# import webbrowser
# import mediapipe as mp
#
# # Initialize Mediapipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
#
# # Load the trained model
# model = load_model('hand_sign_model3.h5')
#
# # Mapping of class indices to labels
# hand_sign_labels = {
#     0: "one",
#     1: "two",
#     2: "three",
#     3: "four",
#     4: "five"
# }
#
# def preprocess_hand_sign_image(image):
#     # Convert the image to grayscale
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Reshape the image to a standard size of (224, 224) using interpolation
#     image = cv2.resize(image, (224, 224))
#
#     # Convert the grayscale image to a 3-channel image
#     image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#
#     # Normalize the image pixel values to a range between 0 and 1
#     image = image.astype('float32')
#     image /= 255.0
#
#     # Return the preprocessed image
#     return image
#
# # Capture real-time data from webcam
# cap = cv2.VideoCapture(0)
#
# # Variable to keep track of the previous predicted class
# prev_predicted_class = None
#
# # Variable to track whether YouTube has been opened
# youtube_opened = False
#
# google_opened = False
#
#
# def open_google():
#     global google_opened
#     if not google_opened:
#         url = "https://www.google.com"
#         webbrowser.open(url)
#         print("google opened successfully!")
#         google_opened = True
#     else:
#         print("google is already open.")
#
#
# def open_youtube():
#     global youtube_opened
#     if not youtube_opened:
#         url = "https://www.youtube.com"
#         webbrowser.open(url)
#         print("YouTube opened successfully!")
#         youtube_opened = True
#     else:
#         print("YouTube is already open.")
#
# def greet_user():
#     print("Hello there! What can I do for you today?")
#
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # Detect hand landmarks
#     results = hands.process(frame_rgb)
#     landmarks = results.multi_hand_landmarks
#     if landmarks:
#         # Draw landmarks and connect them with lines
#         for landmark in landmarks:
#             hand_points = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in landmark.landmark]
#             for i in range(len(hand_points)-1):
#                 cv2.circle(frame, hand_points[i], 5, (0, 255, 0), -1)
#                 cv2.line(frame, hand_points[i], hand_points[i+1], (0, 255, 0), 2)
#
#             cv2.circle(frame, hand_points[-1], 5, (0, 255, 0), -1)
#
#
#     # Preprocess the captured frame
#     preprocessed_frame = preprocess_hand_sign_image(frame)
#
#     # Predict the class probabilities of the hand sign
#     prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0))
#
#     # Get the predicted class index
#     predicted_class_index = np.argmax(prediction)
#
#     # Get the corresponding label
#     predicted_label = hand_sign_labels[predicted_class_index]
#
#     # Check if the predicted class has changed since the last prediction
#     if predicted_class_index != prev_predicted_class:
#         print(f"Detected sign: {predicted_label}, Confidence: {prediction[0][predicted_class_index]}")
#         prev_predicted_class = predicted_class_index
#
#         greet_user()
#
#         if "two" in predicted_label:
#             open_google()
#         elif "four" in predicted_label:
#             open_youtube()
#         else:
#             print("Command not recognized.")
#
#     # Display the captured frame
#     cv2.imshow('Real-time Hand Sign Recognition', frame)
#
#     # Check for 'q' key press to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything is done, release the capture
# cap.release()
# cv2.destroyAllWindows()

import cv2
from tensorflow.keras.models import load_model
import numpy as np
import webbrowser
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Load the trained model
model = load_model('hand_sign_model3.h5')

# Mapping of class indices to labels
hand_sign_labels = {
    0: "one",
    1: "two",
    2: "three",
    3: "four",
    4: "five"
}

# Variable to keep track of the previous predicted class
prev_predicted_class = None

# Variable to track whether YouTube has been opened
youtube_opened = False
google_opened = False

def preprocess_hand_sign_image(image):
    # Convert the image to RGB (MediaPipe Hands requires RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(image_rgb)
    landmarks = results.multi_hand_landmarks

    if landmarks:
        # Draw landmarks and connect them with lines
        for landmark in landmarks:
            hand_points = [(int(point.x * image.shape[1]), int(point.y * image.shape[0])) for point in landmark.landmark]
            for i in range(len(hand_points)-1):
                cv2.circle(image, hand_points[i], 5, (0, 255, 0), -1)
                cv2.line(image, hand_points[i], hand_points[i+1], (0, 255, 0), 2)

            cv2.circle(image, hand_points[-1], 5, (0, 255, 0), -1)

    # Resize the image to a standard size of (224, 224) using interpolation
    image = cv2.resize(image, (224, 224))

    # Normalize the image pixel values to a range between 0 and 1
    image = image.astype('float32')
    image /= 255.0


    # Return the preprocessed image
    return image

def open_google():
    global google_opened
    if not google_opened:
        url = "https://www.google.com"
        webbrowser.open(url)
        print("Google opened successfully!")
        google_opened = True
    else:
        print("Google is already open.")

def open_youtube():
    global youtube_opened
    if not youtube_opened:
        url = "https://www.youtube.com"
        webbrowser.open(url)
        print("YouTube opened successfully!")
        youtube_opened = True
    else:
        print("YouTube is already open.")

def greet_user():
    print("Hello there! What can I do for you today?")

# Capture real-time data from webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the captured frame
    preprocessed_frame = preprocess_hand_sign_image(frame)

    # Predict the class probabilities of the hand sign
    prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0))

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction)

    # Get the corresponding label
    predicted_label = hand_sign_labels[predicted_class_index]

    # Check if the predicted class has changed since the last prediction
    if predicted_class_index != prev_predicted_class:
        print(f"Detected sign: {predicted_label}, Confidence: {prediction[0][predicted_class_index]}")
        prev_predicted_class = predicted_class_index

        greet_user()

        if "two" in predicted_label:
            open_google()
        elif "four" in predicted_label:
            open_youtube()
        else:
            print("Command not recognized.")

    # Display the captured frame
    cv2.imshow('Real-time Hand Sign Recognition', frame)

    # Check for 'q' key press to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
