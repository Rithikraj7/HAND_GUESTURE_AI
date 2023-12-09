# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# import mediapipe as mp
#
# model = load_model('hand_sign_model3.h5')
#
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
#
#
# cap = cv2.VideoCapture(0)
#
# try:
#     while True:
#         ret, frame = cap.read()
#
#         if not ret:
#             print("Can't retrieve frame - stream may have ended. Exiting..")
#             break
#
#         # Convert the frame to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # Use MediaPipe Hands to process the frame
#         results = hands.process(rgb_frame)
#
#         # Check if hand landmarks are detected
#         # if results.multi_hand_landmarks:
#         #     # Get the landmarks of the first hand
#         #     landmarks = results.multi_hand_landmarks[0].landmark
#         #
#         #     # # Preprocess the landmarks for prediction
#         #     # landmarks_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
#         #     # landmarks_array = landmarks_array.flatten()
#         #     # print("Landmarks array shape:", landmarks_array.shape)
#         #     # landmarks_array = landmarks_array.reshape((1, 224, 224, 3))
#         #     #
#         #     #
#         #     # # Make prediction using the loaded model
#         #     # # Make prediction using the loaded model
#         #     # prediction = model.predict(landmarks_array)
#         #     # predicted_class = np.argmax(prediction)
#         #
#         #     # Preprocess the landmarks for prediction
#         #     landmarks_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
#         #     landmarks_array = landmarks_array.flatten()
#         #     print("Landmarks array shape:", landmarks_array.shape)
#         #
#         #     # Reshape the landmarks into an image-like structure
#         #     landmarks_image = landmarks_array.reshape((21, 3))  # Assuming 21 landmarks (adjust if needed)
#         #     landmarks_image = landmarks_image[:, :2]  # Keep only x and y coordinates
#         #     landmarks_image = cv2.resize(landmarks_image, (224, 224))
#         #
#         #     # Expand dimensions to match model input shape and include the channel dimension
#         #     landmarks_image = np.expand_dims(landmarks_image, axis=-1)  # Add channel dimension
#         #     landmarks_image = np.repeat(landmarks_image, 3, axis=-1)     # Duplicate the single channel to create three channels
#         #     landmarks_image = np.expand_dims(landmarks_image, axis=0)     # Add batch dimension
#         #     landmarks_image = landmarks_image.astype(np.float32)          # Ensure data type is float32
#         #
#         #     # Make prediction using the loaded model
#         #     prediction = model.predict(landmarks_image)
#         #     predicted_class = np.argmax(prediction)
#         #
#         if results.multi_hand_landmarks:
#             landmarks = results.multi_hand_landmarks[0].landmark
#
#             # Preprocess the landmarks for prediction
#             landmarks_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
#             landmarks_array = landmarks_array.flatten()
#
#             # Create an RGB image with three channels
#             # Assuming landmarks_array is of size 63
#             landmarks_array = landmarks_array.reshape((21, 3))  # Reshape to (21, 3)
#
#             # Normalize the coordinates to the range [0, 1]
#             landmarks_array[:, :2] /= frame.shape[1], frame.shape[0]
#
#             # Convert landmarks to pixel coordinates
#             landmarks_array[:, :2] *= 224
#
#             # Create an empty image with three channels
#             hand_image = np.zeros((224, 224, 3), dtype=np.uint8)
#
#             # Draw landmarks on the image
#             for landmark in landmarks_array:
#                 x, y = int(landmark[0]), int(landmark[1])
#                 cv2.circle(hand_image, (x, y), 5, (0, 255, 0), -1)
#
#             # Expand dimensions to make it (1, 224, 224, 3)
#             hand_image = np.expand_dims(hand_image, axis=0)
#
#             # Make prediction using the loaded model
#             prediction = model.predict(hand_image)
#             predicted_class = np.argmax(prediction)
#
#
#             print(f"Predicted class: {predicted_class}, Confidence: {prediction[0][predicted_class]}")
#
#             # Display the predicted class on the frame
#             cv2.putText(frame, f"Predicted Class: {predicted_class}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)
#
#         # Display the frame
#         cv2.imshow("Hand Sign Recognition", frame)
#
#         # Exit the loop when 'q' or 'Esc' key is pressed
#         key = cv2.waitKey(1)
#         if key == 27 or key == ord('q'):
#             break
#
# except Exception as e:
#     print(f"An error occurred: {e}")
#
#
# finally:
#     cap.release()
#     cv2.destroyAllWindows()
#

#
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# import mediapipe as mp
#
# model = load_model('hand_sign_model3.h5')
#
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
#
# cap = cv2.VideoCapture(0)
#
# try:
#     while True:
#         ret, frame = cap.read()
#
#         if not ret:
#             print("Can't retrieve frame - stream may have ended. Exiting..")
#             break
#
#         # Convert the frame to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # Use MediaPipe Hands to process the frame
#         results = hands.process(rgb_frame)
#
#         # Check if hand landmarks are detected
#         if results.multi_hand_landmarks:
#             landmarks = results.multi_hand_landmarks[0].landmark
#
#             # Preprocess the landmarks for prediction
#             landmarks_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
#             landmarks_array = landmarks_array.flatten()
#
#             # Create an RGB image with three channels
#             landmarks_array = landmarks_array.reshape((21, 3))  # Reshape to (21, 3)
#
#             # Normalize the coordinates to the range [0, 1]
#             landmarks_array[:, :2] /= frame.shape[1], frame.shape[0]
#
#             # Convert landmarks to pixel coordinates
#             landmarks_array[:, :2] *= 224
#
#             # Create an empty image with three channels
#             hand_image = np.zeros((224, 224, 3), dtype=np.uint8)
#
#             # Draw landmarks on the image
#             for landmark in landmarks_array:
#                 x, y = int(landmark[0]), int(landmark[1])
#                 cv2.circle(hand_image, (x, y), 5, (0, 255, 0), -1)
#
#             # Expand dimensions to make it (1, 224, 224, 3)
#             hand_image = np.expand_dims(hand_image, axis=0)
#
#             # Make prediction using the loaded model
#             prediction = model.predict(hand_image)
#             predicted_class = np.argmax(prediction)
#
#             print(f"Predicted class: {predicted_class}, Confidence: {prediction[0][predicted_class]}")
#
#             # Display the predicted class on the frame
#             cv2.putText(frame, f"Predicted Class: {predicted_class}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#         # Display the frame
#         cv2.imshow("Hand Sign Recognition", frame)
#
#         # Exit the loop when 'q' or 'Esc' key is pressed
#         key = cv2.waitKey(1)
#         if key == 27 or key == ord('q'):
#             break
#
# except Exception as e:
#     print(f"An error occurred: {e}")
#
# finally:
#     cap.release()
#     cv2.destroyAllWindows()

#
# import tensorflow as tf
# import cv2
# from tensorflow.keras.models import load_model
# import numpy as np
#
# # Load the trained model
# model = load_model('hand_sign_model3.h5')
#
# # Mapping of class indices to labels
# hand_sign_labels = {
#     0: "one",
#     1: "two",
#     2: "Three",
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
#     image = image[:, :, np.newaxis] * np.ones((1, 1, 3), dtype=image.dtype)
#     image = image[np.newaxis, ...]
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
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     # Preprocess the captured frame
#     preprocessed_frame = preprocess_hand_sign_image(frame)
#
#     # Predict the class probabilities of the hand sign
#     prediction = model.predict(preprocessed_frame)
#
#     # Get the predicted class index
#     predicted_class_index = np.argmax(prediction)
#
#     # Get the corresponding label
#     predicted_label = hand_sign_labels[predicted_class_index]
#
#     print(f"Predicted label: {predicted_label}, Confidence: {prediction[0][predicted_class_index]}")
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

# import tensorflow as tf
# import cv2
# from tensorflow.keras.models import load_model
# import numpy as np
#
# # Load the trained model
# model = load_model('hand_sign_model3.h5')
#
# # Mapping of class indices to labels
# hand_sign_labels = {
#     0: "open",
#     1: "google",
#     2: "Calander",
#     3: "photo",
#     4: "youtube"
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
#     image = image[:, :, np.newaxis] * np.ones((1, 1, 3), dtype=image.dtype)
#     image = image[np.newaxis, ...]
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
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     # Preprocess the captured frame
#     preprocessed_frame = preprocess_hand_sign_image(frame)
#
#     # Predict the class probabilities of the hand sign
#     prediction = model.predict(preprocessed_frame)
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

import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import webbrowser


# Load the trained model
model = load_model('hand_sign_model3.h5')

# Mapping of class indices to labels
hand_sign_labels = {
    0: "calander",
    1: "youtube",
    2: "open",
    3: "photo",
    4: "google"
}

def preprocess_hand_sign_image(image):
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reshape the image to a standard size of (224, 224) using interpolation
    image = cv2.resize(image, (224, 224))

    # Convert the grayscale image to a 3-channel image
    image = image[:, :, np.newaxis] * np.ones((1, 1, 3), dtype=image.dtype)
    image = image[np.newaxis, ...]

    # Normalize the image pixel values to a range between 0 and 1
    image = image.astype('float32')
    image /= 255.0

    # Return the preprocessed image
    return image

# Capture real-time data from webcam
cap = cv2.VideoCapture(0)

# Variable to keep track of the previous predicted class
prev_predicted_class = None


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the captured frame
    preprocessed_frame = preprocess_hand_sign_image(frame)

    # Predict the class probabilities of the hand sign
    prediction = model.predict(preprocessed_frame)

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction)

    # Get the corresponding label
    predicted_label = hand_sign_labels[predicted_class_index]

    # Check if the predicted class has changed since the last prediction
    if predicted_class_index != prev_predicted_class:
        print(f"Detected sign: {predicted_label}, Confidence: {prediction[0][predicted_class_index]}")
        prev_predicted_class = predicted_class_index


    # Display the captured frame
    cv2.imshow('Real-time Hand Sign Recognition', frame)

    # Check for 'q' key press to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
