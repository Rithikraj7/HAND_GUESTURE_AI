# import cv2
# import mediapipe as mp
# import os
# import numpy as np
# from tensorflow import timestamp
#
#
# # Create a directory to store the dataset
#
# dataset_dir = "dataset/datasets"
# if not os.path.exists(dataset_dir):
#     os.makedirs(dataset_dir)
#
# # Initialize Mediapipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
#
# # Video capture
# cap = cv2.VideoCapture(0)
#
# # Map for hand signs and corresponding labels
# hand_signs = {
#     1: "One",
#     2: "Two",
#     3: "Three",
#     4: "Four",
#     5: "Five"
# }
#
# for sign_id, sign_name in hand_signs.items():
#     print(f"Collecting data for {sign_name}...")
#
#     sign_dir = os.path.join(dataset_dir, sign_name)
#     if not os.path.exists(sign_dir):
#         os.makedirs(sign_dir)
#
#     img_count = 0
#
#     while img_count < 100:
#         _, frame = cap.read()
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # Detect hand landmarks
#         results = hands.process(frame_rgb)
#         landmarks = results.multi_hand_landmarks
#
#         if landmarks:
#             # Draw landmarks and connect them with lines
#             for landmark in landmarks:
#                 hand_points = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in landmark.landmark]
#                 for i in range(len(hand_points)-1):
#                     cv2.circle(frame, hand_points[i], 5, (0, 255, 0), -1)
#                     cv2.line(frame, hand_points[i], hand_points[i+1], (0, 255, 0), 2)
#
#                 cv2.circle(frame, hand_points[-1], 5, (0, 255, 0), -1)
#
#             # Save the hand sign image
#             img_count += 1
#             sign_image = cv2.resize(frame, (224, 224))
#             # sign_image = cv2.resize(frame, (224, 224))[...,::-1]
#             # sign_image = np.expand_dims(sign_image, axis=0)
#             file_name_path = os.path.join(sign_dir, f"{sign_name}_{img_count}.jpg")
#             # file_name_path = os.path.join(sign_dir, f"{sign_name}_{img_count}_{timestamp}.jpg")
#             cv2.imwrite(file_name_path, sign_image)
#
#             cv2.putText(frame, f"Collecting {img_count}/100 for {sign_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                         (0, 0, 255))
#
#         if frame is not None:
#             cv2.imshow('Hand Sign Capture', frame)
#             k = cv2.waitKey(10)
#             if k == 27:
#                 break
#         else:
#             print("Error: Unable to capture frame.")
#             break
#
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import os
# import mediapipe as mp
#
# # Set the label for which you want to capture images
# label = input("enter the name")
#
# # Create a directory to store the captured images
# capture_dir = f"dataset/capture/{label}"
# if not os.path.exists(capture_dir):
#     os.makedirs(capture_dir)
#
# # Initialize Mediapipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
#
# # Initialize the webcam
# cap = cv2.VideoCapture(0)
#
# # Counter for captured images
# img_count = 0
#
# while img_count < 100:
#     # Read a frame from the webcam
#     ret, frame = cap.read()
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Detect hand landmarks
#     results = hands.process(frame_rgb)
#     landmarks = results.multi_hand_landmarks
#
#     if landmarks:
#         # Draw landmarks and connect them with lines
#         for landmark in landmarks:
#             hand_points = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in landmark.landmark]
#             for i in range(len(hand_points) - 1):
#                 cv2.circle(frame, hand_points[i], 5, (0, 255, 0), -1)
#                 cv2.line(frame, hand_points[i], hand_points[i + 1], (0, 255, 0), 2)
#
#             cv2.circle(frame, hand_points[-1], 5, (0, 255, 0), -1)
#
#         # Save the captured image
#         img_count += 1
#         img_path = os.path.join(capture_dir, f"{label}_{img_count}.jpg")
#         cv2.imwrite(img_path, frame)
#
#         # Display the capture count
#         print(f"Captured {img_count}/100 images for label {label}")
#
#     # Display the frame
#     cv2.imshow('Capture Images', frame)
#
#     # Wait for a key press to capture the next image
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the webcam and close the OpenCV windows
# cap.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import os
import numpy as np
import time

# Create a directory to store the dataset
dataset_dir = "../dataset/datasets"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Video capture
cap = cv2.VideoCapture(0)

# Map for hand signs and corresponding labels
hand_signs = {
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five"
}

for sign_id, sign_name in hand_signs.items():
    print(f"Collecting data for {sign_name}...")

    sign_dir = os.path.join(dataset_dir, sign_name)
    if not os.path.exists(sign_dir):
        os.makedirs(sign_dir)

    img_count = 0

    while img_count < 100:
        _, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        results = hands.process(frame_rgb)
        landmarks = results.multi_hand_landmarks

        if landmarks:
            # Draw landmarks and connect them with lines
            for landmark in landmarks:
                hand_points = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in landmark.landmark]
                for i in range(len(hand_points)-1):
                    cv2.circle(frame, hand_points[i], 5, (0, 255, 0), -1)
                    cv2.line(frame, hand_points[i], hand_points[i+1], (0, 255, 0), 2)

                cv2.circle(frame, hand_points[-1], 5, (0, 255, 0), -1)

            # Save the hand sign image
            img_count += 1
            sign_image = cv2.resize(frame, (224, 224))
            file_name_path = os.path.join(sign_dir, f"{sign_name}_{img_count}.jpg")
            cv2.imwrite(file_name_path, sign_image)

            cv2.putText(frame, f"Collecting {img_count}/100 for {sign_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255))

        if frame is not None:
            cv2.imshow('Hand Sign Capture', frame)
            k = cv2.waitKey(10)
            if k == 27:
                break

    # Add a delay of 5 seconds before collecting data for the next label
    time.sleep(5)

cap.release()
cv2.destroyAllWindows()
