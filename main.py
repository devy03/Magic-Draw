import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Function to draw lines between points
def draw_lines(image, points):
    for i in range(len(points)-1):
        cv2.line(image, tuple(points[i]), tuple(points[i+1]), (255, 0, 0), 5)

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands
hands = mp_hands.Hands()

# Flag to track if 'q' key is pressed
draw_flag = False

# List to store finger tip positions
finger_tip_positions = []

while True:
    success, image = cap.read()

    # Flip the image horizontally
    image = cv2.flip(image, 1)

    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(image_rgb)

    # Convert RGB image back to BGR
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get index finger tip coordinates
            index_finger_tip = hand_landmarks.landmark[8].x * image.shape[1], hand_landmarks.landmark[8].y * image.shape[0]
            index_finger_tip = tuple(map(int, index_finger_tip))

            if draw_flag:
                # Append index finger tip position to the list
                finger_tip_positions.append(index_finger_tip)

                # Draw lines in blue color
                draw_lines(image, finger_tip_positions)

    cv2.imshow('MediaPipe Hands', image)
    key = cv2.waitKey(1)

    # Check if 'q' key is pressed
    if key == ord('q'):
        draw_flag = True
        finger_tip_positions = []  # Clear the list when 'q' is pressed
    elif key == ord('r'):
        draw_flag = False
    elif key == 27:  # Press Esc to exit
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
