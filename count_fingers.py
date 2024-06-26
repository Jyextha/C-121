import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

tipIds = [4, 8, 12, 16, 20]

# Define a function to count fingers
def countFingers(image, hand_landmarks, handNo=0):
    fingers = []

    if hand_landmarks:
        landmarks = hand_landmarks[handNo].landmark
        for lm_index in tipIds:
            finger_tip_y = landmarks[lm_index].y
            finger_bottom_y = landmarks[lm_index-2].y

            if finger_tip_y < finger_bottom_y:
                fingers.append(1)  # Finger is open
            else:
                fingers.append(0)  # Finger is closed

        totalFingers = fingers.count(1)
        # Display Text
        text = f'Fingers: {totalFingers}'
        cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Define a function to draw hand landmarks
def drawHandLandmarks(image, hand_landmarks):
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

while True:
    success, image = cap.read()

    image = cv2.flip(image, 1)

    # Detect hand landmarks
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    hand_landmarks = results.multi_hand_landmarks

    # Draw landmarks and count fingers
    drawHandLandmarks(image, hand_landmarks)
    countFingers(image, hand_landmarks)

    cv2.imshow("Media Controller", image)

    # Exit the loop on pressing Spacebar
    key = cv2.waitKey(1)
    if key == 32:
        break

cv2.destroyAllWindows()
cap.release()