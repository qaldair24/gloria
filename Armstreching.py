import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

def check_arms_stretched(wrist, elbow, shoulder):
    if wrist[1] < elbow[1] < shoulder[1]:
        return True
    return False

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

start_time = None
holding = False
second_left = 10
repetitions = 0
series = 0
total_reps = 3 
total_series = 3  

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        try:
            x = check_arms_stretched(right_wrist, right_elbow, right_shoulder)
            y = check_arms_stretched(left_wrist, left_elbow, left_shoulder)

            right_angle = calculate_angle(right_wrist, right_elbow, right_shoulder)
            left_angle = calculate_angle(left_wrist, left_elbow, left_shoulder)

            if x and y and right_angle > 150 and left_angle > 150 and second_left > 0:
                cv2.putText(image, f'Hold it For {second_left} Sec', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                time.sleep(1)
                second_left -= 1
            elif second_left == 0:
                repetitions += 1
                second_left = 10  
                if repetitions == total_reps:
                    series += 1
                    repetitions = 0
                    cv2.putText(image, f'Series completed: {series}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if series == total_series:
                        cv2.putText(image, 'Workout Complete!', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('MediaPipe Pose', image)
                        cv2.waitKey(3000)
                        break

        except:
            pass
        
    
        cv2.putText(image, f'Rep: {repetitions}/{total_reps}', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f'Series: {series}/{total_series}', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(200, 100, 100), thickness=2, circle_radius=2)
        )
        cv2.imshow('MediaPipe ', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()



