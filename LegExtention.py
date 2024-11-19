import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

counter_left = 0
counter_right = 0
series_left = 0
series_right = 0
total_reps = 15  # 15 repeticiones por pierna
total_series = 3  # 3 series
rest_time = 5  # Tiempo de descanso en segundos entre series
resting = False  # Estado de descanso
rest_start_time = None
stage_left = "down"  # Estado inicial de la pierna izquierda
stage_right = "down"  # Estado inicial de la pierna derecha

cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    """Calcula el ángulo entre tres puntos (cadera, rodilla y tobillo)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def is_person_centered(landmarks, frame_width):
    """Verifica si la persona está centrada en la imagen"""
    left_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame_width
    right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame_width
    person_center_x = (left_hip_x + right_hip_x) / 2
    frame_center_x = frame_width / 2
    tolerance = frame_width * 0.15  # Margen de tolerancia para el centrado

    return abs(person_center_x - frame_center_x) < tolerance

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Convertimos la imagen a RGB para procesarla con Mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        # Convertimos la imagen de nuevo a BGR para mostrarla
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        frame_height, frame_width, _ = image.shape

        # Dibujamos los puntos y las conexiones de la pose
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Verificar si la persona está centrada en la cámara
            if is_person_centered(landmarks, frame_width):
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

                # Verificar si los puntos clave de las piernas son visibles
                if (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > 0.7 and
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > 0.7 and
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility > 0.7 and
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility > 0.7 and
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > 0.7 and
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility > 0.7):
                    
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    # Calcular ángulos de las rodillas
                    left_angle = calculate_angle(left_ankle, left_knee, left_hip)
                    right_angle = calculate_angle(right_ankle, right_knee, right_hip)

                    # Control de repeticiones y series con estados
                    if not resting:
                        # Verificar si solo una pierna está levantada a la vez
                        if left_angle > 150 and stage_left == "down":
                            stage_left = "up"
                        if left_angle < 130 and stage_left == "up":
                            stage_left = "down"
                            counter_left += 1
                            print(f"Repetición izquierda: {counter_left}")
                            if counter_left == total_reps:
                                series_left += 1
                                counter_left = 0
                                print(f"Serie izquierda completada: {series_left}")
                                if series_left < total_series:
                                    resting = True
                                    rest_start_time = time.time()

                        if right_angle > 150 and stage_right == "down":
                            stage_right = "up"
                        if right_angle < 130 and stage_right == "up":
                            stage_right = "down"
                            counter_right += 1
                            print(f"Repetición derecha: {counter_right}")
                            if counter_right == total_reps:
                                series_right += 1
                                counter_right = 0
                                print(f"Serie derecha completada: {series_right}")
                                if series_right < total_series:
                                    resting = True
                                    rest_start_time = time.time()

                    else:
                        # Contador de descanso
                        elapsed_time = time.time() - rest_start_time
                        if elapsed_time >= rest_time:
                            resting = False  # Finaliza el descanso
                        else:
                            cv2.putText(image, f'Descanso: {int(rest_time - elapsed_time)}s', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 50, 150), 2, cv2.LINE_AA)

                    # Mostrar los ángulos en la pantalla para referencia
                    cv2.putText(image, f'Angulo Izq: {int(left_angle)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 139), 2)
                    cv2.putText(image, f'Angulo Der: {int(right_angle)}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 139), 2)

            # Mostrar el conteo de repeticiones y series en pantalla
            cv2.putText(image, f'Reps Izq: {counter_left}/{total_reps}', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 139), 2)
            cv2.putText(image, f'Reps Der: {counter_right}/{total_reps}', (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 139), 2)
            cv2.putText(image, f'Series Izq: {series_left}/{total_series}', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 139), 2)
            cv2.putText(image, f'Series Der: {series_right}/{total_series}', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 139), 2)

        # Mostrar la imagen procesada
        cv2.imshow('MediaPipe', image)

        # Salir si se presiona 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()



