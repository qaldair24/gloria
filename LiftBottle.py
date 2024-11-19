import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

counter = 0
series = 0
reps_per_series = 12
total_series = 3
stage = None  # Estado inicial

cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    """Calcula el ángulo entre tres puntos (muñeca, codo, hombro)"""
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
    left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame_width
    right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame_width
    person_center_x = (left_shoulder_x + right_shoulder_x) / 2
    frame_center_x = frame_width / 2
    tolerance = frame_width * 0.15  # Margen de tolerancia para el centrado

    return abs(person_center_x - frame_center_x) < tolerance

# Lógica principal
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Convertimos la imagen a RGB para procesarla con Mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Procesar la imagen con Mediapipe
        results = pose.process(image)

        # Convertimos la imagen de nuevo a BGR para mostrarla
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Dibujar los puntos clave y conexiones del pose
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Verificar si la persona está centrada
            if is_person_centered(landmarks, frame.shape[1]):
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

                try:
                    # Obtener coordenadas de hombro, codo y muñeca
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    # Calcular ángulos del codo
                    left_angle = calculate_angle(left_wrist, left_elbow, left_shoulder)
                    right_angle = calculate_angle(right_wrist, right_elbow, right_shoulder)

                    # Detección del estado de los brazos (subida y bajada)
                    if left_angle > 160 and right_angle > 160:
                        stage = "Down"
                    if 90 < left_angle < 160 and 90 < right_angle < 160 and stage == "Down":
                        stage = "Up"
                        counter += 1
                        print(f'Repetición {counter}')

                    # Verificar si se ha completado una serie
                    if counter >= reps_per_series:
                        series += 1
                        counter = 0
                        print(f'Serie {series} completada')
                        cv2.putText(image, f'Serie {series} completada', (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Verificar si se han completado todas las series
                    if series == total_series:
                        cv2.putText(image, 'Ejercicio completado!', (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        break

                except Exception as e:
                    print("Error al procesar:", e)

            # Mostrar repeticiones y estado en la pantalla
            cv2.putText(image, f"Repeticiones: {counter}/{reps_per_series}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.putText(image, f"Series: {series}/{total_series}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # Mostrar la imagen
        cv2.imshow('MediaPipe Pose', image)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


