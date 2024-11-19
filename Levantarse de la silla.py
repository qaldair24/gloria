import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# TODO: URL del stream (reemplaza con la URL de tu transmision)
# url_stream = 'http://192.168.0.100:8080/video'
# cap = cv2.VideoCapture(url_stream)
cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def is_person_centered(landmarks, frame_width):
    left_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame_width
    right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame_width
    person_center_x = (left_hip_x + right_hip_x) / 2
    frame_center_x = frame_width / 2
    tolerance = frame_width * 0.15  # Margen de tolerancia para el centrado

    return abs(person_center_x - frame_center_x) < tolerance

counter = 0
stage = None
total_reps = 12  
total_series = 3   
series_count = 0  

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Convertimos la imagen a RGB para procesarla con Mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detectar los puntos clave del cuerpo
        results = pose.process(image)

        # Convertimos la imagen de nuevo a BGR para mostrarla
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        frame_height, frame_width, _ = image.shape

        # Dibujamos los puntos y las conexiones de la pose
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Filtrar para asegurarse que la persona esté centrada en la cámara
            if is_person_centered(landmarks, frame_width):
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

                # Obtener coordenadas de caderas, rodillas y tobillos
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # Calcular los ángulos de las rodillas 
                left_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_angle = calculate_angle(right_hip, right_knee, right_ankle)

                # Condiciones para detectar si la persona está completamente de pie o sentada
                if left_angle > 160 and right_angle > 160:
                    stage = "up"
                elif (left_angle < 130 and right_angle < 130) and stage == 'up':
                    stage = "down"
                    counter += 1
                    print(f"Repetición completada: {counter}")

                    # Lógica para series y repeticiones
                    if counter == total_reps:
                        series_count += 1
                        counter = 0
                        if series_count == total_series:
                            cv2.putText(image, '¡Entrenamiento Completado!', (10, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar contador de repeticiones y series en pantalla
        cv2.putText(image, f'Reps: {counter}/{total_reps}', (10, 400), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f'Series: {series_count}/{total_series}', (10, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Mostrar la imagen con la detección
        cv2.imshow('MediaPipe', image)

        # Salir si se presiona 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
