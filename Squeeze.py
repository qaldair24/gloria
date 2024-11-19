import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,  # Aumentamos la confianza para mejorar la detección
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def is_hand_closed(hand_landmarks):
    """Detecta si la mano está cerrada en función de la posición de los dedos"""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y

    # La mano se considera cerrada si todas las puntas de los dedos están por debajo de sus articulaciones
    return (thumb_tip < thumb_mcp) and (index_tip < index_pip) and \
           (middle_tip < index_pip) and (ring_tip < index_pip) and (pinky_tip < index_pip)

def is_person_centered(landmarks, frame_width):
    """Verifica si la mano está centrada en la imagen"""
    hand_x = landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame_width
    frame_center_x = frame_width / 2
    tolerance = frame_width * 0.15  # Margen de tolerancia para el centrado

    return abs(hand_x - frame_center_x) < tolerance

reps = 0
series = 0
max_reps = 12  
max_series = 6  # Total de 6 series: 3 para la mano derecha y 3 para la izquierda
holding = False  
hand_in_use = "Right"  # Control para alternar manos
last_action_time = 0  # Tiempo para evitar contar repeticiones demasiado rápido

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened() and series < max_series:
        ret, frame = cap.read()
        if not ret:
            continue

        # Reflejar horizontalmente la imagen (mirror fix)
        frame = cv2.flip(frame, 1)

        # Convertir la imagen a RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Procesar la imagen para detectar manos
        results_hands = hands.process(image)

        # Convertir la imagen de nuevo a BGR para mostrarla
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Verificar si se detectan manos
        if results_hands.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                # Detectar si la mano es derecha o izquierda
                hand_label = results_hands.multi_handedness[idx].classification[0].label
                
                # Establecer colores según la mano
                if hand_label == "Right":
                    color = (255, 0, 0)  # Azul para la mano derecha
                else:
                    color = (0, 0, 255)  # Rojo para la mano izquierda

                # Solo procesar la mano que corresponde a la secuencia
                if hand_label == hand_in_use:
                    # Verificar si la mano está centrada
                    if is_person_centered(hand_landmarks, frame.shape[1]):
                        # Dibujar los landmarks de la mano
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                        )

                        # Detectar si la mano está cerrada
                        if is_hand_closed(hand_landmarks):
                            if not holding:
                                holding = True
                                # Evitar contar repeticiones demasiado rápido
                                if time.time() - last_action_time > 1:
                                    reps += 1
                                    last_action_time = time.time()
                                    cv2.putText(image, f'Squeeze Complete: {hand_in_use} hand', (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                                1, color, 2)

                                    if reps == max_reps:
                                        series += 1
                                        reps = 0
                                        # Cambiar de mano
                                        hand_in_use = "Left" if hand_in_use == "Right" else "Right"
                                        cv2.putText(image, f'Series {series} Completed', (10, 150),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                        time.sleep(2)  # Pausa entre series

                                        if series == max_series:
                                            cv2.putText(image, 'Training Completed!', (10, 200),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                            cv2.imshow('MediaPipe', image)
                                            cv2.waitKey(5000)  
                                            break
                        else:
                            holding = False
                            cv2.putText(image, f'Squeeze The Ball: {hand_in_use} hand', (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, color, 2)
                    else:
                        cv2.putText(image, 'Center your hand', (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2)
        else:
            # Mensaje de depuración si no se detectan manos
            cv2.putText(image, 'No hand detected', (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        # Mostrar repeticiones y series en la pantalla
        cv2.putText(image, f'Reps: {reps}/{max_reps}', (10, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, f'Series: {series}/{max_series}', (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Mostrar la mano actual en la parte inferior
        cv2.putText(image, f'{hand_in_use} Hand', (10, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Mostrar la imagen procesada
        cv2.imshow('MediaPipe', image)

        # Presionar 'q' para salir
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()




