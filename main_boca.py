import cv2
import dlib
import serial as sp
import numpy as np
import threading

# Inicializar captura de vídeo
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ser = sp.Serial('COM3', 9600)

# Carregar detector de faces e preditor de pontos faciais
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Variáveis globais para threading
frame = None
ret = False
running = True
paused = False
frozen_frame = None

# Controle de rastreamento
face_tracker = None
face_rect = None

# Variáveis de controle
mouth_closed_time = 0
tracking_active = True
display_active = False
headStat = 0
mouthStat = 0
mouth = 0

alpha = 0.7
prev_landmarks = None
nose_history = []

# Função para criar rastreador CSRT
def create_tracker():
    return cv2.TrackerCSRT_create()

# Thread para capturar frames continuamente
def grab_frames():
    global frame, ret, running
    while running:
        ret, frame = cap.read()

# Iniciar a thread de captura
grab_thread = threading.Thread(target=grab_frames)
grab_thread.start()

# Loop principal
while True:
    if not paused:
        if frame is None or not ret:
            continue

        current_frame = frame.copy()
        current_frame = cv2.flip(current_frame, 1)
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        if face_tracker is None:
            faces = detector(gray)
            if faces:
                face_rect = faces[0]
                face_tracker = create_tracker()
                face_tracker.init(current_frame, (face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()))
        else:
            success, new_face_rect = face_tracker.update(current_frame)
            if success:
                face_rect = dlib.rectangle(
                    int(new_face_rect[0]), int(new_face_rect[1]),
                    int(new_face_rect[0] + new_face_rect[2]),
                    int(new_face_rect[1] + new_face_rect[3])
                )
            else:
                face_tracker = None

        if face_rect:
            x = face_rect.left()
            y = face_rect.top()
            w = face_rect.width()
            h = face_rect.height()

            center_x = x + w // 2
            center_y = y + h // 2

            margin_x = int(0.05 * w)
            margin_y = int(0.07 * h)

            cv2.rectangle(current_frame, (x-20, y-20), (x + w +20, y + h +20), (255, 0, 0), 2)

            landmarks = predictor(gray, face_rect)

            if prev_landmarks is not None:
                for n in range(68):
                    x_new = int(alpha * landmarks.part(n).x + (1 - alpha) * prev_landmarks[n][0])
                    y_new = int(alpha * landmarks.part(n).y + (1 - alpha) * prev_landmarks[n][1])
                    prev_landmarks[n] = (x_new, y_new)
            else:
                prev_landmarks = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

            if display_active:
                for x_landmark, y_landmark in prev_landmarks:
                    cv2.circle(current_frame, (x_landmark, y_landmark), 1, (0, 255, 0), -1)

            nose = prev_landmarks[30]
            chin = prev_landmarks[8]

            nose_history.append(nose)
            if len(nose_history) > 5:
                nose_history.pop(0)
            avg_nose = np.mean(nose_history, axis=0).astype(int)

            if display_active:
                if avg_nose[1] < center_y - margin_y:
                    print("Cabeça: Para Cima")
                    headStat = 10
                    mouth = 1
                    mouth_closed_time = 0
                elif avg_nose[1] > center_y + margin_y:
                    print("Cabeça: Para Baixo")
                    headStat = 20
                    mouth = 1
                    mouth_closed_time = 0
                elif avg_nose[0] < center_x - margin_x:
                    print("Cabeça: Para a Esquerda")
                    headStat = 40
                    mouth = 1
                    mouth_closed_time = 0
                elif avg_nose[0] > center_x + margin_x:
                    print("Cabeça: Para a Direita")
                    headStat = 30
                    mouth = 1
                    mouth_closed_time = 0
                else:
                    print("Cabeça: Nível")
                    headStat = 0
                    mouth = 0

            mouth_top = prev_landmarks[52][1]
            mouth_bottom = prev_landmarks[63][1]
            mouth_height = mouth_bottom - mouth_top

            mouth_threshold = 10

            if mouth_height >= mouth_threshold and mouth == 0:
                mouthStat = 50
            else:
                mouthStat = 0

            if mouth_height < mouth_threshold:
                mouth_closed_time += 1

                if mouth_closed_time >= 90:
                    display_active = not display_active
                    mouth_closed_time = 0
                    face_tracker = None if display_active else face_tracker
                    face_rect = None if display_active else face_rect
                    print("Rastreamento Ativado" if display_active else "Rastreamento Parado")

            #ser.write([headStat, mouthStat])

            frozen_frame = current_frame.copy()

        if paused and frozen_frame is not None:
            cv2.imshow("Rastreamento de Cabeça", frozen_frame)
        else:
            if tracking_active:
                cv2.imshow("Rastreamento de Cabeça", current_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        paused = not paused
    elif key == ord('r'):
        face_tracker = None
        face_rect = None
        mouth_closed_time = 0
        display_active = False
        print("Reset")
    elif key == ord('o'):
        display_active = True
    elif key == ord('q'):
        running = False
        break

# Finalizar tudo
grab_thread.join()
cap.release()
cv2.destroyAllWindows()
