import cv2
import dlib
import numpy as np

# Inicializar captura de vídeo
cap = cv2.VideoCapture('test.mp4')

# Carregar detector de faces e preditor de forma
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Inicializar variáveis para rastreamento
face_tracker = None
face_rect = None
paused = False  # Variável para controlar a pausa
frozen_frame = None  # Variável para armazenar o frame congelado

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertendo o frame para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar faces
        if face_tracker is None:
            faces = detector(gray)
            if len(faces) > 0:
                face_rect = faces[0]  # Este é um objeto dlib.rectangle
                face_tracker = cv2.TrackerKCF_create()
                face_tracker.init(frame, (face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()))
        else:
            # Atualizar o rastreador
            success, new_face_rect = face_tracker.update(frame)
            if success:
                face_rect = dlib.rectangle(int(new_face_rect[0]), int(new_face_rect[1]), 
                                           int(new_face_rect[0] + new_face_rect[2]), 
                                           int(new_face_rect[1] + new_face_rect[3]))
            else:
                face_tracker = None  # Reiniciar rastreador se falhar

        if face_rect is not None:
            # Extrair coordenadas do objeto dlib.rectangle
            x = face_rect.left()
            y = face_rect.top()
            w = face_rect.right() - x
            h = face_rect.bottom() - y

            # Calcular o centro do quadrado
            center_x = x + w // 2
            center_y = y + h // 2

            # Calcular a margem de 5%
            margin_x = int(0.05 * w)
            margin_y = int(0.07 * h)

            # Desenhar o retângulo de rastreamento
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Obter marcos
            landmarks = predictor(gray, face_rect)
            # Desenhar marcos
            for n in range(68):
                x_landmark = landmarks.part(n).x
                y_landmark = landmarks.part(n).y
                cv2.circle(frame, (x_landmark, y_landmark), 2, (0, 255, 0), -1)  # Desenhar um ponto verde

            # Obter as coordenadas do nariz e queixo
            nose = landmarks.part(30)  # Ponta do nariz
            chin = landmarks.part(8)    # Queixo

            # Calcular a inclinação da cabeça em relação ao centro do quadrado com margem
            if nose.y < center_y - 10 - margin_y:
                print("Cabeça: Para Cima")
            elif nose.y > center_y + 10 + margin_y:
                print("Cabeça: Para Baixo")
            elif nose.x < center_x - 10 - margin_x:
                print("Cabeça: Para a Esquerda")
            elif nose.x > center_x + 10 + margin_x:
                print("Cabeça: Para a Direita")
            else:
                print("Cabeça: Nível")

                # Detecção de fechamento de olhos (apenas quando a cabeça está em nível)
                left_eye_top = landmarks.part(37).y  # Ponto superior do olho esquerdo
                left_eye_bottom = landmarks.part(41).y  # Ponto inferior do olho esquerdo
                right_eye_top = landmarks.part(44).y  # Ponto superior do olho direito
                right_eye_bottom = landmarks.part(40).y  # Ponto inferior do olho direito

                # Calcular a altura dos olhos
                left_eye_height = left_eye_bottom - left_eye_top
                right_eye_height = right_eye_bottom - right_eye_top

                # Definir um limite para considerar o olho como fechado
                eye_threshold = 10  # Ajuste este valor conforme necessário

                # Verificar se os olhos estão fechados
                if left_eye_height < eye_threshold:
                    print("Olho: Esquerdo Fechado")
                if right_eye_height < eye_threshold:
                    print("Olho: Direito Fechado")

        # Armazenar o frame atual como congelado
        frozen_frame = frame.copy()

    # Exibir o frame (congelado ou atual)
    if paused and frozen_frame is not None:
        cv2.imshow("Rastreamento de Cabeça", frozen_frame)  # Exibir frame congelado
    else:
        cv2.imshow("Rastreamento de Cabeça", frame)  # Exibir frame

    # Controle de pausa
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):  # Pressione 'p' para pausar
        paused = not paused  # Alternar estado de pausa
    elif key == ord('q'):  # Pressione 'q' para sair
        break

# Liberar a captura e fechar janelas
cap.release()
cv2.destroyAllWindows()