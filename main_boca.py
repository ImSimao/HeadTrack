import cv2
import dlib
import serial as sp
import numpy as np

# Inicializar captura de vídeo
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Definir largura do frame  640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Definir altura do frame  480

#Start Serial connection
#ser = sp.Serial('COM3', 9600)


# Carregar detector de faces e preditor de pontos faciais
# dlib.get_frontal_face_detector() -> Detector de rostos pré-treinado
# shape_predictor() -> Modelo de 68 pontos para identificar características faciais

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Inicializar rastreador de rosto
face_tracker = None
face_rect = None
paused = False  # Controle de pausa
frozen_frame = None  # Armazena o frame congelado

# Variáveis para controle do tempo de olhos fechados
mouth_closed_time = 0
tracking_active = True  # Estado do rastreamento

# Variáveis de controle de status da cabeça e olhos
nivel = True
display_active = False
headStat = 0  # Estado da cabeça (cima, baixo, esquerda, direita, nivelado)
mouthStat = 0   # Estado dos olhos (abertos ou fechados)
mouth = 0

# Variáveis para suavização de pontos faciais
alpha = 0.7  # Fator de suavização (peso do novo valor)
prev_landmarks = None  # Armazena as posições anteriores dos pontos
nose_history = []  # Histórico de posições do nariz para filtragem




# Criar o rastreador CSRT (mais preciso para rastrear a cabeça ao longo do tempo)
def create_tracker():
    return cv2.TrackerCSRT_create()

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        


        frame = cv2.flip(frame, 1)  # Espelhar a imagem para comportamento natural
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converter para escala de cinza
        
        # Detecção de rosto e rastreamento
        if face_tracker is None:
            faces = detector(gray)  # Detectar faces
            if faces:
                face_rect = faces[0]  # Seleciona a primeira face detectada
                face_tracker = create_tracker()
                face_tracker.init(frame, (face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()))
        else:
            # Atualizar rastreador para seguir a face detectada
            success, new_face_rect = face_tracker.update(frame)
            if success:
                face_rect = dlib.rectangle(
                    int(new_face_rect[0]), int(new_face_rect[1]), 
                    int(new_face_rect[0] + new_face_rect[2]), 
                    int(new_face_rect[1] + new_face_rect[3])
                )
            else:
                face_tracker = None  # Se falhar, reiniciar rastreamento

        if face_rect:
            # Obter coordenadas da face
            x = face_rect.left()
            y = face_rect.top()
            w = face_rect.width()
            h = face_rect.height()
            
            # Calcular o centro do rosto
            center_x =  x + w // 2
            center_y = y + h // 2
            
            # Definir margens para detecção de movimento
            margin_x = int(0.05 * w)
            margin_y = int(0.07 * h)

            # Desenhar um retângulo ao redor do rosto detectado
            cv2.rectangle(frame, (x-70, y-70), (x + w +70, y + h +70), (255, 0, 0), 2)

            # Obter pontos faciais
            landmarks = predictor(gray, face_rect)

            # Suavização dos pontos faciais
            if prev_landmarks is not None:
                for n in range(68):
                    x_new = int(alpha * landmarks.part(n).x + (1 - alpha) * prev_landmarks[n][0])
                    y_new = int(alpha * landmarks.part(n).y + (1 - alpha) * prev_landmarks[n][1])
                    prev_landmarks[n] = (x_new, y_new)
            else:
                prev_landmarks = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

            # Exibir pontos faciais suavizados
            if display_active:
                for x_landmark, y_landmark in prev_landmarks:
                    cv2.circle(frame, (x_landmark, y_landmark), 1, (0, 255, 0), -1)

            # Pegar a posição do nariz e queixo
            nose = prev_landmarks[30]
            chin = prev_landmarks[8]

            # Filtragem da posição do nariz para maior estabilidade
            nose_history.append(nose)
            if len(nose_history) > 5:
                nose_history.pop(0)
            avg_nose = np.mean(nose_history, axis=0).astype(int)

            # Determinar direção da cabeça
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

            # Verificação do estado dos olhos
            mouth_top = prev_landmarks[52][1]
            mouth_bottom = prev_landmarks[63][1]
            
            print(mouth_top)
            print(mouth_bottom)
            mouth_height = mouth_bottom - mouth_top
            

            mouth_threshold = 10  # Valor limite para considerar o olho fechado
            
            print(mouth_height)
            if mouth_height >= mouth_threshold and mouth == 0:
                mouthStat = 50  # Olho esquerdo fechado
            #elif right_eye_height < mouth_threshold and eye == 0:
            #    mouthStat = 60  # Olho direito fechado
            else:
                mouthStat = 0  # Olhos abertos
            
            # Se ambos os olhos estiverem fechados por tempo suficiente, alternar rastreamento
            if mouth_height < mouth_threshold :
                mouth_closed_time += 1

                if mouth_closed_time >= 90:
                    display_active = not display_active
                    mouth_closed_time = 0
                    face_tracker = None if display_active else face_tracker
                    face_rect = None if display_active else face_rect
                    print("Rastreamento Ativado" if display_active else "Rastreamento Parado")

            
            #ser.write([headStat, mouthStat])


            # Armazenar o frame atual como congelado
            frozen_frame = frame.copy()

            # Exibir o frame (congelado ou atual)
            if paused and frozen_frame is not None:
                cv2.imshow("Rastreamento de Cabeça", frozen_frame)  # Exibir frame congelado
            else:
                if tracking_active:
                    cv2.imshow("Rastreamento de Cabeça", frame)  # Exibir frame



    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):  # Pressione 'p' para pausar
        paused = not paused  # Alternar estado de pausa
    elif key == ord('r'):  # Pressione 'r' para resetar
        face_tracker = None  # Reiniciar o rastreador
        face_rect = None  # Reiniciar o retângulo da face
        mouth_closed_time = 0  # Resetar o tempo de fechamento dos olhos
        nivel = True  # Resetar o nível da cabeça
        display_active = False  # Ativar a exibição da display_active
        print("Reset")  # Mensagem de confirmação
    elif key == ord('o'):
        display_active = True
    elif key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
