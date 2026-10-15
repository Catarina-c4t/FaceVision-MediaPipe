#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FaceVision - Versão Desktop Python (OpenCV + MediaPipe)
Exemplo alternativo em Python puro caso queira rodar diretamente em janela desktop.
Requer: pip install opencv-python mediapipe
"""

import sys

def main():
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        print("[ERRO] Bibliotecas necessarias nao encontradas.")
        print("Para rodar a versao desktop em Python, execute:")
        print("    pip install -r requirements.txt")
        print("\nOu execute o site web diretamente com:")
        print("    python server.py")
        sys.exit(1)

    print("Inicializando MediaPipe Face Mesh...")
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 200))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Nao foi possivel acessar a webcam.")
        sys.exit(1)

    print("Webcam conectada. Pressione 'q' ou 'ESC' na janela para sair.")

    with mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            # Inverte para efeito espelho
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Conversão para RGB para processar no MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = face_mesh.process(rgb_frame)

            # Desenha malha e informações
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=drawing_spec
                    )

                    # Pega coordenadas para indicar texto
                    nose = face_landmarks.landmark[1]
                    cx, cy = int(nose.x * w), int(nose.y * h)
                    cv2.putText(
                        frame,
                        "MediaPipe Face",
                        (cx - 50, cy - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 128),
                        2
                    )

            cv2.imshow("FaceVision Python (Pressione Q para sair)", frame)
            key = cv2.waitKey(5) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
