"""Captura de video desde la cámara del servidor con cv2.VideoCapture y
transmisión en tiempo real como flujo MJPEG (multipart/x-mixed-replace)."""
import time

import cv2
import numpy as np

from .detector import codificar_jpeg, obtener_detector
from .servicios import guardar_resultados


def _fotograma_mensaje(texto):
    imagen = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(imagen, texto, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return imagen


def _parte_mjpeg(imagen):
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + codificar_jpeg(imagen) + b"\r\n"


def generar_stream(indice_camara, sesion=None, detectar=True):
    """Generador que entrega fotogramas JPEG anotados de forma continua."""
    captura = cv2.VideoCapture(indice_camara)
    try:
        if not captura.isOpened():
            yield _parte_mjpeg(_fotograma_mensaje("No se pudo abrir la camara del servidor"))
            return
        detector = obtener_detector() if detectar else None
        fallos = 0
        while True:
            ok, fotograma = captura.read()
            if not ok:
                fallos += 1
                if fallos > 30:
                    yield _parte_mjpeg(_fotograma_mensaje("Se perdio la senal de la camara"))
                    return
                time.sleep(0.05)
                continue
            fallos = 0
            fotograma = cv2.flip(fotograma, 1)  # efecto espejo
            if detector:
                fotograma, resultados = detector.procesar(fotograma)
                if sesion is not None:
                    guardar_resultados(sesion, resultados)
            yield _parte_mjpeg(fotograma)
    finally:
        captura.release()
