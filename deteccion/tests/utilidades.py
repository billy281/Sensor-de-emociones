"""Herramientas comunes para las pruebas: un detector con red neuronal simulada."""
from pathlib import Path
from unittest import mock

import numpy as np

from deteccion.detector import EMOCIONES_MODELO, DetectorEmociones

DATOS = Path(__file__).parent / "datos"
IMAGEN_ROSTRO = DATOS / "rostro.jpg"


class RedFalsa:
    """Imita a cv2.dnn.Net devolviendo puntuaciones fijas."""

    def __init__(self, emocion="felicidad"):
        self.salida = np.zeros((1, len(EMOCIONES_MODELO)), dtype=np.float32)
        self.salida[0, EMOCIONES_MODELO.index(emocion)] = 5.0
        self.entradas = []

    def setInput(self, blob):  # noqa: N802 (API de OpenCV)
        self.entradas.append(blob)

    def forward(self):
        return self.salida


def crear_detector(emocion="felicidad"):
    red = RedFalsa(emocion)
    with mock.patch("deteccion.detector.cv2.dnn.readNetFromONNX", return_value=red):
        detector = DetectorEmociones(__file__)  # cualquier archivo existente sirve de "modelo"
    return detector, red
