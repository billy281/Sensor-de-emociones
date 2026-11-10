"""Módulo de detección de rostros y clasificación de emociones con OpenCV.

1. Detección de rostros: clasificador en cascada de Haar (cv2.CascadeClassifier).
2. Clasificación de emociones: red neuronal FER+ (formato ONNX) ejecutada con
   el módulo DNN de OpenCV (cv2.dnn.readNetFromONNX).
3. Dibujo de resultados: cv2.rectangle / cv2.putText sobre el fotograma.
"""
import base64
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Orden de salida del modelo FER+ (emotion-ferplus-8.onnx)
EMOCIONES = ["neutral", "felicidad", "sorpresa", "tristeza", "enojo", "disgusto", "miedo", "desprecio"]

ETIQUETAS = {
    "neutral": "Neutral",
    "felicidad": "Felicidad",
    "sorpresa": "Sorpresa",
    "tristeza": "Tristeza",
    "enojo": "Enojo",
    "disgusto": "Disgusto",
    "miedo": "Miedo",
    "desprecio": "Desprecio",
}

# Colores en formato BGR (OpenCV)
COLORES = {
    "neutral": (200, 200, 200),
    "felicidad": (0, 215, 255),
    "sorpresa": (255, 191, 0),
    "tristeza": (200, 100, 30),
    "enojo": (40, 40, 230),
    "disgusto": (60, 160, 60),
    "miedo": (180, 80, 160),
    "desprecio": (90, 120, 160),
}


class ModeloNoEncontrado(RuntimeError):
    pass


@dataclass
class ResultadoRostro:
    x: int
    y: int
    ancho: int
    alto: int
    emocion: str
    confianza: float
    probabilidades: dict = field(default_factory=dict)

    @property
    def etiqueta(self):
        return ETIQUETAS[self.emocion]

    def a_dict(self):
        datos = asdict(self)
        datos["etiqueta"] = self.etiqueta
        return datos


class DetectorEmociones:
    """Detecta rostros y clasifica la emoción de cada uno."""

    TAM_ENTRADA = 64  # el modelo FER+ recibe imágenes de 64x64 en escala de grises

    def __init__(self, ruta_modelo, escala=1.1, vecinos=5, tam_minimo=(60, 60)):
        ruta_modelo = Path(ruta_modelo)
        if not ruta_modelo.exists():
            raise ModeloNoEncontrado(
                f"No se encontró el modelo de emociones en {ruta_modelo}. "
                "Ejecuta: python scripts/descargar_modelo.py"
            )
        ruta_cascada = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector_rostros = cv2.CascadeClassifier(str(ruta_cascada))
        if self.detector_rostros.empty():
            raise RuntimeError(f"No se pudo cargar el clasificador Haar: {ruta_cascada}")
        self.red = cv2.dnn.readNetFromONNX(str(ruta_modelo))
        self.escala = escala
        self.vecinos = vecinos
        self.tam_minimo = tam_minimo
        # cv2.dnn.Net no es seguro entre hilos: serializamos su uso
        self._candado = threading.Lock()

    # ------------------------------------------------------------------ #
    def detectar_rostros(self, gris):
        rostros = self.detector_rostros.detectMultiScale(
            gris,
            scaleFactor=self.escala,
            minNeighbors=self.vecinos,
            minSize=self.tam_minimo,
        )
        return [tuple(int(v) for v in r) for r in rostros]

    def clasificar_emocion(self, rostro_gris):
        """Recibe el recorte de un rostro en gris y devuelve (emocion, confianza, probabilidades)."""
        recorte = cv2.resize(rostro_gris, (self.TAM_ENTRADA, self.TAM_ENTRADA))
        blob = recorte.astype(np.float32).reshape(1, 1, self.TAM_ENTRADA, self.TAM_ENTRADA)
        with self._candado:
            self.red.setInput(blob)
            salida = self.red.forward().flatten()
        # softmax
        exp = np.exp(salida - np.max(salida))
        probs = exp / exp.sum()
        indice = int(np.argmax(probs))
        probabilidades = {EMOCIONES[i]: round(float(p), 4) for i, p in enumerate(probs)}
        return EMOCIONES[indice], float(probs[indice]), probabilidades

    def analizar(self, fotograma):
        """Analiza un fotograma BGR y devuelve la lista de ResultadoRostro."""
        gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)
        gris = cv2.equalizeHist(gris)
        resultados = []
        for (x, y, w, h) in self.detectar_rostros(gris):
            emocion, confianza, probs = self.clasificar_emocion(gris[y : y + h, x : x + w])
            resultados.append(ResultadoRostro(x, y, w, h, emocion, confianza, probs))
        return resultados

    # ------------------------------------------------------------------ #
    @staticmethod
    def dibujar(fotograma, resultados):
        """Dibuja recuadros, etiquetas y barra de confianza directamente sobre el fotograma."""
        for r in resultados:
            color = COLORES[r.emocion]
            cv2.rectangle(fotograma, (r.x, r.y), (r.x + r.ancho, r.y + r.alto), color, 2)

            texto = f"{r.etiqueta} {r.confianza:.0%}"
            fuente = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), base = cv2.getTextSize(texto, fuente, 0.7, 2)
            y_texto = max(r.y - 10, th + 10)
            cv2.rectangle(
                fotograma, (r.x, y_texto - th - 8), (r.x + tw + 10, y_texto + base - 2), color, cv2.FILLED
            )
            cv2.putText(fotograma, texto, (r.x + 5, y_texto - 4), fuente, 0.7, (20, 20, 20), 2, cv2.LINE_AA)

            # barra de confianza bajo el rostro
            y_barra = min(r.y + r.alto + 8, fotograma.shape[0] - 8)
            largo = int(r.ancho * r.confianza)
            cv2.rectangle(fotograma, (r.x, y_barra), (r.x + r.ancho, y_barra + 6), (60, 60, 60), cv2.FILLED)
            cv2.rectangle(fotograma, (r.x, y_barra), (r.x + largo, y_barra + 6), color, cv2.FILLED)

        cv2.putText(
            fotograma,
            f"Rostros: {len(resultados)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return fotograma

    def procesar(self, fotograma):
        """Analiza y dibuja. Devuelve (fotograma_anotado, resultados)."""
        resultados = self.analizar(fotograma)
        return self.dibujar(fotograma, resultados), resultados


# ---------------------------------------------------------------------- #
# Utilidades de conversión de imágenes
# ---------------------------------------------------------------------- #
def decodificar_data_url(data_url):
    """Convierte 'data:image/jpeg;base64,...' (o base64 puro) en una imagen BGR de OpenCV."""
    if not data_url:
        raise ValueError("Imagen vacía")
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        bytes_img = base64.b64decode(data_url, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("La imagen no es base64 válido") from exc
    arreglo = np.frombuffer(bytes_img, dtype=np.uint8)
    imagen = cv2.imdecode(arreglo, cv2.IMREAD_COLOR)
    if imagen is None:
        raise ValueError("No se pudo decodificar la imagen")
    return imagen


def codificar_jpeg(imagen, calidad=80):
    ok, buffer = cv2.imencode(".jpg", imagen, [cv2.IMWRITE_JPEG_QUALITY, calidad])
    if not ok:
        raise ValueError("No se pudo codificar la imagen")
    return buffer.tobytes()


def a_data_url(imagen, calidad=80):
    return "data:image/jpeg;base64," + base64.b64encode(codificar_jpeg(imagen, calidad)).decode()


# ---------------------------------------------------------------------- #
# Instancia única (se carga la red una sola vez)
# ---------------------------------------------------------------------- #
_detector = None
_candado_global = threading.Lock()


def obtener_detector():
    global _detector
    if _detector is None:
        with _candado_global:
            if _detector is None:
                from django.conf import settings

                ruta = Path(settings.EMOTION_MODEL_PATH)
                if not ruta.exists():
                    # primer arranque: se intenta descargar el modelo automáticamente
                    try:
                        from scripts.descargar_modelo import descargar

                        descargar(ruta)
                    except Exception:  # noqa: BLE001
                        pass  # DetectorEmociones lanzará ModeloNoEncontrado con instrucciones
                _detector = DetectorEmociones(ruta)
    return _detector
