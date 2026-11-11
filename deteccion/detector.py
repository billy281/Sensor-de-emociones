"""Módulo de detección de rostros y clasificación de emociones con OpenCV.

1. Detección de rostros: clasificador en cascada de Haar (cv2.CascadeClassifier).
2. Clasificación de emociones: red neuronal FER+ (formato ONNX) ejecutada con
   el módulo DNN de OpenCV (cv2.dnn.readNetFromONNX).
3. Dibujo de resultados: cv2.rectangle / cv2.putText sobre el fotograma.
"""
import base64
import threading
import time
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


# Brillo medio (0-255) por debajo del cual se avisa de poca iluminación
BRILLO_MINIMO = 45


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
    brillo: float | None = None  # brillo medio del rostro (0-255) en la imagen original

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
        return self.eliminar_duplicados([tuple(int(v) for v in r) for r in rostros])

    @staticmethod
    def eliminar_duplicados(rostros, umbral=0.4):
        """El clasificador Haar a veces marca dos recuadros solapados sobre la misma
        cara. Se conserva el más grande y se descarta cualquier otro cuya zona
        común supere el `umbral` del área del recuadro más pequeño."""
        conservados = []
        for x, y, w, h in sorted(rostros, key=lambda r: r[2] * r[3], reverse=True):
            duplicado = False
            for cx, cy, cw, ch in conservados:
                ancho_comun = min(x + w, cx + cw) - max(x, cx)
                alto_comun = min(y + h, cy + ch) - max(y, cy)
                if ancho_comun > 0 and alto_comun > 0:
                    if ancho_comun * alto_comun > umbral * min(w * h, cw * ch):
                        duplicado = True
                        break
            if not duplicado:
                conservados.append((x, y, w, h))
        return conservados

    def clasificar_emocion(self, rostro_gris):
        """Recibe el recorte de un rostro en gris y devuelve (emocion, confianza, probabilidades)."""
        recorte = cv2.resize(rostro_gris, (self.TAM_ENTRADA, self.TAM_ENTRADA))
        # Se clasifica el rostro y su reflejo horizontal y se promedian ambas
        # predicciones: reduce errores por iluminación lateral o pose ligeramente girada.
        probs = np.zeros(len(EMOCIONES), dtype=np.float64)
        for imagen in (recorte, cv2.flip(recorte, 1)):
            blob = imagen.astype(np.float32).reshape(1, 1, self.TAM_ENTRADA, self.TAM_ENTRADA)
            with self._candado:
                self.red.setInput(blob)
                salida = self.red.forward().flatten()
            exp = np.exp(salida - np.max(salida))  # softmax
            probs += exp / exp.sum()
        probs /= 2
        indice = int(np.argmax(probs))
        probabilidades = {EMOCIONES[i]: round(float(p), 4) for i, p in enumerate(probs)}
        return EMOCIONES[indice], float(probs[indice]), probabilidades

    def analizar(self, fotograma):
        """Analiza un fotograma BGR y devuelve la lista de ResultadoRostro."""
        original = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)
        gris = cv2.equalizeHist(original)
        resultados = []
        for (x, y, w, h) in self.detectar_rostros(gris):
            emocion, confianza, probs = self.clasificar_emocion(gris[y : y + h, x : x + w])
            brillo = round(float(original[y : y + h, x : x + w].mean()), 1)
            resultados.append(ResultadoRostro(x, y, w, h, emocion, confianza, probs, brillo))
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
        if any(r.brillo is not None and r.brillo < BRILLO_MINIMO for r in resultados):
            aviso = "Poca luz: ilumina tu rostro de frente"
            (tw, th), _ = cv2.getTextSize(aviso, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y0 = fotograma.shape[0] - 15
            cv2.rectangle(fotograma, (5, y0 - th - 10), (15 + tw, y0 + 6), (0, 140, 255), cv2.FILLED)
            cv2.putText(fotograma, aviso, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
        return fotograma

    def procesar(self, fotograma, suavizador=None):
        """Analiza, suaviza (opcional) y dibuja. Devuelve (fotograma_anotado, resultados)."""
        resultados = self.analizar(fotograma)
        if suavizador is not None:
            resultados = suavizador.suavizar(resultados)
        return self.dibujar(fotograma, resultados), resultados


class SuavizadorEmociones:
    """Suavizado temporal de las emociones de cada rostro.

    Además, un rostro solo se muestra si aparece en al menos `min_apariciones`
    fotogramas seguidos, lo que elimina detecciones falsas de un solo fotograma.

    Una sola imagen puede clasificarse mal (parpadeo, transición entre gestos,
    desenfoque por movimiento). Para evitar que la etiqueta "salte" de una
    emoción a otra, se sigue cada rostro entre fotogramas (por la cercanía de su
    centro) y se promedian sus probabilidades con una media móvil exponencial.
    """

    def __init__(self, alfa=0.35, caducidad_seg=1.5, min_apariciones=2):
        self.alfa = alfa  # peso del fotograma nuevo (0-1); menor = más estable
        # un rostro se muestra solo tras aparecer en varios fotogramas seguidos:
        # elimina las detecciones falsas momentáneas (sombras, fondo oscuro)
        self.min_apariciones = min_apariciones
        self.caducidad = caducidad_seg
        self.pistas = []  # [{"cx", "cy", "ancho", "probs", "visto"}]
        self.ultimo_uso = time.monotonic()

    def suavizar(self, resultados):
        ahora = time.monotonic()
        self.ultimo_uso = ahora
        self.pistas = [p for p in self.pistas if ahora - p["visto"] <= self.caducidad]
        libres = list(self.pistas)
        suavizados = []
        for r in resultados:
            cx, cy = r.x + r.ancho / 2, r.y + r.alto / 2
            nuevas = np.array([r.probabilidades[e] for e in EMOCIONES])
            pista = min(
                libres,
                key=lambda p: (p["cx"] - cx) ** 2 + (p["cy"] - cy) ** 2,
                default=None,
            )
            if pista and ((pista["cx"] - cx) ** 2 + (pista["cy"] - cy) ** 2) ** 0.5 < 0.6 * max(r.ancho, pista["ancho"]):
                libres.remove(pista)
                pista["probs"] = self.alfa * nuevas + (1 - self.alfa) * pista["probs"]
                pista.update(cx=cx, cy=cy, ancho=r.ancho, visto=ahora)
                pista["apariciones"] += 1
            else:
                pista = {"cx": cx, "cy": cy, "ancho": r.ancho, "probs": nuevas, "visto": ahora, "apariciones": 1}
                self.pistas.append(pista)
            if pista["apariciones"] < self.min_apariciones:
                continue
            probs = pista["probs"] / pista["probs"].sum()
            indice = int(np.argmax(probs))
            suavizados.append(
                ResultadoRostro(
                    r.x, r.y, r.ancho, r.alto, EMOCIONES[indice], float(probs[indice]),
                    {e: round(float(p), 4) for e, p in zip(EMOCIONES, probs)},
                    r.brillo,
                )
            )
        return suavizados


_suavizadores = {}
_candado_suavizadores = threading.Lock()


def obtener_suavizador(clave):
    """Devuelve el suavizador asociado a un cliente (uno por pestaña del navegador)."""
    if not clave:
        return None
    with _candado_suavizadores:
        ahora = time.monotonic()
        for k in [k for k, s in _suavizadores.items() if ahora - s.ultimo_uso > 300]:
            del _suavizadores[k]  # limpieza de clientes inactivos
        if clave not in _suavizadores:
            _suavizadores[clave] = SuavizadorEmociones()
        return _suavizadores[clave]


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
