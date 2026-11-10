import base64
import unittest

import cv2
import numpy as np
from django.conf import settings
from django.test import SimpleTestCase

from deteccion.detector import (
    EMOCIONES,
    DetectorEmociones,
    ModeloNoEncontrado,
    ResultadoRostro,
    a_data_url,
    decodificar_data_url,
)

from .utilidades import IMAGEN_ROSTRO, crear_detector


class ConversionImagenTests(SimpleTestCase):
    def test_ida_y_vuelta_data_url(self):
        imagen = np.full((120, 160, 3), 127, dtype=np.uint8)
        data_url = a_data_url(imagen)
        self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))
        recuperada = decodificar_data_url(data_url)
        self.assertEqual(recuperada.shape, (120, 160, 3))

    def test_base64_sin_prefijo(self):
        ok, buf = cv2.imencode(".png", np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertEqual(decodificar_data_url(base64.b64encode(buf).decode()).shape, (10, 10, 3))

    def test_imagen_vacia_o_invalida(self):
        for valor in ("", "data:image/jpeg;base64,###", base64.b64encode(b"no es imagen").decode()):
            with self.subTest(valor=valor), self.assertRaises(ValueError):
                decodificar_data_url(valor)


class DetectorTests(SimpleTestCase):
    def test_modelo_inexistente(self):
        with self.assertRaises(ModeloNoEncontrado):
            DetectorEmociones("/ruta/que/no/existe.onnx")

    def test_imagen_sin_rostros(self):
        detector, _ = crear_detector()
        self.assertEqual(detector.analizar(np.zeros((480, 640, 3), dtype=np.uint8)), [])

    def test_detecta_rostro_real(self):
        detector, red = crear_detector("sorpresa")
        imagen = cv2.imread(str(IMAGEN_ROSTRO))
        resultados = detector.analizar(imagen)
        self.assertEqual(len(resultados), 1)
        r = resultados[0]
        self.assertEqual(r.emocion, "sorpresa")
        self.assertGreater(r.confianza, 0.9)
        self.assertAlmostEqual(sum(r.probabilidades.values()), 1.0, places=2)
        # la red recibe un blob 1x1x64x64
        self.assertEqual(red.entradas[0].shape, (1, 1, 64, 64))

    def test_dibujar_modifica_fotograma(self):
        fotograma = np.zeros((300, 300, 3), dtype=np.uint8)
        r = ResultadoRostro(50, 60, 120, 120, "enojo", 0.8, {})
        DetectorEmociones.dibujar(fotograma, [r])
        self.assertGreater(int(fotograma.sum()), 0)
        # el borde del recuadro tiene el color asignado a "enojo"
        self.assertEqual(tuple(fotograma[120, 50]), (40, 40, 230))

    def test_a_dict(self):
        r = ResultadoRostro(1, 2, 3, 4, "miedo", 0.5, {"miedo": 0.5})
        self.assertEqual(r.a_dict()["etiqueta"], "Miedo")


@unittest.skipUnless(settings.EMOTION_MODEL_PATH.exists(), "Modelo FER+ no descargado")
class ModeloRealTests(SimpleTestCase):
    def test_modelo_real_clasifica(self):
        detector = DetectorEmociones(settings.EMOTION_MODEL_PATH)
        resultados = detector.analizar(cv2.imread(str(IMAGEN_ROSTRO)))
        self.assertEqual(len(resultados), 1)
        self.assertIn(resultados[0].emocion, EMOCIONES)
