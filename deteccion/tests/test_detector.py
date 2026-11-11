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
    SuavizadorEmociones,
    a_data_url,
    decodificar_data_url,
    obtener_suavizador,
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

    def test_elimina_recuadros_duplicados(self):
        # dos recuadros sobre la misma cara (como en una webcam con poca luz) + otra cara aparte
        rostros = [(100, 100, 200, 200), (110, 150, 180, 180), (500, 100, 150, 150)]
        self.assertEqual(
            DetectorEmociones.eliminar_duplicados(rostros), [(100, 100, 200, 200), (500, 100, 150, 150)]
        )

    def test_aviso_de_poca_luz(self):
        oscuro = np.zeros((300, 400, 3), dtype=np.uint8)
        normal = oscuro.copy()
        DetectorEmociones.dibujar(oscuro, [ResultadoRostro(50, 50, 100, 100, "neutral", 0.9, {}, brillo=15)])
        DetectorEmociones.dibujar(normal, [ResultadoRostro(50, 50, 100, 100, "neutral", 0.9, {}, brillo=120)])
        # el aviso naranja se dibuja en la parte inferior solo cuando hay poca luz
        self.assertGreater(int(oscuro[260:, :].sum()), 0)
        self.assertEqual(int(normal[260:, :].sum()), 0)

    def test_brillo_del_rostro(self):
        detector, _ = crear_detector()
        r = detector.analizar(cv2.imread(str(IMAGEN_ROSTRO)))[0]
        self.assertGreater(r.brillo, 45)

    def test_a_dict(self):
        r = ResultadoRostro(1, 2, 3, 4, "tristeza", 0.5, {"tristeza": 0.5})
        self.assertEqual(r.a_dict()["etiqueta"], "Tristeza")

    def test_solo_cinco_emociones(self):
        # aunque la red diga "miedo", la app elige entre las 5 emociones activas
        detector, red = crear_detector("miedo")
        red.salida[0, 1] = 2.0  # felicidad es la siguiente más probable
        emocion, _, probs = detector.clasificar_emocion(np.zeros((80, 80), dtype=np.uint8))
        self.assertEqual(emocion, "felicidad")
        self.assertEqual(set(probs), set(EMOCIONES))
        self.assertEqual(len(EMOCIONES), 5)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=3)


@unittest.skipUnless(settings.EMOTION_MODEL_PATH.exists(), "Modelo FER+ no descargado")
class ModeloRealTests(SimpleTestCase):
    def test_modelo_real_clasifica(self):
        detector = DetectorEmociones(settings.EMOTION_MODEL_PATH)
        resultados = detector.analizar(cv2.imread(str(IMAGEN_ROSTRO)))
        self.assertEqual(len(resultados), 1)
        self.assertIn(resultados[0].emocion, EMOCIONES)


class SuavizadorTests(SimpleTestCase):
    """El suavizado evita que un fotograma aislado cambie la emoción mostrada."""

    @staticmethod
    def _rostro(emocion, x=100):
        probs = {e: 0.0 for e in EMOCIONES}
        probs[emocion] = 1.0
        return ResultadoRostro(x, 100, 120, 120, emocion, 1.0, probs)

    def test_fotograma_aislado_no_cambia_la_emocion(self):
        s = SuavizadorEmociones()
        for _ in range(5):
            s.suavizar([self._rostro("felicidad")])
        r = s.suavizar([self._rostro("tristeza")])[0]
        self.assertEqual(r.emocion, "felicidad")

    def test_cambio_sostenido_si_se_refleja(self):
        s = SuavizadorEmociones()
        for _ in range(5):
            s.suavizar([self._rostro("felicidad")])
        for _ in range(3):
            r = s.suavizar([self._rostro("sorpresa")])[0]
        self.assertEqual(r.emocion, "sorpresa")

    def test_rostros_distintos_se_siguen_por_separado(self):
        s = SuavizadorEmociones()
        for _ in range(4):
            visibles = s.suavizar([self._rostro("felicidad", x=0), self._rostro("enojo", x=500)])
        a, b = visibles
        self.assertEqual((a.emocion, b.emocion), ("felicidad", "enojo"))
        self.assertAlmostEqual(sum(a.probabilidades.values()), 1.0, places=2)

    def test_deteccion_falsa_de_un_solo_fotograma_se_ignora(self):
        s = SuavizadorEmociones()
        self.assertEqual(s.suavizar([self._rostro("neutral")]), [])  # 1.ª aparición: aún no se muestra
        self.assertEqual(len(s.suavizar([self._rostro("neutral")])), 1)  # 2.ª: ya es un rostro real
        # una sombra que aparece un único fotograma en otra zona no se muestra
        visibles = s.suavizar([self._rostro("neutral"), self._rostro("tristeza", x=600)])
        self.assertEqual(len(visibles), 1)

    def test_obtener_suavizador_por_cliente(self):
        self.assertIsNone(obtener_suavizador(""))
        self.assertIs(obtener_suavizador("pestana-1"), obtener_suavizador("pestana-1"))
        self.assertIsNot(obtener_suavizador("pestana-1"), obtener_suavizador("pestana-2"))

    def test_clasificacion_promedia_imagen_y_reflejo(self):
        detector, red = crear_detector("felicidad")
        detector.clasificar_emocion(np.zeros((80, 80), dtype=np.uint8))
        self.assertEqual(len(red.entradas), 2)


@unittest.skipUnless(settings.EMOTION_MODEL_PATH.exists(), "Modelo FER+ no descargado")
class CalibracionModeloRealTests(SimpleTestCase):
    """Con el modelo real, una sonrisa debe reconocerse como felicidad aun con
    poca luz, ruido o baja resolución (condiciones típicas de una webcam)."""

    def test_sonrisa_en_condiciones_de_webcam(self):
        detector = DetectorEmociones(settings.EMOTION_MODEL_PATH)
        img = cv2.imread(str(IMAGEN_ROSTRO))
        rng = np.random.default_rng(0)
        variantes = {
            "original": img,
            "poca luz": np.clip(255 * (img / 255.0) ** 2.2, 0, 255).astype(np.uint8),
            "contraluz": (img * 0.45).astype(np.uint8),
            "ruido": np.clip(img + rng.normal(0, 18, img.shape), 0, 255).astype(np.uint8),
            "baja resolucion": cv2.resize(cv2.resize(img, None, fx=0.5, fy=0.5), img.shape[1::-1]),
        }
        for nombre, imagen in variantes.items():
            with self.subTest(condicion=nombre):
                resultados = detector.analizar(imagen)
                self.assertEqual(len(resultados), 1)
                self.assertEqual(resultados[0].emocion, "felicidad")
