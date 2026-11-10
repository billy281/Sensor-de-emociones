import json
from unittest import mock

import cv2
from django.test import TestCase, override_settings
from django.urls import reverse

from deteccion.detector import a_data_url
from deteccion.models import RegistroEmocion, SesionDeteccion

from .utilidades import IMAGEN_ROSTRO, crear_detector


def post_json(client, url, datos=None):
    return client.post(url, data=json.dumps(datos or {}), content_type="application/json")


class PaginasTests(TestCase):
    def test_paginas_cargan(self):
        for nombre in ("inicio", "camara", "camara_servidor", "historial"):
            with self.subTest(pagina=nombre):
                r = self.client.get(reverse(f"deteccion:{nombre}"))
                self.assertEqual(r.status_code, 200)

    def test_detalle_y_eliminar_sesion(self):
        sesion = SesionDeteccion.objects.create(nombre="Prueba")
        RegistroEmocion.objects.create(
            sesion=sesion, emocion="felicidad", confianza=0.9, rostro_x=0, rostro_y=0, rostro_ancho=10, rostro_alto=10
        )
        r = self.client.get(reverse("deteccion:detalle", args=[sesion.pk]))
        self.assertContains(r, "Felicidad")
        self.assertEqual(r.context["resumen"][0]["porcentaje"], 100.0)

        r = self.client.post(reverse("deteccion:eliminar", args=[sesion.pk]))
        self.assertRedirects(r, reverse("deteccion:historial"))
        self.assertFalse(SesionDeteccion.objects.exists())
        self.assertFalse(RegistroEmocion.objects.exists())

    def test_detalle_inexistente(self):
        self.assertEqual(self.client.get(reverse("deteccion:detalle", args=[999])).status_code, 404)


@override_settings(INTERVALO_REGISTRO_SEG=60)
class ApiTests(TestCase):
    def setUp(self):
        self.detector, _ = crear_detector("felicidad")
        parche = mock.patch("deteccion.views.obtener_detector", return_value=self.detector)
        parche.start()
        self.addCleanup(parche.stop)
        self.imagen = a_data_url(cv2.imread(str(IMAGEN_ROSTRO)))

    def test_ciclo_completo_de_sesion(self):
        r = post_json(self.client, reverse("deteccion:api_iniciar"), {"nombre": "Clase"})
        self.assertEqual(r.status_code, 201)
        sesion_id = r.json()["id"]

        r = post_json(self.client, reverse("deteccion:api_procesar"), {"imagen": self.imagen, "sesion_id": sesion_id})
        self.assertEqual(r.status_code, 200)
        datos = r.json()
        self.assertEqual(len(datos["rostros"]), 1)
        self.assertEqual(datos["rostros"][0]["emocion"], "felicidad")
        self.assertTrue(datos["imagen"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(datos["guardados"], 1)

        # un segundo fotograma inmediato no se guarda (límite de frecuencia)
        r = post_json(self.client, reverse("deteccion:api_procesar"), {"imagen": self.imagen, "sesion_id": sesion_id})
        self.assertEqual(r.json()["guardados"], 0)

        r = post_json(self.client, reverse("deteccion:api_finalizar", args=[sesion_id]))
        self.assertEqual(r.json()["registros"], 1)
        sesion = SesionDeteccion.objects.get(pk=sesion_id)
        self.assertFalse(sesion.activa)
        self.assertEqual(sesion.emocion_predominante(), "felicidad")

        # una sesión finalizada ya no acepta registros
        r = post_json(self.client, reverse("deteccion:api_procesar"), {"imagen": self.imagen, "sesion_id": sesion_id})
        self.assertEqual(r.json()["guardados"], 0)

    def test_procesar_sin_sesion_no_guarda(self):
        r = post_json(self.client, reverse("deteccion:api_procesar"), {"imagen": self.imagen})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(RegistroEmocion.objects.count(), 0)

    def test_procesar_imagen_invalida(self):
        r = post_json(self.client, reverse("deteccion:api_procesar"), {"imagen": "basura"})
        self.assertEqual(r.status_code, 400)

    def test_procesar_json_invalido(self):
        r = self.client.post(reverse("deteccion:api_procesar"), data="{", content_type="application/json")
        self.assertEqual(r.status_code, 400)

    def test_metodo_get_no_permitido(self):
        self.assertEqual(self.client.get(reverse("deteccion:api_procesar")).status_code, 405)

    def test_estadisticas(self):
        sesion = SesionDeteccion.objects.create()
        for emo in ("felicidad", "felicidad", "tristeza"):
            RegistroEmocion.objects.create(
                sesion=sesion, emocion=emo, confianza=0.7, rostro_x=0, rostro_y=0, rostro_ancho=1, rostro_alto=1
            )
        datos = self.client.get(reverse("deteccion:api_estadisticas")).json()
        self.assertEqual(datos["registros"], 3)
        totales = {e["emocion"]: e["total"] for e in datos["por_emocion"]}
        self.assertEqual(totales, {"felicidad": 2, "tristeza": 1})


class StreamTests(TestCase):
    @override_settings(CAMERA_INDEX=99)
    def test_stream_sin_camara_devuelve_mensaje(self):
        detector, _ = crear_detector()
        with mock.patch("deteccion.camara.obtener_detector", return_value=detector), \
             mock.patch("deteccion.views.obtener_detector", return_value=detector):
            r = self.client.get(reverse("deteccion:stream"))
            self.assertEqual(r["Content-Type"], "multipart/x-mixed-replace; boundary=frame")
            contenido = b"".join(r.streaming_content)
        self.assertTrue(contenido.startswith(b"--frame\r\nContent-Type: image/jpeg"))
