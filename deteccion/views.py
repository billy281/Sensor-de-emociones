import json

from django.conf import settings
from django.db.models import Avg, Count
from django.http import HttpResponseBadRequest, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from .camara import generar_stream
from .detector import ETIQUETAS, ModeloNoEncontrado, a_data_url, decodificar_data_url, obtener_detector
from .models import RegistroEmocion, SesionDeteccion
from .servicios import guardar_resultados


# ---------------------------------------------------------------------- #
# Páginas
# ---------------------------------------------------------------------- #
def inicio(request):
    """Página principal: la detección arranca sola al abrir la web."""
    return render(request, "deteccion/camara.html", {"etiquetas": ETIQUETAS})


def camara_navegador(request):
    return redirect("deteccion:inicio")


def camara_servidor(request):
    return render(request, "deteccion/camara_servidor.html")


def historial(request):
    sesiones = SesionDeteccion.objects.annotate(n=Count("registros"), conf=Avg("registros__confianza"))
    return render(request, "deteccion/historial.html", {"sesiones": sesiones})


def detalle_sesion(request, pk):
    sesion = get_object_or_404(SesionDeteccion, pk=pk)
    resumen = sesion.resumen_emociones()
    total = sum(r["total"] for r in resumen) or 1
    for r in resumen:
        r["etiqueta"] = ETIQUETAS.get(r["emocion"], r["emocion"])
        r["porcentaje"] = round(100 * r["total"] / total, 1)
    registros = sesion.registros.order_by("fecha")
    linea_tiempo = [
        {"t": reg.fecha.isoformat(), "emocion": reg.emocion, "confianza": round(reg.confianza, 3)}
        for reg in registros
    ]
    return render(
        request,
        "deteccion/detalle.html",
        {
            "sesion": sesion,
            "resumen": resumen,
            "ultimos": sesion.registros.all()[:20],
            "datos_grafico": {"resumen": resumen, "linea": linea_tiempo, "etiquetas": ETIQUETAS},
        },
    )


@require_POST
def eliminar_sesion(request, pk):
    get_object_or_404(SesionDeteccion, pk=pk).delete()
    return redirect("deteccion:historial")


# ---------------------------------------------------------------------- #
# Transmisión MJPEG desde la cámara del servidor
# ---------------------------------------------------------------------- #
@require_GET
def stream_servidor(request):
    detectar = request.GET.get("detectar", "1") != "0"
    sesion = None
    if request.GET.get("guardar") == "1":
        sesion = SesionDeteccion.objects.create(
            origen=SesionDeteccion.ORIGEN_SERVIDOR, nombre=request.GET.get("nombre", "")[:100]
        )
    try:
        obtener_detector()
    except ModeloNoEncontrado:
        detectar = False
    respuesta = StreamingHttpResponse(
        generar_stream(settings.CAMERA_INDEX, sesion=sesion, detectar=detectar),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )
    respuesta["Cache-Control"] = "no-cache"
    return respuesta


# ---------------------------------------------------------------------- #
# API JSON (usada por la cámara del navegador)
# ---------------------------------------------------------------------- #
def _leer_json(request):
    try:
        return json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return None


@require_POST
def api_iniciar_sesion(request):
    datos = _leer_json(request) or {}
    sesion = SesionDeteccion.objects.create(
        nombre=str(datos.get("nombre", ""))[:100], origen=SesionDeteccion.ORIGEN_NAVEGADOR
    )
    return JsonResponse({"id": sesion.pk, "nombre": str(sesion)}, status=201)


@require_POST
def api_finalizar_sesion(request, pk):
    sesion = get_object_or_404(SesionDeteccion, pk=pk)
    if sesion.fin is None:
        sesion.fin = timezone.now()
        sesion.save(update_fields=["fin"])
    return JsonResponse({"id": sesion.pk, "registros": sesion.registros.count(), "url": f"/historial/{sesion.pk}/"})


@require_POST
def api_procesar_fotograma(request):
    """Recibe un fotograma (data URL JPEG), detecta emociones con OpenCV y
    devuelve la imagen anotada junto con los resultados."""
    datos = _leer_json(request)
    if datos is None:
        return HttpResponseBadRequest("JSON inválido")
    try:
        fotograma = decodificar_data_url(datos.get("imagen", ""))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    try:
        detector = obtener_detector()
    except ModeloNoEncontrado as exc:
        return JsonResponse({"error": str(exc)}, status=503)

    anotado, resultados = detector.procesar(fotograma)

    guardados = 0
    sesion_id = datos.get("sesion_id")
    if sesion_id:
        sesion = SesionDeteccion.objects.filter(pk=sesion_id, fin__isnull=True).first()
        if sesion:
            guardados = guardar_resultados(sesion, resultados)

    return JsonResponse(
        {
            "rostros": [r.a_dict() for r in resultados],
            "imagen": a_data_url(anotado) if datos.get("devolver_imagen", True) else None,
            "guardados": guardados,
        }
    )


@require_GET
def api_estadisticas(request):
    por_emocion = list(
        RegistroEmocion.objects.values("emocion").annotate(total=Count("id"), confianza=Avg("confianza"))
    )
    for e in por_emocion:
        e["etiqueta"] = ETIQUETAS.get(e["emocion"], e["emocion"])
    return JsonResponse(
        {
            "sesiones": SesionDeteccion.objects.count(),
            "registros": RegistroEmocion.objects.count(),
            "por_emocion": por_emocion,
        }
    )
