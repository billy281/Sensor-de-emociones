"""Lógica de persistencia de resultados en PostgreSQL."""
import time

from django.conf import settings

from .models import RegistroEmocion

# Última vez que se guardó un registro para cada sesión (limita escrituras a la BD)
_ultimo_guardado = {}


def guardar_resultados(sesion, resultados, forzar=False):
    """Guarda los rostros detectados como máximo una vez por INTERVALO_REGISTRO_SEG.

    Devuelve el número de registros creados.
    """
    if not resultados or sesion is None or not sesion.activa:
        return 0
    ahora = time.monotonic()
    ultimo = _ultimo_guardado.get(sesion.pk, 0.0)
    if not forzar and ahora - ultimo < settings.INTERVALO_REGISTRO_SEG:
        return 0
    _ultimo_guardado[sesion.pk] = ahora
    registros = [
        RegistroEmocion(
            sesion=sesion,
            emocion=r.emocion,
            confianza=r.confianza,
            probabilidades=r.probabilidades,
            rostro_x=r.x,
            rostro_y=r.y,
            rostro_ancho=r.ancho,
            rostro_alto=r.alto,
        )
        for r in resultados
    ]
    RegistroEmocion.objects.bulk_create(registros)
    return len(registros)
