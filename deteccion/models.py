from django.db import models
from django.db.models import Count


class SesionDeteccion(models.Model):
    """Una sesión de detección (desde que se enciende la cámara hasta que se apaga)."""

    ORIGEN_NAVEGADOR = "navegador"
    ORIGEN_SERVIDOR = "servidor"
    ORIGENES = [
        (ORIGEN_NAVEGADOR, "Cámara del navegador"),
        (ORIGEN_SERVIDOR, "Cámara del servidor"),
    ]

    nombre = models.CharField(max_length=100, blank=True)
    origen = models.CharField(max_length=20, choices=ORIGENES, default=ORIGEN_NAVEGADOR)
    inicio = models.DateTimeField(auto_now_add=True)
    fin = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-inicio"]
        verbose_name = "sesión de detección"
        verbose_name_plural = "sesiones de detección"

    def __str__(self):
        return self.nombre or f"Sesión #{self.pk}"

    @property
    def activa(self):
        return self.fin is None

    @property
    def duracion(self):
        if not self.fin:
            return None
        return self.fin - self.inicio

    def resumen_emociones(self):
        """Devuelve [{'emocion': ..., 'total': ...}] ordenado de mayor a menor."""
        return list(
            self.registros.values("emocion").annotate(total=Count("id")).order_by("-total")
        )

    def emocion_predominante(self):
        resumen = self.resumen_emociones()
        return resumen[0]["emocion"] if resumen else None


class RegistroEmocion(models.Model):
    """Emoción detectada en un rostro en un instante determinado."""

    EMOCIONES = [
        ("neutral", "Neutral"),
        ("felicidad", "Felicidad"),
        ("sorpresa", "Sorpresa"),
        ("tristeza", "Tristeza"),
        ("enojo", "Enojo"),
        ("disgusto", "Disgusto"),
        ("miedo", "Miedo"),
        ("desprecio", "Desprecio"),
    ]

    sesion = models.ForeignKey(SesionDeteccion, on_delete=models.CASCADE, related_name="registros")
    emocion = models.CharField(max_length=20, choices=EMOCIONES)
    confianza = models.FloatField()
    probabilidades = models.JSONField(default=dict)
    rostro_x = models.IntegerField()
    rostro_y = models.IntegerField()
    rostro_ancho = models.IntegerField()
    rostro_alto = models.IntegerField()
    fecha = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-fecha"]
        verbose_name = "registro de emoción"
        verbose_name_plural = "registros de emociones"

    def __str__(self):
        return f"{self.get_emocion_display()} ({self.confianza:.0%})"
