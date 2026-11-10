from django.urls import path

from . import views

app_name = "deteccion"

urlpatterns = [
    path("", views.inicio, name="inicio"),
    path("camara/", views.camara_navegador, name="camara"),
    path("camara-servidor/", views.camara_servidor, name="camara_servidor"),
    path("stream/", views.stream_servidor, name="stream"),
    path("historial/", views.historial, name="historial"),
    path("historial/<int:pk>/", views.detalle_sesion, name="detalle"),
    path("historial/<int:pk>/eliminar/", views.eliminar_sesion, name="eliminar"),
    path("api/sesiones/", views.api_iniciar_sesion, name="api_iniciar"),
    path("api/sesiones/<int:pk>/finalizar/", views.api_finalizar_sesion, name="api_finalizar"),
    path("api/procesar/", views.api_procesar_fotograma, name="api_procesar"),
    path("api/estadisticas/", views.api_estadisticas, name="api_estadisticas"),
]
