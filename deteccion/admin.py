from django.contrib import admin

from .models import RegistroEmocion, SesionDeteccion


class RegistroInline(admin.TabularInline):
    model = RegistroEmocion
    extra = 0
    fields = ("fecha", "emocion", "confianza")
    readonly_fields = fields
    can_delete = False
    show_change_link = True


@admin.register(SesionDeteccion)
class SesionAdmin(admin.ModelAdmin):
    list_display = ("id", "nombre", "origen", "inicio", "fin")
    list_filter = ("origen",)
    inlines = [RegistroInline]


@admin.register(RegistroEmocion)
class RegistroAdmin(admin.ModelAdmin):
    list_display = ("id", "sesion", "emocion", "confianza", "fecha")
    list_filter = ("emocion", "sesion")
