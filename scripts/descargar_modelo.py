"""Descarga el modelo de emociones FER+ (ONNX Model Zoo, licencia MIT).

Uso:  python scripts/descargar_modelo.py
"""
import sys
import urllib.request
from pathlib import Path

DESTINO = Path(__file__).resolve().parent.parent / "deteccion" / "ml_models" / "emotion-ferplus-8.onnx"
URLS = [
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
    "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
]
TAM_MINIMO = 1_000_000  # el modelo pesa ~34 MB; algo menor es un error


def descargar(destino=DESTINO):
    destino = Path(destino)
    if destino.exists() and destino.stat().st_size > TAM_MINIMO:
        print(f"El modelo ya existe: {destino}")
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(".part")
    for url in URLS:
        try:
            print(f"Descargando {url} ...")
            urllib.request.urlretrieve(url, temporal)
            if temporal.stat().st_size > TAM_MINIMO:
                temporal.replace(destino)
                print(f"Modelo guardado en {destino} ({destino.stat().st_size / 1e6:.1f} MB)")
                return destino
            print("  archivo demasiado pequeño, probando otra fuente")
        except Exception as exc:  # noqa: BLE001
            print(f"  error: {exc}")
    temporal.unlink(missing_ok=True)
    raise RuntimeError("No se pudo descargar el modelo. Descárgalo manualmente desde " + URLS[0])


if __name__ == "__main__":
    try:
        descargar()
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)
