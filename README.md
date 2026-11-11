# EmoVision — Detección de emociones mediante expresiones faciales

Aplicación web desarrollada con **Django** que captura video en tiempo real, detecta los rostros
con **OpenCV** y clasifica la emoción de cada uno. Los resultados se dibujan directamente sobre el
video y se almacenan en **PostgreSQL** para su consulta posterior.

## Funcionalidades

- **Detección en vivo:** la webcam se captura con `getUserMedia`, cada
  fotograma se envía al servidor, se procesa con OpenCV y se devuelve anotado (≈10 FPS).
- **Detección de rostros:** clasificador en cascada de Haar (`haarcascade_frontalface_default.xml`).
- **Reconocimiento de 8 emociones:** neutral, felicidad, sorpresa, tristeza, enojo, disgusto,
  miedo y desprecio, usando la red FER+ en formato ONNX con `cv2.dnn`.
- **Predicciones estables:** cada rostro se clasifica junto con su reflejo horizontal (se promedian
  ambas lecturas) y se sigue entre fotogramas con una media móvil exponencial, de modo que un
  parpadeo o una transición entre gestos no hace "saltar" la emoción mostrada.
- **Filtrado de detecciones:** se eliminan los recuadros duplicados sobre una misma cara y las
  detecciones falsas que aparecen en un solo fotograma (sombras o zonas oscuras del fondo).
- **Aviso de poca luz:** si el rostro está demasiado oscuro, se muestra un aviso en el video y en el
  panel para que el usuario se ilumine de frente.
- **Resultados sobre el video:** recuadro y etiqueta con el color de cada emoción, porcentaje de
  confianza, barra de confianza y contador de rostros (`cv2.rectangle`, `cv2.putText`).
- **Panel en tiempo real:** emoción principal, probabilidades de las 8 emociones, FPS.
- **Sesiones e historial:** cada sesión se guarda en PostgreSQL (máx. un registro por segundo y
  rostro); el historial muestra resumen, emoción predominante y gráficos (Chart.js).
- **Panel de administración** de Django.

## Tecnologías

| Componente | Tecnología |
|---|---|
| Backend | Python 3.10+ · Django 5.2 |
| Visión por computador | OpenCV (`cv2`): `CascadeClassifier`, `dnn` |
| Modelo de emociones | FER+ (`emotion-ferplus-8.onnx`, ONNX Model Zoo, licencia MIT) |
| Base de datos | PostgreSQL |
| Frontend | HTML, CSS, JavaScript, Chart.js |

## Funcionamiento

```
Webcam ─► Navegador (canvas → JPEG base64) ─► POST /api/procesar/
                                                   │
                         ┌─────────────────────────┘
                         ▼
     OpenCV: escala de grises + equalizeHist
          ─► CascadeClassifier.detectMultiScale  (rostros)
          ─► recorte 64×64 (+ reflejo) ─► cv2.dnn FER+ ─► softmax
          ─► suavizado temporal por rostro (emoción + confianza)
          ─► cv2.rectangle / cv2.putText          (anotación)
          ─► PostgreSQL (RegistroEmocion)
                         │
Navegador ◄── JSON { imagen anotada, rostros[] } ◄┘
```

## Instalación

### 1. Requisitos previos
- Python 3.10 o superior
- PostgreSQL 13 o superior
- Una cámara web

### 2. Clonar el repositorio
```bash
git clone https://github.com/USUARIO/emociones_web.git
cd emociones_web
```

### 3. Crear el entorno virtual e instalar dependencias
(el entorno virtual **no** se sube al repositorio, está en `.gitignore`)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Crear la base de datos en PostgreSQL
```sql
CREATE DATABASE emociones_db;
```

### 5. Configurar variables de entorno
Copia `.env.example` como `.env` y escribe tus credenciales de PostgreSQL:
```
DB_NAME=emociones_db
DB_USER=postgres
DB_PASSWORD=tu_contraseña
DB_HOST=localhost
DB_PORT=5432
```

### 6. Descargar el modelo de emociones (si no está en `deteccion/ml_models/`)
```bash
python scripts/descargar_modelo.py
```

### 7. Migrar y ejecutar
```bash
python manage.py migrate
python manage.py createsuperuser   # opcional, para /admin
python manage.py runserver
```
Abre <http://127.0.0.1:8000/>: la cámara se enciende sola y la detección empieza de inmediato.
El navegador pedirá permiso para usar la cámara (funciona en `localhost` o con HTTPS).

## Uso

1. **Detección en vivo (página principal):** al abrir la web la cámara se enciende automáticamente
   y se crea una sesión en la base de datos. Haz distintas expresiones y observa el recuadro y el
   panel lateral. Al cerrar la pestaña la sesión se finaliza sola.
2. **Historial:** lista de sesiones con número de registros y confianza media; el detalle muestra
   la emoción predominante, la distribución y la evolución en el tiempo.

## Rutas principales

| Ruta | Descripción |
|---|---|
| `/` | Detección en vivo con la cámara del navegador (arranca sola) |
| `/historial/` · `/historial/<id>/` | Historial y detalle de sesión |
| `POST /api/sesiones/` | Inicia una sesión |
| `POST /api/sesiones/<id>/finalizar/` | Finaliza una sesión |
| `POST /api/procesar/` | Procesa un fotograma `{imagen, sesion_id}` |
| `GET /api/estadisticas/` | Totales por emoción |
| `/admin/` | Administración de Django |

## Pruebas

Pruebas automatizadas (30 casos: conversión de imágenes, detección de rostros, dibujo, API,
límite de escritura en BD, sesiones, vistas, suavizado temporal, recuadros duplicados, aviso de
poca luz y reconocimiento de una sonrisa con poca luz, contraluz, ruido y baja resolución):
```bash
python manage.py test
```
La prueba con el modelo real se ejecuta solo si el archivo ONNX está descargado.

### Casos de prueba manuales

| # | Caso | Resultado esperado |
|---|---|---|
| 1 | Abrir la web y mirar de frente | Recuadro sobre el rostro con emoción y % |
| 2 | Sonreír ampliamente | Emoción *Felicidad* con alta confianza |
| 3 | Abrir los ojos y la boca | Emoción *Sorpresa* |
| 4 | Expresión seria | Emoción *Neutral* |
| 5 | Fruncir el ceño | *Enojo* o *Tristeza* |
| 6 | Dos personas frente a la cámara | Dos recuadros, contador “Rostros: 2” |
| 7 | Apagar la luz del cuarto | Aviso naranja “Poca luz: ilumina tu rostro de frente” |
| 8 | Salir del encuadre | Panel muestra “Sin rostro”, “Rostros: 0” |
| 9 | Denegar el permiso de cámara | Mensaje “No se pudo acceder a la cámara” |
| 10 | Cerrar la pestaña | Sesión finalizada y visible en Historial con sus registros |

## Estructura del proyecto

```
emociones_web/
├── config/                 # Configuración de Django (settings, urls, wsgi, asgi)
├── deteccion/              # Aplicación principal
│   ├── detector.py         # Detección de rostros y emociones con OpenCV
│   ├── servicios.py        # Guardado de resultados en PostgreSQL
│   ├── models.py           # SesionDeteccion, RegistroEmocion
│   ├── views.py · urls.py · admin.py
│   ├── ml_models/          # Modelo emotion-ferplus-8.onnx
│   ├── templates/deteccion/
│   ├── static/deteccion/   # CSS y JavaScript (captura de la webcam)
│   └── tests/              # Pruebas automatizadas
├── scripts/descargar_modelo.py
├── manage.py
├── requirements.txt
├── .env.example
└── README.md
```

## Créditos

- Modelo FER+: Barsoum et al., *Training Deep Networks for Facial Expression Recognition with
  Crowd-Sourced Label Distribution* (2016). ONNX Model Zoo, licencia MIT.
- Clasificador Haar: OpenCV.
- Imagen de prueba (`tests/datos/rostro.jpg`): retrato de la astronauta Eileen Collins, NASA (dominio público).
