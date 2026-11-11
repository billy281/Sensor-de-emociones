// Al abrir la página se enciende la webcam automáticamente: cada fotograma se
// envía al servidor Django (que lo procesa con OpenCV) y se muestra anotado.
(() => {
  const $ = id => document.getElementById(id);
  const video = $('video'), salida = $('salida'), placeholder = $('placeholder');
  const csrf = document.querySelector('[name=csrfmiddlewaretoken]').value;
  const lienzo = document.createElement('canvas');
  const ctx = lienzo.getContext('2d');
  const ANCHO = 640;

  let activo = false, sesionId = null, guardados = 0;
  let cuadros = 0, tUltimo = performance.now();

  const estado = txt => { $('estado').textContent = txt; };

  async function post(url, cuerpo) {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrf },
      body: JSON.stringify(cuerpo || {})
    });
    const datos = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(datos.error || ('HTTP ' + r.status));
    return datos;
  }

  async function iniciar() {
    let flujo;
    try {
      flujo = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    } catch (e) {
      placeholder.textContent = 'No se pudo acceder a la cámara. Permite el acceso y recarga la página.';
      estado('Sin cámara');
      return;
    }
    video.srcObject = flujo;
    await video.play();
    lienzo.width = ANCHO;
    lienzo.height = Math.round(ANCHO * video.videoHeight / video.videoWidth) || 480;

    try {
      sesionId = (await post(URLS.iniciar, {})).id;
    } catch (e) {
      sesionId = null; // si la BD falla, la detección sigue funcionando sin guardar
    }
    activo = true;
    placeholder.style.display = 'none';
    salida.style.display = 'block';
    estado(sesionId ? `Detectando · sesión #${sesionId}` : 'Detectando (sin guardar)');
    bucle();
  }

  async function bucle() {
    while (activo) {
      // efecto espejo
      ctx.save();
      ctx.translate(lienzo.width, 0); ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, lienzo.width, lienzo.height);
      ctx.restore();
      const imagen = lienzo.toDataURL('image/jpeg', 0.7);
      try {
        const r = await post(URLS.procesar, { imagen, sesion_id: sesionId });
        if (r.imagen) salida.src = r.imagen;
        actualizarPanel(r.rostros);
        guardados += r.guardados || 0;
        $('nGuardados').textContent = guardados;
        medirFps();
      } catch (e) {
        estado('Error: ' + e.message);
        await new Promise(res => setTimeout(res, 1000));
      }
    }
  }

  function medirFps() {
    cuadros++;
    const ahora = performance.now();
    if (ahora - tUltimo >= 1000) {
      $('fps').textContent = (cuadros * 1000 / (ahora - tUltimo)).toFixed(1) + ' FPS';
      cuadros = 0; tUltimo = ahora;
    }
  }

  function actualizarPanel(rostros) {
    $('nRostros').textContent = rostros.length;
    // se muestra el rostro más grande
    const r = rostros.slice().sort((a, b) => b.ancho * b.alto - a.ancho * a.alto)[0];
    const principal = $('principal');
    principal.className = 'principal' + (r ? ' emo-' + r.emocion : '');
    principal.textContent = r ? `${r.etiqueta} · ${Math.round(r.confianza * 100)}%` : 'Sin rostro';
    document.querySelectorAll('.barra-emo').forEach(fila => {
      const p = r ? (r.probabilidades[fila.dataset.emo] || 0) : 0;
      fila.querySelector('.relleno').style.width = (p * 100) + '%';
      fila.querySelector('.val').textContent = Math.round(p * 100) + '%';
    });
  }

  // al cerrar o salir de la página se finaliza la sesión
  window.addEventListener('pagehide', () => {
    activo = false;
    if (sesionId) {
      const datos = new FormData();
      datos.append('csrfmiddlewaretoken', csrf); // sendBeacon no permite cabeceras
      navigator.sendBeacon(URLS.finalizar.replace('ID', sesionId), datos);
    }
  });

  iniciar();
})();
