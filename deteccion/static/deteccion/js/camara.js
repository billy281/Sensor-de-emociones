// Captura la webcam en el navegador, envía fotogramas al servidor Django
// (que los procesa con OpenCV) y muestra el video anotado en tiempo real.
(() => {
  const $ = id => document.getElementById(id);
  const video = $('video'), salida = $('salida'), placeholder = $('placeholder');
  const btnIniciar = $('btnIniciar'), btnDetener = $('btnDetener'), btnCaptura = $('btnCaptura');
  const csrf = document.querySelector('[name=csrfmiddlewaretoken]').value;
  const lienzo = document.createElement('canvas');
  const ctx = lienzo.getContext('2d');
  const ANCHO = 640;

  let flujo = null, activo = false, sesionId = null, guardados = 0;
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
    try {
      flujo = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    } catch (e) {
      estado('No se pudo acceder a la cámara: ' + e.message);
      return;
    }
    video.srcObject = flujo;
    await video.play();
    lienzo.width = ANCHO;
    lienzo.height = Math.round(ANCHO * video.videoHeight / video.videoWidth) || 480;

    sesionId = null; guardados = 0; $('nGuardados').textContent = '0';
    $('verSesion').classList.add('oculto');
    if ($('guardar').checked) {
      const s = await post(URLS.iniciar, { nombre: $('nombre').value });
      sesionId = s.id;
    }
    activo = true;
    placeholder.style.display = 'none';
    salida.style.display = 'block';
    btnIniciar.disabled = true; btnDetener.disabled = false; btnCaptura.disabled = false;
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
        if (!activo) break;
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

  async function detener() {
    activo = false;
    if (flujo) flujo.getTracks().forEach(t => t.stop());
    flujo = null;
    video.srcObject = null;
    salida.style.display = 'none'; placeholder.style.display = 'flex';
    btnIniciar.disabled = false; btnDetener.disabled = true; btnCaptura.disabled = true;
    $('fps').textContent = '0 FPS';
    if (sesionId) {
      try {
        const r = await post(URLS.finalizar.replace('ID', sesionId));
        const enlace = $('verSesion');
        enlace.href = r.url; enlace.classList.remove('oculto');
        estado(`Sesión #${sesionId} finalizada · ${r.registros} registros`);
      } catch (e) { estado('Error al finalizar: ' + e.message); }
    } else {
      estado('Detenido');
    }
  }

  function capturar() {
    if (!salida.src) return;
    const a = document.createElement('a');
    a.href = salida.src;
    a.download = 'emocion_' + new Date().toISOString().replace(/[:.]/g, '-') + '.jpg';
    a.click();
  }

  btnIniciar.addEventListener('click', iniciar);
  btnDetener.addEventListener('click', detener);
  btnCaptura.addEventListener('click', capturar);
  window.addEventListener('beforeunload', () => {
    if (sesionId && activo) {
      navigator.sendBeacon(URLS.finalizar.replace('ID', sesionId),
        new Blob([JSON.stringify({})], { type: 'application/json' }));
    }
  });
})();
