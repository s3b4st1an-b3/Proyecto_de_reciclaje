const video         = document.getElementById('webcam');
const canvas        = document.getElementById('canvas');
const placeholder   = document.getElementById('camera-placeholder');
const scanLine      = document.getElementById('scan-line');
const statusDot     = document.getElementById('status-dot');
const statusText    = document.getElementById('status-text');

const btnStart   = document.getElementById('btn-start');
const btnStop    = document.getElementById('btn-stop');
const btnCapture = document.getElementById('btn-capture');

const outputClass      = document.getElementById('output-class');
const outputConfidence = document.getElementById('output-confidence');
const loading          = document.getElementById('loading');

let streamActivo = null;

// ── Encender cámara ──────────────────────────────────────
btnStart.addEventListener('click', async () => {
    try {
        streamActivo = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: false
        });

        video.srcObject = streamActivo;
        video.style.display  = 'block';
        placeholder.style.display = 'none';
        scanLine.classList.add('active');
        statusDot.classList.add('active');
        statusText.textContent = 'Cámara activa';

        btnStart.disabled   = true;
        btnStop.disabled    = false;
        btnCapture.disabled = false;

    } catch (error) {
        console.error('Error al acceder a la cámara:', error);
        statusText.textContent = 'Sin acceso a la cámara';
        alert('No se pudo acceder a la cámara web.');
    }
});

// ── Apagar cámara ────────────────────────────────────────
btnStop.addEventListener('click', () => {
    if (streamActivo) {
        streamActivo.getTracks().forEach(t => t.stop());
        streamActivo = null;
    }

    video.srcObject = null;
    video.style.display = 'none';
    placeholder.style.display = 'flex';
    scanLine.classList.remove('active');
    statusDot.classList.remove('active');
    statusText.textContent = 'Sistema en espera';

    btnStart.disabled   = false;
    btnStop.disabled    = true;
    btnCapture.disabled = true;

    outputClass.textContent      = 'Esperando escaneo...';
    outputConfidence.textContent = '--';
});

// ── Escanear objeto ──────────────────────────────────────
btnCapture.addEventListener('click', async () => {
    if (!streamActivo) return;

    loading.style.display          = 'flex';
    outputClass.textContent        = 'Analizando...';
    outputConfidence.textContent   = '--';
    btnCapture.disabled            = true;
    statusText.textContent         = 'Analizando...';

    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

    const fotoBase64 = canvas.toDataURL('image/jpeg', 0.8);

    try {
        // Esta es la URL que acabamos de validar que funciona
        const respuesta = await fetch('https://proyecto-reciclaje-as3u.vercel.app/api/classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: fotoBase64 })
        });

        const resultado = await respuesta.json();

        if (resultado.error) throw new Error(resultado.error);

        outputClass.textContent        = resultado.clase;
        outputConfidence.textContent   = resultado.confianza;
        statusText.textContent         = 'Análisis completo';

    } catch (error) {
        console.error('Error en la clasificación:', error);
        outputClass.textContent        = 'Error de lectura';
        outputConfidence.textContent   = '0.0%';
        statusText.textContent         = 'Error de conexión';
        alert('Ocurrió un problema: ' + error.message);
    } finally {
        loading.style.display = 'none';
        btnCapture.disabled   = false;
    }
});