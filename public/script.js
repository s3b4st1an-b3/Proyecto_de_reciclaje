const video = document.getElementById("webcam");
const imagePreview = document.getElementById("image-preview");
const canvas = document.getElementById("canvas");
const placeholder = document.getElementById("camera-placeholder");
const scanLine = document.getElementById("scan-line");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");

const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");
const btnCapture = document.getElementById("btn-capture");
const btnUpload = document.getElementById("btn-upload");
const imageInput = document.getElementById("image-input");

const outputClass = document.getElementById("output-class");
const outputConfidence = document.getElementById("output-confidence");
const loading = document.getElementById("loading");

let streamActivo = null;
let analizando = false;

function actualizarEstado(texto, activo = false) {
    statusText.textContent = texto;
    statusDot.classList.toggle("active", activo);
}

function mostrarPlaceholder(mostrar) {
    placeholder.style.display = mostrar ? "flex" : "none";
}

function mensajeErrorCamara(error) {
    const mensajes = {
        NotAllowedError: "El navegador bloqueó la cámara. Permite el acceso desde el icono junto a la dirección y recarga la página.",
        NotFoundError: "No se encontró ninguna cámara en este dispositivo.",
        NotReadableError: "La cámara está siendo usada por otra aplicación. Cierra esa aplicación e intenta de nuevo.",
        OverconstrainedError: "La cámara no admite la configuración solicitada.",
        SecurityError: "La cámara requiere HTTPS o http://localhost.",
    };

    return mensajes[error.name] || `No se pudo acceder a la cámara (${error.name || "error desconocido"}).`;
}

function esperarVideoListo() {
    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
        return Promise.resolve();
    }

    return new Promise((resolve, reject) => {
        const timeout = window.setTimeout(
            () => reject(new Error("La cámara tardó demasiado en iniciar.")),
            10000,
        );

        video.addEventListener("loadedmetadata", () => {
            window.clearTimeout(timeout);
            resolve();
        }, { once: true });
    });
}

function detenerCamara({ limpiarResultado = true } = {}) {
    if (streamActivo) {
        streamActivo.getTracks().forEach((track) => track.stop());
    }

    streamActivo = null;
    video.srcObject = null;
    video.style.display = "none";
    scanLine.classList.remove("active");
    btnStart.disabled = false;
    btnStop.disabled = true;
    btnCapture.disabled = true;

    if (imagePreview.style.display !== "block") {
        mostrarPlaceholder(true);
    }

    actualizarEstado("Sistema en espera");

    if (limpiarResultado) {
        outputClass.textContent = "Esperando escaneo...";
        outputConfidence.textContent = "--";
    }
}

function convertirArchivoADataUrl(archivo) {
    return new Promise((resolve, reject) => {
        const lector = new FileReader();
        lector.onload = () => resolve(lector.result);
        lector.onerror = () => reject(new Error("No se pudo leer la imagen seleccionada."));
        lector.readAsDataURL(archivo);
    });
}

function cargarImagen(dataUrl) {
    return new Promise((resolve, reject) => {
        const imagen = new Image();
        imagen.onload = () => resolve(imagen);
        imagen.onerror = () => reject(new Error("El archivo seleccionado no es una imagen válida."));
        imagen.src = dataUrl;
    });
}

async function prepararImagen(dataUrl) {
    const imagen = await cargarImagen(dataUrl);
    const ladoMaximo = 640;
    const escala = Math.min(1, ladoMaximo / Math.max(imagen.naturalWidth, imagen.naturalHeight));

    canvas.width = Math.max(1, Math.round(imagen.naturalWidth * escala));
    canvas.height = Math.max(1, Math.round(imagen.naturalHeight * escala));
    canvas.getContext("2d").drawImage(imagen, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.82);
}

async function analizarImagen(fotoBase64) {
    if (analizando) {
        return;
    }

    analizando = true;
    loading.style.display = "flex";
    outputClass.textContent = "Analizando...";
    outputConfidence.textContent = "--";
    btnCapture.disabled = true;
    btnUpload.disabled = true;
    actualizarEstado("Analizando...", true);

    const controlador = new AbortController();
    const timeout = window.setTimeout(() => controlador.abort(), 45000);

    try {
        const respuesta = await fetch("/api/classify", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: fotoBase64 }),
            signal: controlador.signal,
        });

        const tipoContenido = respuesta.headers.get("content-type") || "";
        if (!tipoContenido.includes("application/json")) {
            throw new Error(
                "La API no está ejecutándose. Inicia el proyecto con “python run_local.py” y abre http://localhost:3000.",
            );
        }

        const resultado = await respuesta.json();
        if (!respuesta.ok) {
            throw new Error(resultado.error || `Error del servidor (${respuesta.status}).`);
        }
        if (typeof resultado.clase !== "string" || !Number.isFinite(resultado.confianza)) {
            throw new Error("La respuesta del modelo no tiene el formato esperado.");
        }

        const confianza = Math.max(0, Math.min(1, resultado.confianza));
        outputClass.textContent = resultado.clase;
        outputConfidence.textContent = `${(confianza * 100).toFixed(2)}%`;
        actualizarEstado("Análisis completo", true);
    } catch (error) {
        console.error("Error en la clasificación:", error);
        let mensaje = error.message;

        if (error.name === "AbortError") {
            mensaje = "El análisis tardó demasiado. Intenta nuevamente.";
        } else if (error instanceof TypeError) {
            mensaje = "No hay conexión con la API. Ejecuta “python run_local.py” y abre http://localhost:3000.";
        }

        outputClass.textContent = "No se pudo analizar";
        outputConfidence.textContent = "--";
        actualizarEstado("API no disponible");
        window.alert(mensaje);
    } finally {
        window.clearTimeout(timeout);
        loading.style.display = "none";
        analizando = false;
        btnCapture.disabled = !streamActivo;
        btnUpload.disabled = false;
    }
}

btnStart.addEventListener("click", async () => {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
        actualizarEstado("Cámara no disponible");
        window.alert("Abre la aplicación mediante HTTPS o exactamente desde http://localhost:3000.");
        return;
    }

    btnStart.disabled = true;
    imagePreview.style.display = "none";
    mostrarPlaceholder(true);
    actualizarEstado("Solicitando permiso...");

    try {
        streamActivo = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: { ideal: "environment" },
            },
            audio: false,
        });

        video.srcObject = streamActivo;
        await esperarVideoListo();
        await video.play();

        video.style.display = "block";
        mostrarPlaceholder(false);
        scanLine.classList.add("active");
        actualizarEstado("Cámara activa", true);
        btnStop.disabled = false;
        btnCapture.disabled = false;
    } catch (error) {
        console.error("Error al acceder a la cámara:", error);
        detenerCamara({ limpiarResultado: false });
        actualizarEstado("Sin acceso a la cámara");
        window.alert(mensajeErrorCamara(error));
    }
});

btnStop.addEventListener("click", () => detenerCamara());

btnCapture.addEventListener("click", async () => {
    if (!streamActivo || analizando || video.videoWidth === 0) {
        return;
    }

    const ladoMaximo = 640;
    const escala = Math.min(1, ladoMaximo / Math.max(video.videoWidth, video.videoHeight));
    canvas.width = Math.round(video.videoWidth * escala);
    canvas.height = Math.round(video.videoHeight * escala);
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    await analizarImagen(canvas.toDataURL("image/jpeg", 0.82));
});

btnUpload.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", async () => {
    const archivo = imageInput.files?.[0];
    if (!archivo) {
        return;
    }

    if (!archivo.type.startsWith("image/") || archivo.size > 10 * 1024 * 1024) {
        window.alert("Selecciona una imagen JPG, PNG o WebP de máximo 10 MB.");
        imageInput.value = "";
        return;
    }

    try {
        detenerCamara({ limpiarResultado: false });
        const dataUrl = await convertirArchivoADataUrl(archivo);
        imagePreview.src = dataUrl;
        imagePreview.style.display = "block";
        mostrarPlaceholder(false);
        actualizarEstado("Imagen lista", true);
        await analizarImagen(await prepararImagen(dataUrl));
    } catch (error) {
        console.error(error);
        window.alert(error.message);
    } finally {
        imageInput.value = "";
    }
});

window.addEventListener("pagehide", () => detenerCamara({ limpiarResultado: false }));
