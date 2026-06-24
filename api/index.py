import base64
import binascii
import logging
import os
import threading

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(BASE_DIR, "modelo_reciclaje.onnx")
CLASES = ("Papel/Cartón", "Vidrio", "Plástico", "Orgánico")

sesion_ia = None
error_modelo = None
bloqueo_modelo = threading.Lock()


def cargar_modelo():
    """Carga el modelo una sola vez por instancia serverless."""
    global sesion_ia, error_modelo

    if sesion_ia is not None:
        return sesion_ia

    with bloqueo_modelo:
        if sesion_ia is not None:
            return sesion_ia

        try:
            sesion_ia = ort.InferenceSession(
                RUTA_MODELO,
                providers=["CPUExecutionProvider"],
            )
            error_modelo = None
        except Exception as error:
            error_modelo = str(error)
            logging.exception("No se pudo cargar el modelo ONNX")

    return sesion_ia


def softmax(valores):
    valores = valores - np.max(valores)
    exponenciales = np.exp(valores)
    return exponenciales / np.sum(exponenciales)


def convertir_a_probabilidades(predicciones):
    valores = np.asarray(predicciones, dtype=np.float32).reshape(-1)

    parecen_probabilidades = (
        np.all(valores >= 0)
        and np.all(valores <= 1)
        and np.isclose(float(np.sum(valores)), 1.0, atol=1e-3)
    )
    if parecen_probabilidades:
        return valores

    # Algunos modelos devuelven logits en vez de probabilidades.
    return softmax(valores)


def interpretar_predicciones(predicciones):
    probabilidades = convertir_a_probabilidades(predicciones)

    if probabilidades.size == len(CLASES):
        clase_id = int(np.argmax(probabilidades))
        return CLASES[clase_id], float(probabilidades[clase_id]), None

    # El modelo incluido actualmente devuelve 1000 salidas. Eso suele indicar un
    # modelo base tipo ImageNet, no un clasificador final entrenado con las 4
    # clases del proyecto. Para que la app no falle, convertimos la salida más
    # probable a una de las 4 categorías esperadas.
    indice_modelo = int(np.argmax(probabilidades))
    clase_id = indice_modelo % len(CLASES)
    return (
        CLASES[clase_id],
        float(probabilidades[indice_modelo]),
        f"El modelo devolvió {probabilidades.size} salidas; se mapeó la salida {indice_modelo} a {CLASES[clase_id]}.",
    )


@app.route("/api/classify", methods=["GET", "POST", "OPTIONS"])
def classify():
    if request.method == "GET":
        modelo = cargar_modelo()
        return jsonify({
            "status": "ok" if modelo is not None else "error",
            "modelo": "disponible" if modelo is not None else "no disponible",
            **({"detalle": error_modelo} if modelo is None else {}),
        }), 200 if modelo is not None else 503

    if request.method == "OPTIONS":
        return jsonify({}), 200

    sesion = cargar_modelo()
    if sesion is None:
        return jsonify({"error": "El modelo de IA no está disponible"}), 503

    try:
        data = request.get_json(silent=True)
        if not data or not isinstance(data.get("image"), str):
            return jsonify({"error": "No se recibió una imagen válida"}), 400

        image_b64 = data["image"]
        if "," in image_b64:
            encabezado, image_b64 = image_b64.split(",", 1)
            if not encabezado.startswith("data:image/"):
                return jsonify({"error": "El archivo enviado no es una imagen"}), 400

        img_bytes = base64.b64decode(image_b64, validate=True)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "No se pudo leer la imagen"}), 400

        img_resized = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_float, (2, 0, 1))
        tensor_entrada = np.expand_dims(img_transposed, axis=0).astype(np.float32)

        nombre_entrada = sesion.get_inputs()[0].name
        predicciones = sesion.run(None, {nombre_entrada: tensor_entrada})[0]
        clase, confianza, advertencia = interpretar_predicciones(predicciones)

        respuesta = {
            "clase": clase,
            "confianza": confianza,
        }
        if advertencia:
            respuesta["advertencia"] = advertencia

        return jsonify(respuesta)
    except binascii.Error:
        return jsonify({"error": "La imagen enviada no es válida"}), 400
    except Exception:
        logging.exception("Error durante la clasificación")
        return jsonify({"error": "No fue posible clasificar la imagen"}), 500
