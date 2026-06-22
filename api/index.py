from flask import Flask, request, jsonify
import base64
import os
import numpy as np
import cv2
import onnxruntime as ort
import serverless_wsgi

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ruta_modelo = os.path.join(BASE_DIR, "modelo_reciclaje.onnx")
sesion_ia = None

def cargar_modelo():
    global sesion_ia
    if sesion_ia is None:
        try:
            sesion_ia = ort.InferenceSession(ruta_modelo)
            print("Modelo cargado correctamente")
        except Exception as e:
            print(f"Error cargando modelo: {e}")

cargar_modelo()

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

@app.route('/api/classify', methods=['POST', 'OPTIONS'])
def classify():

    if request.method == 'OPTIONS':
        return jsonify({}), 200

    if not sesion_ia:
        return jsonify({"error": "Modelo no disponible"}), 500

    try:
        data = request.json
        image_b64 = data.get('image')

        if not image_b64:
            return jsonify({"error": "No se recibió imagen"}), 400

        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        img_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        img_resized = cv2.resize(img, (224, 224))
        img_float = img_resized.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_float, (2, 0, 1))
        tensor_entrada = np.expand_dims(img_transposed, axis=0)

        nombre_entrada = sesion_ia.get_inputs()[0].name
        predicciones = sesion_ia.run(None, {nombre_entrada: tensor_entrada})[0]

        clase_id = int(np.argmax(predicciones))

        clases = ["Papel/Cartón", "Vidrio", "Plástico", "Orgánico"]

        return jsonify({
            "clase": clases[clase_id % 4],
            "confianza": float(np.max(predicciones))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def handler(event, context):
    return serverless_wsgi.handle_request(app, event, context)