import base64
import binascii
import json
import logging
import os
import threading

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge

from api.categories import (
    CLASE_NO_IDENTIFICADA,
    CLASES,
    IDENTIFICADORES_CLASE,
    UMBRAL_CONFIANZA,
    obtener_recomendacion,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "public"),
    static_url_path=""
)

app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024


API_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(API_DIR, "modelo_reciclaje.onnx")
PREPROCESAMIENTO_ESPERADO = "resize_shorter_side_center_crop"
LAYOUT_ESPERADO = "NCHW"
ESPACIO_COLOR_ESPERADO = "RGB"

sesion_ia = None
error_modelo = None
detalle_modelo = None
preprocesamiento_modelo = None
umbral_modelo = UMBRAL_CONFIANZA
modelo_revisado = False
bloqueo_modelo = threading.Lock()


@app.errorhandler(413)
def imagen_demasiado_grande(_error):
    return jsonify({
        "error": "La imagen enviada supera el límite permitido de 4 MB"
    }), 413


def leer_metadatos_modelo(sesion):
    metadatos = sesion.get_modelmeta().custom_metadata_map
    requeridos = {
        "classes",
        "display_names",
        "image_size",
        "resize_size",
        "preprocessing",
        "input_layout",
        "color_space",
        "normalization_mean",
        "normalization_std",
        "output",
    }
    faltantes = sorted(requeridos - metadatos.keys())
    if faltantes:
        raise ValueError(
            "El modelo no declara los metadatos requeridos: "
            f"{', '.join(faltantes)}."
        )

    try:
        clases_modelo = json.loads(metadatos["classes"])
        nombres_visibles = json.loads(metadatos["display_names"])
        medias = [float(valor) for valor in json.loads(
            metadatos["normalization_mean"]
        )]
        desviaciones = [float(valor) for valor in json.loads(
            metadatos["normalization_std"]
        )]
        image_size = int(metadatos["image_size"])
        resize_size = int(metadatos["resize_size"])
        confidence_threshold = float(
            metadatos.get("confidence_threshold", UMBRAL_CONFIANZA)
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            "Los metadatos de preprocesamiento del modelo no son válidos."
        ) from error

    if nombres_visibles != list(CLASES):
        raise ValueError(
            "El orden de categorías del modelo no coincide con la aplicación."
        )
    if clases_modelo != list(IDENTIFICADORES_CLASE):
        raise ValueError(
            "El orden de identificadores de clase no coincide con la aplicación."
        )
    if len(medias) != 3 or len(desviaciones) != 3:
        raise ValueError(
            "La normalización debe declarar tres medias y tres desviaciones."
        )
    if any(desviacion <= 0 for desviacion in desviaciones):
        raise ValueError("Las desviaciones de normalización deben ser positivas.")
    if image_size <= 0 or resize_size <= image_size:
        raise ValueError(
            "Los tamaños de preprocesamiento del modelo no son válidos."
        )
    if metadatos["preprocessing"] != PREPROCESAMIENTO_ESPERADO:
        raise ValueError("El método de preprocesamiento no es compatible.")
    if metadatos["input_layout"] != LAYOUT_ESPERADO:
        raise ValueError("El modelo debe utilizar una entrada NCHW.")
    if metadatos["color_space"] != ESPACIO_COLOR_ESPERADO:
        raise ValueError("El modelo debe utilizar imágenes RGB.")
    if metadatos["output"] != "logits":
        raise ValueError("El modelo debe devolver logits.")
    if not 0 < confidence_threshold < 1:
        raise ValueError("El umbral de confianza del modelo no es válido.")

    return {
        "identificadores_clase": clases_modelo,
        "categorias": nombres_visibles,
        "image_size": image_size,
        "resize_size": resize_size,
        "preprocessing": metadatos["preprocessing"],
        "input_layout": metadatos["input_layout"],
        "color_space": metadatos["color_space"],
        "normalization_mean": medias,
        "normalization_std": desviaciones,
        "output": metadatos["output"],
        "confidence_threshold": confidence_threshold,
    }


def obtener_detalle_modelo(sesion):
    entradas = sesion.get_inputs()
    salidas = sesion.get_outputs()

    if len(entradas) != 1:
        raise ValueError(
            f"El modelo debe tener una entrada, pero tiene {len(entradas)}."
        )

    if len(salidas) != 1:
        raise ValueError(
            f"El modelo debe tener una salida, pero tiene {len(salidas)}."
        )

    entrada = entradas[0]
    salida = salidas[0]
    forma_entrada = entrada.shape
    forma_salida = salida.shape
    cantidad_salidas = forma_salida[-1] if forma_salida else None

    if entrada.type != "tensor(float)":
        raise ValueError(
            f"El modelo requiere una entrada {entrada.type}; se esperaba "
            "tensor(float)."
        )
    if salida.type != "tensor(float)":
        raise ValueError(
            f"El modelo devuelve {salida.type}; se esperaba tensor(float)."
        )

    if not isinstance(cantidad_salidas, int):
        raise ValueError(
            "No fue posible determinar la cantidad de categorías del modelo."
        )

    if cantidad_salidas != len(CLASES):
        raise ValueError(
            f"El modelo devuelve {cantidad_salidas} categorías, pero la "
            f"aplicación requiere exactamente {len(CLASES)}."
        )

    configuracion = leer_metadatos_modelo(sesion)
    forma_esperada = [
        3,
        configuracion["image_size"],
        configuracion["image_size"],
    ]
    if len(forma_entrada) != 4 or forma_entrada[1:] != forma_esperada:
        raise ValueError(
            f"El modelo declara una entrada {forma_entrada}; se esperaba "
            f"[batch, {', '.join(map(str, forma_esperada))}]."
        )

    return {
        "entrada": {
            "nombre": entrada.name,
            "forma": entrada.shape,
            "tipo": entrada.type,
        },
        "salida": {
            "nombre": salida.name,
            "forma": salida.shape,
            "tipo": salida.type,
        },
        "categorias": list(CLASES),
        "preprocesamiento": configuracion,
    }


def cargar_modelo():
    """Carga y valida el modelo una sola vez por instancia serverless."""
    global sesion_ia, error_modelo, detalle_modelo
    global preprocesamiento_modelo, umbral_modelo, modelo_revisado

    if modelo_revisado:
        return sesion_ia

    with bloqueo_modelo:
        if modelo_revisado:
            return sesion_ia

        try:
            sesion = ort.InferenceSession(
                RUTA_MODELO,
                providers=["CPUExecutionProvider"],
            )
            detalle_modelo = obtener_detalle_modelo(sesion)
            preprocesamiento_modelo = detalle_modelo["preprocesamiento"]
            umbral_modelo = preprocesamiento_modelo["confidence_threshold"]
            sesion_ia = sesion
            error_modelo = None
        except Exception as error:
            sesion_ia = None
            detalle_modelo = None
            preprocesamiento_modelo = None
            umbral_modelo = UMBRAL_CONFIANZA
            error_modelo = str(error)
            logging.exception("No se pudo cargar el modelo ONNX")
        finally:
            modelo_revisado = True

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

    if probabilidades.size != len(CLASES):
        raise ValueError(
            f"El modelo devolvió {probabilidades.size} categorías durante la "
            f"predicción; se esperaban {len(CLASES)}."
        )

    indices_ordenados = np.argsort(probabilidades)[::-1]
    clase_id = int(indices_ordenados[0])
    segunda_clase_id = int(indices_ordenados[1])
    confianza = float(probabilidades[clase_id])
    es_confiable = confianza + 1e-7 >= umbral_modelo
    clase = CLASES[clase_id] if es_confiable else CLASE_NO_IDENTIFICADA

    return {
        "clase": clase,
        "confianza": confianza,
        "es_confiable": es_confiable,
        "probabilidades": {
            nombre_clase: float(probabilidades[indice])
            for indice, nombre_clase in enumerate(CLASES)
        },
        "segunda_opcion": {
            "clase": CLASES[segunda_clase_id],
            "confianza": float(probabilidades[segunda_clase_id]),
        },
        "recomendacion": obtener_recomendacion(clase),
    }


def redimensionar_lado_corto(imagen, lado_corto):
    alto, ancho = imagen.shape[:2]
    if alto <= 0 or ancho <= 0:
        raise ValueError("La imagen no tiene dimensiones válidas.")

    if ancho <= alto:
        nuevo_ancho = lado_corto
        nuevo_alto = int(lado_corto * alto / ancho)
    else:
        nuevo_alto = lado_corto
        nuevo_ancho = int(lado_corto * ancho / alto)
    imagen_pil = Image.fromarray(imagen)
    return np.asarray(
        imagen_pil.resize(
            (nuevo_ancho, nuevo_alto),
            resample=Image.Resampling.BILINEAR,
        )
    )


def recortar_centro(imagen, tamano):
    alto, ancho = imagen.shape[:2]
    if alto < tamano or ancho < tamano:
        raise ValueError("La imagen redimensionada es menor que el recorte.")

    inicio_y = int(round((alto - tamano) / 2.0))
    inicio_x = int(round((ancho - tamano) / 2.0))
    return imagen[
        inicio_y:inicio_y + tamano,
        inicio_x:inicio_x + tamano,
    ]


def preparar_tensor(imagen, configuracion):
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen_redimensionada = redimensionar_lado_corto(
        imagen_rgb,
        configuracion["resize_size"],
    )
    imagen_recortada = recortar_centro(
        imagen_redimensionada,
        configuracion["image_size"],
    )
    imagen_float = imagen_recortada.astype(np.float32) / 255.0
    medias = np.asarray(
        configuracion["normalization_mean"],
        dtype=np.float32,
    ).reshape(1, 1, 3)
    desviaciones = np.asarray(
        configuracion["normalization_std"],
        dtype=np.float32,
    ).reshape(1, 1, 3)
    imagen_normalizada = (imagen_float - medias) / desviaciones
    imagen_transpuesta = np.transpose(imagen_normalizada, (2, 0, 1))
    return np.expand_dims(imagen_transpuesta, axis=0).astype(np.float32)

from flask import send_from_directory


@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/classify", methods=["GET", "POST", "OPTIONS"])
def classify():
    if request.method == "GET":
        modelo = cargar_modelo()
        respuesta = {
            "status": "ok" if modelo is not None else "error",
            "modelo": "compatible" if modelo is not None else "incompatible",
            "categorias_esperadas": list(CLASES),
            "umbral_confianza": umbral_modelo,
            **({"detalle": error_modelo} if modelo is None else {}),
            **({"configuracion": detalle_modelo} if modelo is not None else {}),
        }
        return jsonify(respuesta), 200 if modelo is not None else 503

    if request.method == "OPTIONS":
        return jsonify({}), 200

    sesion = cargar_modelo()
    if sesion is None:
        return jsonify({
            "error": "El modelo de IA no es compatible con esta aplicación",
            "detalle": error_modelo,
            "categorias_esperadas": list(CLASES),
            "umbral_confianza": umbral_modelo,
        }), 503

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

        tensor_entrada = preparar_tensor(img, preprocesamiento_modelo)

        nombre_entrada = sesion.get_inputs()[0].name
        predicciones = sesion.run(None, {nombre_entrada: tensor_entrada})[0]
        resultado = interpretar_predicciones(predicciones)

        return jsonify(resultado)
    except binascii.Error:
        return jsonify({"error": "La imagen enviada no es válida"}), 400
    except RequestEntityTooLarge:
        raise
    except Exception:
        logging.exception("Error durante la clasificación")
        return jsonify({"error": "No fue posible clasificar la imagen"}), 500
