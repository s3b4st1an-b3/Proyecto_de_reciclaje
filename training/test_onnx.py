"""Compara el modelo ONNX candidato contra PyTorch antes de instalarlo."""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from dataset_common import (
    CLASES_DATASET,
    calcular_hash,
    listar_imagenes,
)
from export_onnx import cargar_checkpoint, crear_modelo_desde_checkpoint
from train import crear_datasets, crear_transformaciones


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path(__file__).with_name("artifacts") / "best_model.pt"
DEFAULT_ONNX = Path(__file__).with_name("artifacts") / "modelo_reciclaje.onnx"
DEFAULT_DATASET = PROJECT_ROOT / "dataset"
PRODUCTION_MODEL = PROJECT_ROOT / "api" / "modelo_reciclaje.onnx"
BACKUP_DIR = Path(__file__).with_name("artifacts") / "backups"


def softmax_numpy(logits):
    desplazados = logits - np.max(logits, axis=1, keepdims=True)
    exponenciales = np.exp(desplazados)
    return exponenciales / np.sum(exponenciales, axis=1, keepdims=True)


def leer_metadatos_onnx(ruta):
    modelo = onnx.load(ruta, load_external_data=True)
    onnx.checker.check_model(modelo, full_check=True)
    return {propiedad.key: propiedad.value for propiedad in modelo.metadata_props}


def validar_metadatos(metadatos, checkpoint):
    clases = json.loads(metadatos.get("classes", "[]"))
    if clases != list(CLASES_DATASET):
        raise ValueError("Las clases guardadas en el ONNX no coinciden.")
    nombres = json.loads(metadatos.get("display_names", "[]"))
    nombres_esperados = [
        checkpoint["display_names"][clase]
        for clase in checkpoint["classes"]
    ]
    if nombres != nombres_esperados:
        raise ValueError("Los nombres visibles del ONNX no coinciden.")
    if int(metadatos.get("image_size", 0)) != checkpoint["image_size"]:
        raise ValueError("El tamaño de imagen del ONNX no coincide.")
    if int(metadatos.get("resize_size", 0)) != checkpoint["resize_size"]:
        raise ValueError("El tamaño de redimensión del ONNX no coincide.")
    if metadatos.get("preprocessing") != checkpoint["preprocessing"]:
        raise ValueError("El método de preprocesamiento del ONNX no coincide.")
    if metadatos.get("input_layout") != "NCHW":
        raise ValueError("El ONNX no declara entrada NCHW.")
    if metadatos.get("color_space") != "RGB":
        raise ValueError("El ONNX no declara imágenes RGB.")
    if metadatos.get("output") != "logits":
        raise ValueError("El ONNX no declara una salida de logits.")
    confidence_threshold = float(
        metadatos.get(
            "confidence_threshold",
            checkpoint["config"].get("initial_confidence_threshold", 0.65),
        )
    )
    if not 0 < confidence_threshold < 1:
        raise ValueError("El ONNX no declara un umbral de confianza válido.")
    mean = tuple(json.loads(metadatos.get("normalization_mean", "[]")))
    std = tuple(json.loads(metadatos.get("normalization_std", "[]")))
    normalizacion = checkpoint["normalization"]
    if mean != tuple(normalizacion["mean"]):
        raise ValueError("La media de normalización del ONNX no coincide.")
    if std != tuple(normalizacion["std"]):
        raise ValueError("La desviación de normalización del ONNX no coincide.")


def seleccionar_muestras(dataset, max_samples):
    if not dataset.samples:
        raise ValueError("El conjunto de prueba no contiene imágenes.")
    cantidad = min(max_samples, len(dataset))
    indices = np.linspace(
        0,
        len(dataset) - 1,
        num=cantidad,
        dtype=int,
    )
    return [dataset[int(indice)] for indice in indices]


def deduplicar_rutas(rutas):
    unicas = []
    hashes = set()
    for ruta in rutas:
        digest = calcular_hash(ruta)
        if digest in hashes:
            continue
        hashes.add(digest)
        unicas.append(ruta)
    return unicas


def dividir_desconocidas(dataset_root, transform, seed):
    rutas = deduplicar_rutas(listar_imagenes(dataset_root / "desconocido"))
    if len(rutas) < 20:
        raise ValueError(
            "Se requieren al menos 20 imágenes desconocidas únicas para "
            "calibrar y evaluar el umbral."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(rutas)
    mitad = len(rutas) // 2

    def cargar(grupo):
        tensores = []
        for ruta in grupo:
            with Image.open(ruta) as imagen:
                tensores.append(transform(imagen.convert("RGB")))
        return tensores

    return cargar(rutas[:mitad]), cargar(rutas[mitad:]), len(rutas)


def inferir_onnx(sesion, tensores, batch_size=32):
    entrada = sesion.get_inputs()[0].name
    salida = sesion.get_outputs()[0].name
    resultados = []
    for inicio in range(0, len(tensores), batch_size):
        lote = torch.stack(tensores[inicio:inicio + batch_size])
        logits = sesion.run(
            [salida],
            {entrada: lote.numpy().astype(np.float32)},
        )[0]
        resultados.append(softmax_numpy(logits))
    return np.concatenate(resultados, axis=0)


def inferir_dataset_onnx(sesion, dataset):
    tensores = [dataset[indice][0] for indice in range(len(dataset))]
    etiquetas = np.array(dataset.targets, dtype=np.int64)
    return inferir_onnx(sesion, tensores), etiquetas


def seleccionar_umbral(probabilidades_conocidas, etiquetas, probabilidades_desconocidas):
    confianza_conocida = probabilidades_conocidas.max(axis=1)
    predicciones = probabilidades_conocidas.argmax(axis=1)
    aciertos = predicciones == etiquetas
    confianza_desconocida = probabilidades_desconocidas.max(axis=1)

    candidatos = np.arange(0.35, 0.951, 0.01)
    mejor = None
    for umbral in candidatos:
        conocidos_correctos_aceptados = np.mean(
            aciertos & (confianza_conocida >= umbral)
        )
        desconocidos_rechazados = np.mean(
            confianza_desconocida < umbral
        )
        puntuacion = (
            conocidos_correctos_aceptados + desconocidos_rechazados
        ) / 2
        candidato = (
            float(puntuacion),
            float(conocidos_correctos_aceptados),
            float(desconocidos_rechazados),
            -float(umbral),
        )
        if mejor is None or candidato > mejor:
            mejor = candidato

    return round(-mejor[3], 2), {
        "balanced_score": mejor[0],
        "known_correct_acceptance_rate": mejor[1],
        "unknown_rejection_rate": mejor[2],
    }


def evaluar_umbral(probabilidades_conocidas, etiquetas, probabilidades_desconocidas, umbral):
    confianza_conocida = probabilidades_conocidas.max(axis=1)
    predicciones = probabilidades_conocidas.argmax(axis=1)
    aciertos = predicciones == etiquetas
    aceptadas = confianza_conocida >= umbral
    confianza_desconocida = probabilidades_desconocidas.max(axis=1)

    return {
        "threshold": umbral,
        "known_samples": int(len(etiquetas)),
        "known_raw_accuracy": float(np.mean(aciertos)),
        "known_correct_acceptance_rate": float(np.mean(aciertos & aceptadas)),
        "known_acceptance_rate": float(np.mean(aceptadas)),
        "known_accuracy_when_accepted": (
            float(np.mean(aciertos[aceptadas]))
            if np.any(aceptadas)
            else 0.0
        ),
        "unknown_samples": int(len(probabilidades_desconocidas)),
        "unknown_rejection_rate": float(
            np.mean(confianza_desconocida < umbral)
        ),
    }


def actualizar_umbral_onnx(ruta, umbral):
    modelo = onnx.load(ruta, load_external_data=True)
    metadatos = {
        propiedad.key: propiedad.value
        for propiedad in modelo.metadata_props
    }
    metadatos["confidence_threshold"] = str(umbral)
    onnx.helper.set_model_props(modelo, metadatos)
    onnx.checker.check_model(modelo, full_check=True)
    onnx.save_model(modelo, ruta)


def comparar_modelos(checkpoint_path, onnx_path, dataset_root, max_samples):
    checkpoint = cargar_checkpoint(checkpoint_path)
    modelo_torch = crear_modelo_desde_checkpoint(checkpoint)
    metadatos = leer_metadatos_onnx(onnx_path)
    validar_metadatos(metadatos, checkpoint)

    datasets = crear_datasets(dataset_root, checkpoint["config"])
    muestras = seleccionar_muestras(datasets["test"], max_samples)
    imagenes = torch.stack([imagen for imagen, _ in muestras])
    etiquetas = np.array([etiqueta for _, etiqueta in muestras])

    with torch.inference_mode():
        logits_torch = modelo_torch(imagenes).cpu().numpy()

    sesion = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    entradas = sesion.get_inputs()
    salidas = sesion.get_outputs()
    if len(entradas) != 1 or len(salidas) != 1:
        raise ValueError("ONNX Runtime detectó una forma de E/S inesperada.")
    if salidas[0].shape[-1] != len(CLASES_DATASET):
        raise ValueError(
            f"ONNX Runtime detectó {salidas[0].shape[-1]} salidas."
        )

    logits_onnx = sesion.run(
        [salidas[0].name],
        {entradas[0].name: imagenes.numpy().astype(np.float32)},
    )[0]

    diferencia_maxima = float(np.max(np.abs(logits_torch - logits_onnx)))
    probabilidades_torch = softmax_numpy(logits_torch)
    probabilidades_onnx = softmax_numpy(logits_onnx)
    diferencia_probabilidad = float(
        np.max(np.abs(probabilidades_torch - probabilidades_onnx))
    )
    predicciones_torch = np.argmax(logits_torch, axis=1)
    predicciones_onnx = np.argmax(logits_onnx, axis=1)
    coincidencias = int(np.sum(predicciones_torch == predicciones_onnx))
    accuracy_onnx = float(np.mean(predicciones_onnx == etiquetas))

    tolerancia_logits = 1e-4
    tolerancia_probabilidades = 1e-5
    if diferencia_maxima > tolerancia_logits:
        raise ValueError(
            f"Diferencia de logits demasiado alta: {diferencia_maxima:.8f}"
        )
    if diferencia_probabilidad > tolerancia_probabilidades:
        raise ValueError(
            "Diferencia de probabilidades demasiado alta: "
            f"{diferencia_probabilidad:.8f}"
        )
    if coincidencias != len(muestras):
        raise ValueError(
            f"Solo coincidieron {coincidencias}/{len(muestras)} predicciones."
        )

    _, evaluation_transform = crear_transformaciones(checkpoint["config"])
    desconocidas_calibracion, desconocidas_evaluacion, total_desconocidas = (
        dividir_desconocidas(
            dataset_root,
            evaluation_transform,
            checkpoint["config"]["seed"],
        )
    )
    probabilidades_validation, etiquetas_validation = inferir_dataset_onnx(
        sesion,
        datasets["validation"],
    )
    probabilidades_desconocidas_calibracion = inferir_onnx(
        sesion,
        desconocidas_calibracion,
    )
    umbral, metricas_calibracion = seleccionar_umbral(
        probabilidades_validation,
        etiquetas_validation,
        probabilidades_desconocidas_calibracion,
    )

    probabilidades_test, etiquetas_test = inferir_dataset_onnx(
        sesion,
        datasets["test"],
    )
    probabilidades_desconocidas_evaluacion = inferir_onnx(
        sesion,
        desconocidas_evaluacion,
    )
    metricas_umbral = evaluar_umbral(
        probabilidades_test,
        etiquetas_test,
        probabilidades_desconocidas_evaluacion,
        umbral,
    )
    actualizar_umbral_onnx(onnx_path, umbral)

    reporte = {
        "samples": len(muestras),
        "matching_predictions": coincidencias,
        "maximum_logit_difference": diferencia_maxima,
        "maximum_probability_difference": diferencia_probabilidad,
        "onnx_accuracy_on_samples": accuracy_onnx,
        "classes": list(CLASES_DATASET),
        "unknown_unique_images": total_desconocidas,
        "threshold_calibration": {
            "known_validation_samples": len(etiquetas_validation),
            "unknown_calibration_samples": len(desconocidas_calibracion),
            "selected_threshold": umbral,
            **metricas_calibracion,
        },
        "threshold_evaluation": metricas_umbral,
        "input": {
            "name": entradas[0].name,
            "shape": entradas[0].shape,
            "type": entradas[0].type,
        },
        "output": {
            "name": salidas[0].name,
            "shape": salidas[0].shape,
            "type": salidas[0].type,
        },
    }
    return reporte


def instalar_modelo(candidato, destino):
    destino.parent.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")

    if destino.exists():
        respaldo = BACKUP_DIR / f"{destino.stem}_{marca}.onnx"
        shutil.copy2(destino, respaldo)
        print(f"Respaldo creado: {respaldo}")

    datos_externos = destino.with_suffix(destino.suffix + ".data")
    if datos_externos.exists():
        respaldo_datos = BACKUP_DIR / f"{destino.stem}_{marca}.onnx.data"
        shutil.copy2(datos_externos, respaldo_datos)
        datos_externos.unlink()
        print(f"Pesos externos respaldados: {respaldo_datos}")

    shutil.copy2(candidato, destino)
    print(f"Modelo instalado en: {destino}")


def main():
    parser = argparse.ArgumentParser(
        description="Compara PyTorch y ONNX antes de instalar el modelo."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument(
        "--install",
        action="store_true",
        help="Instala el ONNX validado en api/modelo_reciclaje.onnx.",
    )
    argumentos = parser.parse_args()

    if argumentos.max_samples < 1:
        parser.error("--max-samples debe ser mayor que cero.")

    try:
        reporte = comparar_modelos(
            argumentos.checkpoint,
            argumentos.onnx,
            argumentos.dataset,
            argumentos.max_samples,
        )
        reporte_path = argumentos.onnx.with_suffix(".validation.json")
        with reporte_path.open("w", encoding="utf-8") as archivo:
            json.dump(reporte, archivo, indent=2, ensure_ascii=False)

        print(json.dumps(reporte, indent=2, ensure_ascii=False))
        print(f"\nValidación superada. Reporte: {reporte_path.resolve()}")
        if argumentos.install:
            instalar_modelo(argumentos.onnx, PRODUCTION_MODEL)
        else:
            print(
                "El modelo de producción no fue modificado. "
                "Usa --install cuando quieras instalar este candidato."
            )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
