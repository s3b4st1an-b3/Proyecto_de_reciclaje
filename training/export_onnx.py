"""Exporta el mejor checkpoint de PyTorch a un modelo ONNX candidato."""

import argparse
import json
import sys
from pathlib import Path

import onnx
import torch

from dataset_common import CLASES_DATASET
from train import (
    EVALUATION_RESIZE_MARGIN,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PREPROCESSING_MODE,
    crear_modelo,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path(__file__).with_name("artifacts") / "best_model.pt"
DEFAULT_OUTPUT = Path(__file__).with_name("artifacts") / "modelo_reciclaje.onnx"
DEFAULT_OPSET = 17


def cargar_checkpoint(ruta):
    checkpoint = torch.load(ruta, map_location="cpu", weights_only=False)
    requeridos = {
        "model_state_dict",
        "architecture",
        "classes",
        "display_names",
        "image_size",
        "normalization",
        "config",
    }
    faltantes = sorted(requeridos - checkpoint.keys())
    if faltantes:
        raise ValueError(
            f"El checkpoint no contiene: {', '.join(faltantes)}"
        )
    if checkpoint["architecture"] != "mobilenet_v3_small":
        raise ValueError(
            f"Arquitectura no soportada: {checkpoint['architecture']}"
        )
    if checkpoint["classes"] != list(CLASES_DATASET):
        raise ValueError(
            "El orden de clases del checkpoint no coincide con la API."
        )

    normalizacion = checkpoint["normalization"]
    if tuple(normalizacion["mean"]) != IMAGENET_MEAN:
        raise ValueError("La media de normalización del checkpoint no coincide.")
    if tuple(normalizacion["std"]) != IMAGENET_STD:
        raise ValueError(
            "La desviación de normalización del checkpoint no coincide."
        )

    image_size = int(checkpoint["image_size"])
    resize_size = int(
        checkpoint.get(
            "resize_size",
            image_size + EVALUATION_RESIZE_MARGIN,
        )
    )
    if resize_size <= image_size:
        raise ValueError("resize_size debe ser mayor que image_size.")
    checkpoint["resize_size"] = resize_size
    checkpoint["preprocessing"] = checkpoint.get(
        "preprocessing",
        PREPROCESSING_MODE,
    )
    checkpoint["input_layout"] = checkpoint.get("input_layout", "NCHW")
    checkpoint["color_space"] = checkpoint.get("color_space", "RGB")
    return checkpoint


def crear_modelo_desde_checkpoint(checkpoint):
    modelo = crear_modelo(checkpoint["config"]["dropout"])
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.eval()
    return modelo


def agregar_metadatos(ruta, checkpoint):
    modelo_onnx = onnx.load(ruta, load_external_data=True)
    metadatos = {
        "architecture": checkpoint["architecture"],
        "classes": json.dumps(checkpoint["classes"], ensure_ascii=False),
        "display_names": json.dumps(
            [
                checkpoint["display_names"][clase]
                for clase in checkpoint["classes"]
            ],
            ensure_ascii=False,
        ),
        "image_size": str(checkpoint["image_size"]),
        "resize_size": str(checkpoint["resize_size"]),
        "preprocessing": checkpoint["preprocessing"],
        "input_layout": checkpoint["input_layout"],
        "color_space": checkpoint["color_space"],
        "normalization_mean": json.dumps(list(IMAGENET_MEAN)),
        "normalization_std": json.dumps(list(IMAGENET_STD)),
        "output": "logits",
        "confidence_threshold": str(
            checkpoint["config"]["initial_confidence_threshold"]
        ),
    }
    onnx.helper.set_model_props(modelo_onnx, metadatos)

    onnx.checker.check_model(modelo_onnx, full_check=True)
    onnx.save_model(modelo_onnx, ruta)


def validar_forma_modelo(ruta, image_size):
    modelo = onnx.load(ruta, load_external_data=True)
    onnx.checker.check_model(modelo, full_check=True)

    if len(modelo.graph.input) != 1 or len(modelo.graph.output) != 1:
        raise ValueError("El ONNX debe tener exactamente una entrada y una salida.")

    salida = modelo.graph.output[0].type.tensor_type.shape.dim
    cantidad_clases = salida[-1].dim_value if salida else 0
    if cantidad_clases != len(CLASES_DATASET):
        raise ValueError(
            f"El ONNX devuelve {cantidad_clases} clases; "
            f"se esperaban {len(CLASES_DATASET)}."
        )

    entrada = modelo.graph.input[0].type.tensor_type.shape.dim
    dimensiones_fijas = [dimension.dim_value for dimension in entrada[1:]]
    esperadas = [3, image_size, image_size]
    if dimensiones_fijas != esperadas:
        raise ValueError(
            f"Entrada ONNX inesperada: {dimensiones_fijas}; se esperaba {esperadas}."
        )


def exportar(checkpoint_path, output_path, opset):
    checkpoint = cargar_checkpoint(checkpoint_path)
    modelo = crear_modelo_desde_checkpoint(checkpoint)
    image_size = int(checkpoint["image_size"])
    entrada_ejemplo = torch.randn(1, 3, image_size, image_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        modelo,
        entrada_ejemplo,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        dynamo=False,
    )
    agregar_metadatos(output_path, checkpoint)
    validar_forma_modelo(output_path, image_size)
    print(f"ONNX candidato exportado: {output_path.resolve()}")
    print(f"Entrada: [batch, 3, {image_size}, {image_size}]")
    print(f"Salida: [batch, {len(CLASES_DATASET)}] (logits)")
    print("Siguiente paso: ejecutar training/test_onnx.py")


def main():
    parser = argparse.ArgumentParser(
        description="Exporta best_model.pt a un ONNX candidato."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    argumentos = parser.parse_args()

    try:
        exportar(argumentos.checkpoint, argumentos.output, argumentos.opset)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
