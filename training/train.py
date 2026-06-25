"""Entrena MobileNetV3 Small para las cuatro categorías de reciclaje."""

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from dataset_common import CLASES_DATASET

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "dataset"
DEFAULT_CONFIG = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT = Path(__file__).with_name("artifacts")

CLASS_DISPLAY_NAMES = {
    "papel_carton": "Papel/Cartón",
    "vidrio": "Vidrio",
    "plastico": "Plástico",
    "organico": "Orgánico",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EVALUATION_RESIZE_MARGIN = 32
PREPROCESSING_MODE = "resize_shorter_side_center_crop"


class OrderedImageFolder(ImageFolder):
    """ImageFolder con un orden de clases fijo y compatible con la API."""

    def find_classes(self, directory):
        base = Path(directory)
        faltantes = [
            clase for clase in CLASES_DATASET
            if not (base / clase).is_dir()
        ]
        if faltantes:
            raise FileNotFoundError(
                f"Faltan categorías en {base}: {', '.join(faltantes)}"
            )

        clases_adicionales = sorted(
            ruta.name
            for ruta in base.iterdir()
            if ruta.is_dir() and ruta.name not in CLASES_DATASET
        )
        if clases_adicionales:
            raise ValueError(
                f"Hay categorías no configuradas en {base}: "
                f"{', '.join(clases_adicionales)}"
            )

        clases = list(CLASES_DATASET)
        return clases, {clase: indice for indice, clase in enumerate(clases)}


def cargar_configuracion(ruta):
    with ruta.open("r", encoding="utf-8") as archivo:
        config = json.load(archivo)

    requeridos = {
        "seed",
        "image_size",
        "batch_size",
        "num_workers",
        "epochs",
        "frozen_epochs",
        "learning_rate",
        "fine_tune_learning_rate",
        "weight_decay",
        "label_smoothing",
        "early_stopping_patience",
        "minimum_images_per_class",
        "dropout",
        "rotation_degrees",
        "color_jitter",
        "random_erasing_probability",
        "initial_confidence_threshold",
    }
    faltantes = sorted(requeridos - config.keys())
    if faltantes:
        raise ValueError(
            f"Faltan opciones en {ruta}: {', '.join(faltantes)}"
        )
    if config["frozen_epochs"] >= config["epochs"]:
        raise ValueError("frozen_epochs debe ser menor que epochs.")
    if not 0 < config["initial_confidence_threshold"] < 1:
        raise ValueError(
            "initial_confidence_threshold debe estar entre 0 y 1."
        )
    return config


def fijar_semilla(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def crear_transformaciones(config):
    image_size = config["image_size"]
    jitter = config["color_jitter"]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.75, 1.0),
            ratio=(0.8, 1.2),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(config["rotation_degrees"]),
        transforms.ColorJitter(
            brightness=jitter,
            contrast=jitter,
            saturation=jitter,
            hue=min(0.1, jitter / 2),
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(
            p=config["random_erasing_probability"],
            scale=(0.02, 0.12),
        ),
    ])

    evaluation_transform = transforms.Compose([
        transforms.Resize(image_size + EVALUATION_RESIZE_MARGIN),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, evaluation_transform


def crear_datasets(dataset_root, config):
    train_transform, evaluation_transform = crear_transformaciones(config)
    datasets = {
        "train": OrderedImageFolder(
            dataset_root / "train",
            transform=train_transform,
        ),
        "validation": OrderedImageFolder(
            dataset_root / "validation",
            transform=evaluation_transform,
        ),
        "test": OrderedImageFolder(
            dataset_root / "test",
            transform=evaluation_transform,
        ),
    }

    for nombre, dataset in datasets.items():
        if dataset.classes != list(CLASES_DATASET):
            raise ValueError(
                f"Orden inesperado de clases en {nombre}: {dataset.classes}"
            )
    return datasets


def validar_cantidad_imagenes(datasets, minimum):
    for nombre, dataset in datasets.items():
        conteos = Counter(dataset.targets)
        for indice, clase in enumerate(CLASES_DATASET):
            cantidad = conteos[indice]
            if cantidad == 0:
                raise ValueError(f"{nombre}/{clase} no contiene imágenes.")
            if nombre == "train" and cantidad < minimum:
                raise ValueError(
                    f"train/{clase} tiene {cantidad} imágenes; se requieren "
                    f"al menos {minimum}."
                )


def crear_dataloaders(datasets, config, device):
    usar_pin_memory = device.type == "cuda"
    generador = torch.Generator()
    generador.manual_seed(config["seed"])

    return {
        nombre: DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=nombre == "train",
            num_workers=config["num_workers"],
            pin_memory=usar_pin_memory,
            persistent_workers=config["num_workers"] > 0,
            generator=generador if nombre == "train" else None,
        )
        for nombre, dataset in datasets.items()
    }


def calcular_pesos_clase(targets):
    conteos = np.bincount(targets, minlength=len(CLASES_DATASET))
    total = conteos.sum()
    pesos = total / (len(CLASES_DATASET) * conteos)
    return torch.tensor(pesos, dtype=torch.float32)


def crear_modelo(dropout):
    weights = MobileNet_V3_Small_Weights.DEFAULT
    modelo = mobilenet_v3_small(weights=weights)
    entrada_clasificador = modelo.classifier[-1].in_features
    modelo.classifier[2].p = dropout
    modelo.classifier[-1] = nn.Linear(
        entrada_clasificador,
        len(CLASES_DATASET),
    )
    return modelo


def congelar_extractor(modelo, congelado):
    for parametro in modelo.features.parameters():
        parametro.requires_grad = not congelado


def calcular_metricas(matriz):
    total = int(matriz.sum())
    aciertos = int(np.trace(matriz))
    accuracy = aciertos / total if total else 0.0
    metricas_clase = {}
    f1_values = []

    for indice, clase in enumerate(CLASES_DATASET):
        tp = int(matriz[indice, indice])
        fp = int(matriz[:, indice].sum() - tp)
        fn = int(matriz[indice, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        metricas_clase[clase] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(matriz[indice, :].sum()),
        }

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "per_class": metricas_clase,
    }


def ejecutar_epoca(modelo, dataloader, criterion, device, optimizer=None):
    entrenando = optimizer is not None
    modelo.train(entrenando)
    if entrenando and not any(
        parametro.requires_grad for parametro in modelo.features.parameters()
    ):
        modelo.features.eval()
    perdida_total = 0.0
    matriz = np.zeros(
        (len(CLASES_DATASET), len(CLASES_DATASET)),
        dtype=np.int64,
    )

    contexto = torch.enable_grad() if entrenando else torch.inference_mode()
    with contexto:
        for imagenes, etiquetas in dataloader:
            imagenes = imagenes.to(device, non_blocking=True)
            etiquetas = etiquetas.to(device, non_blocking=True)

            if entrenando:
                optimizer.zero_grad(set_to_none=True)

            logits = modelo(imagenes)
            perdida = criterion(logits, etiquetas)

            if entrenando:
                perdida.backward()
                optimizer.step()

            perdida_total += perdida.item() * imagenes.size(0)
            predicciones = logits.argmax(dim=1)
            for real, prediccion in zip(
                etiquetas.detach().cpu().numpy(),
                predicciones.detach().cpu().numpy(),
            ):
                matriz[real, prediccion] += 1

    metricas = calcular_metricas(matriz)
    metricas["loss"] = perdida_total / len(dataloader.dataset)
    return metricas, matriz


def guardar_checkpoint(ruta, modelo, config, metricas, epoch):
    torch.save(
        {
            "model_state_dict": modelo.state_dict(),
            "architecture": "mobilenet_v3_small",
            "classes": list(CLASES_DATASET),
            "display_names": CLASS_DISPLAY_NAMES,
            "image_size": config["image_size"],
            "resize_size": config["image_size"] + EVALUATION_RESIZE_MARGIN,
            "preprocessing": PREPROCESSING_MODE,
            "input_layout": "NCHW",
            "color_space": "RGB",
            "normalization": {
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
            },
            "epoch": epoch,
            "validation_metrics": metricas,
            "config": config,
        },
        ruta,
    )


def guardar_historial(historial, ruta):
    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(historial, archivo, indent=2, ensure_ascii=False)


def guardar_matriz_confusion(matriz, ruta):
    etiquetas = [CLASS_DISPLAY_NAMES[clase] for clase in CLASES_DATASET]
    figura, eje = plt.subplots(figsize=(8, 7))
    imagen = eje.imshow(matriz, interpolation="nearest", cmap="Greens")
    figura.colorbar(imagen, ax=eje)
    eje.set(
        xticks=np.arange(len(etiquetas)),
        yticks=np.arange(len(etiquetas)),
        xticklabels=etiquetas,
        yticklabels=etiquetas,
        ylabel="Clase real",
        xlabel="Predicción",
        title="Matriz de confusión — conjunto de prueba",
    )
    plt.setp(eje.get_xticklabels(), rotation=35, ha="right")

    umbral = matriz.max() / 2 if matriz.size else 0
    for fila in range(matriz.shape[0]):
        for columna in range(matriz.shape[1]):
            eje.text(
                columna,
                fila,
                int(matriz[fila, columna]),
                ha="center",
                va="center",
                color="white" if matriz[fila, columna] > umbral else "black",
            )

    figura.tight_layout()
    figura.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(figura)


def imprimir_metricas(nombre, metricas):
    print(
        f"{nombre}: loss={metricas['loss']:.4f} "
        f"accuracy={metricas['accuracy']:.4f} "
        f"macro_f1={metricas['macro_f1']:.4f}"
    )


def entrenar(dataset_root, output_dir, config):
    fijar_semilla(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    datasets = crear_datasets(dataset_root, config)
    validar_cantidad_imagenes(
        datasets,
        config["minimum_images_per_class"],
    )
    dataloaders = crear_dataloaders(datasets, config, device)

    pesos = calcular_pesos_clase(datasets["train"].targets).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=pesos,
        label_smoothing=config["label_smoothing"],
    )

    modelo = crear_modelo(config["dropout"]).to(device)
    congelar_extractor(modelo, True)
    optimizer = torch.optim.AdamW(
        filter(lambda parametro: parametro.requires_grad, modelo.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    history_path = output_dir / "history.json"
    mejor_f1 = -math.inf
    epocas_sin_mejora = 0
    historial = []

    for epoch in range(1, config["epochs"] + 1):
        if epoch == config["frozen_epochs"] + 1:
            congelar_extractor(modelo, False)
            optimizer = torch.optim.AdamW(
                modelo.parameters(),
                lr=config["fine_tune_learning_rate"],
                weight_decay=config["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=2,
            )
            print("\nExtractor desbloqueado para ajuste fino.")

        print(f"\nÉpoca {epoch}/{config['epochs']}")
        train_metrics, _ = ejecutar_epoca(
            modelo,
            dataloaders["train"],
            criterion,
            device,
            optimizer,
        )
        validation_metrics, _ = ejecutar_epoca(
            modelo,
            dataloaders["validation"],
            criterion,
            device,
        )
        scheduler.step(validation_metrics["macro_f1"])
        imprimir_metricas("Entrenamiento", train_metrics)
        imprimir_metricas("Validación", validation_metrics)

        historial.append({
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        })
        guardar_historial(historial, history_path)

        if validation_metrics["macro_f1"] > mejor_f1:
            mejor_f1 = validation_metrics["macro_f1"]
            epocas_sin_mejora = 0
            guardar_checkpoint(
                checkpoint_path,
                modelo,
                config,
                validation_metrics,
                epoch,
            )
            print(f"Nuevo mejor modelo guardado: macro_f1={mejor_f1:.4f}")
        else:
            epocas_sin_mejora += 1

        if epocas_sin_mejora >= config["early_stopping_patience"]:
            print("\nEntrenamiento detenido por falta de mejora.")
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    modelo.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, test_matrix = ejecutar_epoca(
        modelo,
        dataloaders["test"],
        criterion,
        device,
    )
    imprimir_metricas("\nPrueba final", test_metrics)

    metrics_path = output_dir / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as archivo:
        json.dump(test_metrics, archivo, indent=2, ensure_ascii=False)
    np.savetxt(
        output_dir / "confusion_matrix.csv",
        test_matrix,
        delimiter=",",
        fmt="%d",
    )
    guardar_matriz_confusion(
        test_matrix,
        output_dir / "confusion_matrix.png",
    )

    print(f"\nArtefactos guardados en: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Entrena el clasificador de residuos con MobileNetV3 Small."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    argumentos = parser.parse_args()

    try:
        config = cargar_configuracion(argumentos.config)
        entrenar(argumentos.dataset, argumentos.output, config)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
