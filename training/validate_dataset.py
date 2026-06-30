"""Audita estructura, imágenes dañadas y duplicados exactos del dataset."""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from dataset_common import (
    CLASES_DATASET,
    CONJUNTOS,
    EXTENSIONES_IMAGEN,
    calcular_hash,
    listar_imagenes,
    ruta_relativa,
)


def validar_imagen(ruta):
    try:
        contenido = ruta.read_bytes()
    except OSError as error:
        return f"No se pudo leer: {error}"

    if not contenido:
        return "El archivo está vacío"

    imagen = cv2.imdecode(np.frombuffer(contenido, dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagen is None:
        return "OpenCV no pudo decodificar la imagen"

    alto, ancho = imagen.shape[:2]
    if ancho < 32 or alto < 32:
        return f"Resolución demasiado pequeña: {ancho}x{alto}"

    return None


def buscar_archivos_no_admitidos(dataset):
    ignorados = {".gitkeep", "README.md"}
    return sorted(
        ruta
        for ruta in dataset.rglob("*")
        if ruta.is_file()
        and ruta.name not in ignorados
        and ruta.suffix.lower() not in EXTENSIONES_IMAGEN
    )


def auditar_dataset(dataset):
    errores_estructura = []
    advertencias = []
    conteos = {}
    imagenes = []

    for conjunto in CONJUNTOS:
        for clase in CLASES_DATASET:
            carpeta = dataset / conjunto / clase
            if not carpeta.is_dir():
                errores_estructura.append(
                    f"Falta la carpeta {ruta_relativa(carpeta, dataset)}"
                )
                conteos[(conjunto, clase)] = 0
                continue

            archivos = listar_imagenes(carpeta)
            conteos[(conjunto, clase)] = len(archivos)
            if not archivos:
                errores_estructura.append(
                    f"No hay imágenes en {ruta_relativa(carpeta, dataset)}"
                )
            imagenes.extend(archivos)

    desconocidas = listar_imagenes(dataset / "desconocido")
    imagenes.extend(desconocidas)

    danadas = []
    hashes = defaultdict(list)
    for ruta in imagenes:
        error = validar_imagen(ruta)
        if error:
            danadas.append((ruta, error))
            continue
        hashes[calcular_hash(ruta)].append(ruta)

    duplicados = [
        rutas for rutas in hashes.values()
        if len(rutas) > 1
    ]
    duplicados_desconocidos = [
        rutas for rutas in duplicados
        if all((dataset / "desconocido") in ruta.parents for ruta in rutas)
    ]
    duplicados_dataset = [
        rutas for rutas in duplicados
        if rutas not in duplicados_desconocidos
    ]

    totales_clase = {
        clase: sum(
            conteos[(conjunto, clase)]
            for conjunto in CONJUNTOS
        )
        for clase in CLASES_DATASET
    }
    minimo = min(totales_clase.values(), default=0)
    maximo = max(totales_clase.values(), default=0)
    if minimo and maximo > minimo * 1.5:
        advertencias.append(
            "El dataset está desbalanceado: "
            + ", ".join(
                f"{clase}={cantidad}"
                for clase, cantidad in totales_clase.items()
            )
            + ". El entrenamiento aplicará pesos por clase."
        )
    if duplicados_desconocidos:
        advertencias.append(
            f"Hay {len(duplicados_desconocidos)} grupo(s) de duplicados "
            "exactos en desconocido; se ignorarán al calibrar el umbral."
        )

    return {
        "errores_estructura": errores_estructura,
        "advertencias": advertencias,
        "conteos": conteos,
        "desconocidas": len(desconocidas),
        "danadas": danadas,
        "duplicados": duplicados_dataset,
        "duplicados_desconocidos": duplicados_desconocidos,
        "no_admitidos": buscar_archivos_no_admitidos(dataset),
    }


def imprimir_reporte(dataset, reporte):
    print(f"Dataset: {dataset.resolve()}")
    print("\nConteo por conjunto y categoría:")
    for conjunto in CONJUNTOS:
        print(f"\n  {conjunto}")
        for clase in CLASES_DATASET:
            print(f"    {clase}: {reporte['conteos'][(conjunto, clase)]}")
    print(f"\n  desconocido: {reporte['desconocidas']}")

    if reporte["errores_estructura"]:
        print("\nProblemas de estructura:")
        for mensaje in reporte["errores_estructura"]:
            print(f"  - {mensaje}")

    if reporte["advertencias"]:
        print("\nAdvertencias:")
        for mensaje in reporte["advertencias"]:
            print(f"  - {mensaje}")

    if reporte["danadas"]:
        print("\nImágenes dañadas o inválidas:")
        for ruta, error in reporte["danadas"]:
            print(f"  - {ruta_relativa(ruta, dataset)}: {error}")

    if reporte["duplicados"]:
        print("\nDuplicados exactos:")
        for numero, grupo in enumerate(reporte["duplicados"], start=1):
            print(f"  Grupo {numero}:")
            for ruta in grupo:
                print(f"    - {ruta_relativa(ruta, dataset)}")

    if reporte["no_admitidos"]:
        print("\nArchivos con formato no admitido:")
        for ruta in reporte["no_admitidos"]:
            print(f"  - {ruta_relativa(ruta, dataset)}")

    problemas = sum(
        (
            len(reporte["errores_estructura"]),
            len(reporte["danadas"]),
            len(reporte["duplicados"]),
            len(reporte["no_admitidos"]),
        )
    )
    if problemas:
        print(f"\nAuditoría terminada con {problemas} problema(s).")
    else:
        print("\nAuditoría terminada sin problemas.")

    return problemas


def main():
    parser = argparse.ArgumentParser(
        description="Valida estructura, imágenes y duplicados del dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset",
        help="Ruta de la carpeta dataset.",
    )
    argumentos = parser.parse_args()

    if not argumentos.dataset.is_dir():
        parser.error(f"No existe el dataset: {argumentos.dataset}")

    reporte = auditar_dataset(argumentos.dataset)
    problemas = imprimir_reporte(argumentos.dataset, reporte)
    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())
