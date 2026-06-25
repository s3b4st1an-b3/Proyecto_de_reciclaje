"""Divide imágenes por categoría sin mover ni borrar los archivos originales."""

import argparse
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from dataset_common import (
    CLASES_DATASET,
    CONJUNTOS,
    calcular_hash,
    listar_imagenes,
)

BACKUP_ROOT = Path(__file__).with_name("artifacts") / "backups"


def calcular_cantidades(total, proporcion_train, proporcion_validation):
    proporciones = (
        proporcion_train,
        proporcion_validation,
        1 - proporcion_train - proporcion_validation,
    )
    cantidades = [1, 1, 1]
    ideales = [total * proporcion for proporcion in proporciones]

    for _ in range(total - 3):
        indice = max(
            range(3),
            key=lambda posicion: ideales[posicion] - cantidades[posicion],
        )
        cantidades[indice] += 1

    return tuple(cantidades)


def comprobar_destino_vacio(destino):
    existentes = []
    for conjunto in CONJUNTOS:
        for clase in CLASES_DATASET:
            existentes.extend(listar_imagenes(destino / conjunto / clase))
    return existentes


def filtrar_calidad(imagenes_por_clase, lado_corto_minimo):
    filtradas = {}
    invalidas = []
    pequenas = []

    for clase, imagenes in imagenes_por_clase.items():
        aceptadas = []
        for ruta in imagenes:
            contenido = np.fromfile(ruta, dtype=np.uint8)
            imagen = cv2.imdecode(contenido, cv2.IMREAD_COLOR)
            if imagen is None:
                invalidas.append((clase, ruta))
                continue

            alto, ancho = imagen.shape[:2]
            if min(alto, ancho) < lado_corto_minimo:
                pequenas.append((clase, ruta, ancho, alto))
                continue
            aceptadas.append(ruta)
        filtradas[clase] = aceptadas

    return filtradas, invalidas, pequenas


def deduplicar_imagenes(imagenes_por_clase):
    hashes = defaultdict(list)
    for clase, imagenes in imagenes_por_clase.items():
        for ruta in imagenes:
            hashes[calcular_hash(ruta)].append((clase, ruta))

    ambiguas = []
    duplicadas = []
    conservar = {
        clase: []
        for clase in CLASES_DATASET
    }

    for coincidencias in hashes.values():
        clases = {clase for clase, _ruta in coincidencias}
        if len(clases) > 1:
            ambiguas.append(coincidencias)
            continue

        clase = coincidencias[0][0]
        conservar[clase].append(coincidencias[0][1])
        duplicadas.extend(coincidencias[1:])

    for clase in conservar:
        conservar[clase].sort()

    return conservar, duplicadas, ambiguas


def nombre_destino(indice, origen):
    return f"{indice:06d}{origen.suffix.lower()}"


def dividir_clase(imagenes, rng, proporcion_train, proporcion_validation):
    imagenes = list(imagenes)
    rng.shuffle(imagenes)
    cantidad_train, cantidad_validation, _ = calcular_cantidades(
        len(imagenes),
        proporcion_train,
        proporcion_validation,
    )

    fin_validation = cantidad_train + cantidad_validation
    return {
        "train": imagenes[:cantidad_train],
        "validation": imagenes[cantidad_train:fin_validation],
        "test": imagenes[fin_validation:],
    }


def copiar_division(destino, clase, division):
    conteos = {}
    for conjunto, imagenes in division.items():
        carpeta_destino = destino / conjunto / clase
        carpeta_destino.mkdir(parents=True, exist_ok=True)

        for indice, origen in enumerate(imagenes, start=1):
            destino_imagen = carpeta_destino / nombre_destino(indice, origen)
            shutil.copy2(origen, destino_imagen)

        conteos[conjunto] = len(imagenes)
    return conteos


def activar_nueva_division(staging, destino):
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    respaldo = BACKUP_ROOT / f"dataset_split_{marca}"
    respaldo.mkdir(parents=True, exist_ok=False)
    movidos = []

    completado = False
    try:
        destino.mkdir(parents=True, exist_ok=True)
        for conjunto in CONJUNTOS:
            actual = destino / conjunto
            if actual.exists():
                destino_respaldo = respaldo / conjunto
                shutil.move(str(actual), str(destino_respaldo))
                movidos.append((destino_respaldo, actual))

        for conjunto in CONJUNTOS:
            shutil.move(
                str(staging / conjunto),
                str(destino / conjunto),
            )
        completado = True
    except Exception:
        staging.mkdir(parents=True, exist_ok=True)
        for conjunto in CONJUNTOS:
            nueva = destino / conjunto
            temporal = staging / conjunto
            if nueva.exists() and not temporal.exists():
                shutil.move(str(nueva), str(temporal))
        for respaldo_conjunto, destino_original in reversed(movidos):
            if respaldo_conjunto.exists() and not destino_original.exists():
                shutil.move(
                    str(respaldo_conjunto),
                    str(destino_original),
                )
        raise
    finally:
        if completado and staging.exists():
            shutil.rmtree(staging)

    return respaldo


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copia imágenes desde carpetas por categoría y crea una división "
            "reproducible de entrenamiento, validación y prueba."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help=(
            "Carpeta fuente que contiene papel_carton, vidrio, plastico y "
            "organico."
        ),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset",
        help="Carpeta dataset de destino.",
    )
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--validation", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--min-short-side",
        type=int,
        default=128,
        help="Descarta imágenes cuyo lado corto sea menor a este valor.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Reconstruye train/validation/test y respalda la división anterior. "
            "No modifica dataset/desconocido."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra la división resultante sin copiar ni reemplazar archivos.",
    )
    argumentos = parser.parse_args()

    proporcion_test = 1 - argumentos.train - argumentos.validation
    if (
        argumentos.train <= 0
        or argumentos.validation <= 0
        or proporcion_test <= 0
    ):
        parser.error("Las proporciones deben ser positivas y sumar menos de 1.")
    if argumentos.min_short_side < 32:
        parser.error("--min-short-side debe ser al menos 32.")

    fuente = argumentos.source.resolve()
    destino = argumentos.destination.resolve()
    if not fuente.is_dir():
        parser.error(f"No existe la carpeta fuente: {fuente}")
    if (
        fuente == destino
        or destino in fuente.parents
        or fuente in destino.parents
    ):
        parser.error(
            "La fuente y el destino deben ser carpetas separadas y no pueden "
            "estar anidadas entre sí."
        )

    existentes = comprobar_destino_vacio(destino)
    if existentes and not argumentos.replace:
        parser.error(
            "El destino ya contiene imágenes. Usa --replace para reconstruir "
            "la división con respaldo automático."
        )

    imagenes_por_clase = {}
    for clase in CLASES_DATASET:
        carpeta = fuente / clase
        imagenes = listar_imagenes(carpeta)
        if not carpeta.is_dir():
            parser.error(f"Falta la categoría fuente: {carpeta}")
        if len(imagenes) < 3:
            parser.error(
                f"La categoría {clase} necesita al menos 3 imágenes; "
                f"se encontraron {len(imagenes)}."
            )
        imagenes_por_clase[clase] = imagenes

    imagenes_por_clase, invalidas, pequenas = filtrar_calidad(
        imagenes_por_clase,
        argumentos.min_short_side,
    )
    if invalidas:
        print("Se encontraron imágenes que no pudieron decodificarse:")
        for clase, ruta in invalidas:
            print(f"  - [{clase}] {ruta}")
        parser.error("Corrige o elimina las imágenes inválidas.")

    imagenes_por_clase, duplicadas, ambiguas = deduplicar_imagenes(
        imagenes_por_clase
    )
    if ambiguas:
        print("Se encontraron imágenes idénticas en categorías diferentes:")
        for numero, grupo in enumerate(ambiguas, start=1):
            print(f"\n  Grupo {numero}:")
            for clase, ruta in grupo:
                print(f"    - [{clase}] {ruta}")
        print("\nEstas imágenes ambiguas serán descartadas de todas las clases.")

    for clase, imagenes in imagenes_por_clase.items():
        if len(imagenes) < 3:
            parser.error(
                f"La categoría {clase} quedó con menos de 3 imágenes válidas."
            )

    staging = (
        BACKUP_ROOT / "dataset_staging"
        if argumentos.replace
        else destino
    )
    if argumentos.replace and not argumentos.dry_run:
        if staging.exists():
            parser.error(
                f"Ya existe una preparación incompleta: {staging}. "
                "Revísala antes de volver a ejecutar."
            )
        staging.mkdir(parents=True)

    rng = random.Random(argumentos.seed)
    print(f"Fuente: {fuente}")
    print(f"Destino: {destino}")
    print(f"Semilla: {argumentos.seed}\n")
    print(f"Imágenes pequeñas descartadas: {len(pequenas)}")
    print(f"Duplicados internos descartados: {len(duplicadas)}\n")
    print(
        "Archivos con etiqueta ambigua descartados: "
        f"{sum(len(grupo) for grupo in ambiguas)}\n"
    )

    for clase, imagenes in imagenes_por_clase.items():
        division = dividir_clase(
            imagenes,
            rng,
            argumentos.train,
            argumentos.validation,
        )
        conteos = {
            conjunto: len(rutas)
            for conjunto, rutas in division.items()
        }
        if not argumentos.dry_run:
            copiar_division(staging, clase, division)
        print(
            f"{clase}: train={conteos['train']}, "
            f"validation={conteos['validation']}, test={conteos['test']}"
        )

    if argumentos.dry_run:
        print("\nSimulación completada. No se modificó ningún archivo.")
        return 0

    if argumentos.replace:
        respaldo = activar_nueva_division(staging, destino)
        print(f"\nDivisión anterior respaldada en: {respaldo.resolve()}")

    print("División completada. Los archivos originales no fueron modificados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
