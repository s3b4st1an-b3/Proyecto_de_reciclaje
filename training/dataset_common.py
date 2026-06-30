"""Reglas compartidas por las herramientas de preparación del dataset."""

import hashlib
from pathlib import Path

EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".webp"}
CLASES_DATASET = ("papel_carton", "vidrio", "plastico", "organico")
CONJUNTOS = ("train", "validation", "test")


def es_imagen_admitida(ruta):
    return ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN


def listar_imagenes(directorio):
    if not directorio.exists():
        return []

    return sorted(
        ruta
        for ruta in directorio.rglob("*")
        if es_imagen_admitida(ruta)
    )


def ruta_relativa(ruta, base):
    try:
        return str(ruta.relative_to(base))
    except ValueError:
        return str(ruta)


def calcular_hash(ruta):
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()
