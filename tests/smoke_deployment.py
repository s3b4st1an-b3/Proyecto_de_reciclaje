"""Comprueba una API local o desplegada usando únicamente la biblioteca estándar."""

import argparse
import base64
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def solicitar_json(request, timeout):
    try:
        with urlopen(request, timeout=timeout) as respuesta:
            contenido = respuesta.read().decode("utf-8")
            return respuesta.status, json.loads(contenido)
    except HTTPError as error:
        contenido = error.read().decode("utf-8")
        try:
            detalle = json.loads(contenido)
        except json.JSONDecodeError:
            detalle = {"error": contenido}
        return error.code, detalle


def comprobar_estado(endpoint, timeout):
    status, resultado = solicitar_json(Request(endpoint), timeout)
    if status != 200 or resultado.get("status") != "ok":
        raise RuntimeError(
            f"Diagnóstico fallido ({status}): "
            f"{json.dumps(resultado, ensure_ascii=False)}"
        )
    print(f"Diagnóstico correcto: {resultado['modelo']}")


def comprobar_imagen(endpoint, ruta, timeout):
    extension = ruta.suffix.lower()
    tipos = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    if extension not in tipos:
        raise ValueError("La imagen debe ser JPG, PNG o WebP.")

    contenido = base64.b64encode(ruta.read_bytes()).decode("ascii")
    payload = json.dumps({
        "image": f"data:{tipos[extension]};base64,{contenido}"
    }).encode("utf-8")
    request = Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    status, resultado = solicitar_json(request, timeout)
    requeridos = {
        "clase",
        "confianza",
        "es_confiable",
        "probabilidades",
        "segunda_opcion",
        "recomendacion",
    }
    if status != 200 or not requeridos.issubset(resultado):
        raise RuntimeError(
            f"Clasificación fallida ({status}): "
            f"{json.dumps(resultado, ensure_ascii=False)}"
        )
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description="Prueba rápidamente una API local o desplegada."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:3000",
        help="Origen del sitio, sin /api/classify.",
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--timeout", type=int, default=60)
    argumentos = parser.parse_args()

    endpoint = argumentos.base_url.rstrip("/") + "/api/classify"
    try:
        comprobar_estado(endpoint, argumentos.timeout)
        if argumentos.image:
            comprobar_imagen(endpoint, argumentos.image, argumentos.timeout)
    except (OSError, ValueError, RuntimeError, URLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
