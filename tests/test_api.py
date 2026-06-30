import base64

import numpy as np

from api.categories import CLASES
from tests.conftest import crear_data_url


def test_clasifica_imagen_valida_y_devuelve_contrato_completo(
    cliente,
    instalar_sesion,
    data_url_imagen,
):
    sesion = instalar_sesion([0.1, 0.2, 4.0, 0.0])

    respuesta = cliente.post("/api/classify", json={"image": data_url_imagen})

    assert respuesta.status_code == 200
    resultado = respuesta.get_json()
    assert resultado["clase"] == "Plástico"
    assert resultado["es_confiable"] is True
    assert resultado["segunda_opcion"]["clase"] == "Vidrio"
    assert set(resultado["probabilidades"]) == set(CLASES)
    assert isinstance(resultado["recomendacion"], str)
    assert sesion.ultimo_tensor.shape == (1, 3, 224, 224)
    assert sesion.ultimo_tensor.dtype == np.float32


def test_rechaza_solicitud_sin_imagen(cliente, instalar_sesion):
    instalar_sesion([4.0, 0.0, 0.0, 0.0])

    respuesta = cliente.post("/api/classify", json={})

    assert respuesta.status_code == 400
    assert "imagen" in respuesta.get_json()["error"].lower()


def test_rechaza_data_url_que_no_es_imagen(cliente, instalar_sesion):
    instalar_sesion([4.0, 0.0, 0.0, 0.0])
    contenido = base64.b64encode(b"texto").decode("ascii")

    respuesta = cliente.post(
        "/api/classify",
        json={"image": f"data:text/plain;base64,{contenido}"},
    )

    assert respuesta.status_code == 400
    assert "no es una imagen" in respuesta.get_json()["error"].lower()


def test_rechaza_base64_invalido(cliente, instalar_sesion):
    instalar_sesion([4.0, 0.0, 0.0, 0.0])

    respuesta = cliente.post(
        "/api/classify",
        json={"image": "data:image/jpeg;base64,%%%"},
    )

    assert respuesta.status_code == 400
    assert "válida" in respuesta.get_json()["error"].lower()


def test_rechaza_bytes_que_no_forman_imagen(cliente, instalar_sesion):
    instalar_sesion([4.0, 0.0, 0.0, 0.0])
    contenido = base64.b64encode(b"no soy una imagen").decode("ascii")

    respuesta = cliente.post(
        "/api/classify",
        json={"image": f"data:image/jpeg;base64,{contenido}"},
    )

    assert respuesta.status_code == 400
    assert "leer la imagen" in respuesta.get_json()["error"].lower()


def test_rechaza_carga_superior_a_cuatro_mb(cliente, instalar_sesion):
    instalar_sesion([4.0, 0.0, 0.0, 0.0])
    cuerpo = b'{"image":"' + (b"a" * (4 * 1024 * 1024)) + b'"}'

    respuesta = cliente.post(
        "/api/classify",
        data=cuerpo,
        content_type="application/json",
    )

    assert respuesta.status_code == 413
    assert respuesta.is_json
    assert "4 mb" in respuesta.get_json()["error"].lower()


def test_acepta_imagen_png(cliente, instalar_sesion):
    instalar_sesion([0.0, 4.0, 0.0, 0.0])

    respuesta = cliente.post(
        "/api/classify",
        json={"image": crear_data_url(formato="PNG")},
    )

    assert respuesta.status_code == 200
    assert respuesta.get_json()["clase"] == "Vidrio"
