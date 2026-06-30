import numpy as np
import pytest

import api.index as api_module
from api.categories import CLASES, UMBRAL_CONFIANZA
from api.index import interpretar_predicciones, preparar_tensor


@pytest.mark.parametrize(
    ("indice", "clase"),
    list(enumerate(CLASES)),
)
def test_identifica_cada_categoria(indice, clase):
    logits = np.full(len(CLASES), -4.0, dtype=np.float32)
    logits[indice] = 4.0

    resultado = interpretar_predicciones(logits)

    assert resultado["clase"] == clase
    assert resultado["es_confiable"] is True
    assert resultado["confianza"] > 0.99


def test_devuelve_no_identificado_bajo_el_umbral():
    probabilidades = np.array([0.40, 0.30, 0.20, 0.10], dtype=np.float32)

    resultado = interpretar_predicciones(probabilidades)

    assert resultado["clase"] == "No identificado"
    assert resultado["es_confiable"] is False
    assert resultado["confianza"] < UMBRAL_CONFIANZA
    assert resultado["segunda_opcion"] == {
        "clase": "Vidrio",
        "confianza": pytest.approx(0.30),
    }


def test_acepta_confianza_exactamente_en_el_umbral():
    restante = (1 - UMBRAL_CONFIANZA) / 3
    probabilidades = np.array(
        [UMBRAL_CONFIANZA, restante, restante, restante],
        dtype=np.float32,
    )

    resultado = interpretar_predicciones(probabilidades)

    assert resultado["clase"] == "Papel/Cartón"
    assert resultado["es_confiable"] is True


def test_utiliza_umbral_calibrado_del_modelo(monkeypatch):
    monkeypatch.setattr(api_module, "umbral_modelo", 0.80)
    probabilidades = np.array([0.70, 0.10, 0.10, 0.10], dtype=np.float32)

    resultado = interpretar_predicciones(probabilidades)

    assert resultado["clase"] == "No identificado"
    assert resultado["es_confiable"] is False


def test_rechaza_prediccion_con_cantidad_incorrecta_de_salidas():
    with pytest.raises(ValueError, match="esperaban 4"):
        interpretar_predicciones(np.zeros(1000, dtype=np.float32))


def test_preprocesamiento_produce_tensor_normalizado(
    configuracion_preprocesamiento,
):
    imagen_bgr = np.zeros((300, 500, 3), dtype=np.uint8)
    imagen_bgr[:, :] = (0, 0, 255)

    tensor = preparar_tensor(
        imagen_bgr,
        configuracion_preprocesamiento,
    )

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    assert tensor[0, 0].mean() == pytest.approx(
        (1.0 - 0.485) / 0.229,
        abs=1e-5,
    )
    assert tensor[0, 1].mean() == pytest.approx(
        (0.0 - 0.456) / 0.224,
        abs=1e-5,
    )
