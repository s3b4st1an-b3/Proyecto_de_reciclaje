import base64
import io

import numpy as np
import pytest
from PIL import Image

import api.index as api_module
from api.categories import CLASES


class EntradaFalsa:
    name = "image"
    shape = ["batch_size", 3, 224, 224]
    type = "tensor(float)"


class SalidaFalsa:
    name = "logits"
    shape = ["batch_size", len(CLASES)]
    type = "tensor(float)"


class SesionFalsa:
    def __init__(self, logits):
        self.logits = np.asarray([logits], dtype=np.float32)
        self.ultimo_tensor = None

    def get_inputs(self):
        return [EntradaFalsa()]

    def get_outputs(self):
        return [SalidaFalsa()]

    def run(self, _salidas, entradas):
        self.ultimo_tensor = entradas["image"]
        return [self.logits]


@pytest.fixture
def configuracion_preprocesamiento():
    return {
        "identificadores_clase": [
            "papel_carton",
            "vidrio",
            "plastico",
            "organico",
        ],
        "categorias": list(CLASES),
        "image_size": 224,
        "resize_size": 256,
        "preprocessing": "resize_shorter_side_center_crop",
        "input_layout": "NCHW",
        "color_space": "RGB",
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "output": "logits",
        "confidence_threshold": 0.65,
    }


@pytest.fixture
def instalar_sesion(monkeypatch, configuracion_preprocesamiento):
    def _instalar(logits):
        sesion = SesionFalsa(logits)
        monkeypatch.setattr(api_module, "sesion_ia", sesion)
        monkeypatch.setattr(api_module, "modelo_revisado", True)
        monkeypatch.setattr(
            api_module,
            "preprocesamiento_modelo",
            configuracion_preprocesamiento,
        )
        monkeypatch.setattr(api_module, "error_modelo", None)
        monkeypatch.setattr(api_module, "umbral_modelo", 0.65)
        return sesion

    return _instalar


@pytest.fixture
def cliente():
    api_module.app.config.update(TESTING=True)
    return api_module.app.test_client()


def crear_data_url(color=(120, 80, 40), size=(640, 480), formato="JPEG"):
    imagen = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    imagen.save(buffer, format=formato)
    contenido = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/png" if formato == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{contenido}"


@pytest.fixture
def data_url_imagen():
    return crear_data_url()
