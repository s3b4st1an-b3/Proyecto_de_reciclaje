import sys
from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING_DIR))

import split_dataset as split_module
from dataset_common import CLASES_DATASET, CONJUNTOS


def crear_archivo(ruta, contenido):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(contenido)
    return ruta


def test_deduplicar_descarta_copias_y_etiquetas_ambiguas(tmp_path):
    imagenes = {
        clase: []
        for clase in CLASES_DATASET
    }
    imagenes["papel_carton"] = [
        crear_archivo(tmp_path / "papel_1.jpg", b"papel"),
        crear_archivo(tmp_path / "papel_2.jpg", b"papel"),
    ]
    imagenes["vidrio"] = [
        crear_archivo(tmp_path / "vidrio.jpg", b"ambigua"),
    ]
    imagenes["plastico"] = [
        crear_archivo(tmp_path / "plastico.jpg", b"ambigua"),
    ]
    imagenes["organico"] = [
        crear_archivo(tmp_path / "organico.jpg", b"organico"),
    ]

    limpias, duplicadas, ambiguas = split_module.deduplicar_imagenes(imagenes)

    assert len(limpias["papel_carton"]) == 1
    assert len(duplicadas) == 1
    assert limpias["vidrio"] == []
    assert limpias["plastico"] == []
    assert len(ambiguas) == 1
    assert len(ambiguas[0]) == 2


def test_reemplazo_conserva_desconocido_y_respalda_division(
    tmp_path,
    monkeypatch,
):
    destino = tmp_path / "dataset"
    staging = tmp_path / "staging"
    backup_root = tmp_path / "backups"
    monkeypatch.setattr(split_module, "BACKUP_ROOT", backup_root)

    for conjunto in CONJUNTOS:
        crear_archivo(destino / conjunto / "anterior.txt", b"anterior")
        crear_archivo(staging / conjunto / "nuevo.txt", b"nuevo")
    crear_archivo(destino / "desconocido" / "objeto.jpg", b"desconocido")

    respaldo = split_module.activar_nueva_division(staging, destino)

    for conjunto in CONJUNTOS:
        assert (destino / conjunto / "nuevo.txt").read_bytes() == b"nuevo"
        assert (respaldo / conjunto / "anterior.txt").read_bytes() == b"anterior"
    assert (destino / "desconocido" / "objeto.jpg").exists()
    assert not staging.exists()
