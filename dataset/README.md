# Dataset de clasificación de residuos

El modelo se entrenará con cuatro categorías, en este orden:

1. `papel_carton`
2. `vidrio`
3. `plastico`
4. `organico`

Las imágenes deben ubicarse en `train`, `validation` o `test` según el
conjunto al que pertenezcan. No copies una misma imagen en más de un conjunto.

Distribución inicial recomendada:

- `train`: 70 % de las imágenes.
- `validation`: 15 % de las imágenes.
- `test`: 15 % de las imágenes.

La carpeta `desconocido` contiene objetos que no pertenecen a las cuatro
categorías oficiales. Estas imágenes no deben tratarse como una quinta salida
durante el entrenamiento: se utilizarán para comprobar y ajustar el umbral que
produce la respuesta `No identificado`.

Formatos recomendados: JPG, JPEG, PNG y WebP. Procura incluir diferentes
fondos, cámaras, distancias, posiciones y condiciones de iluminación.

Las imágenes están excluidas de Git para evitar que el repositorio aumente
demasiado de tamaño. Los archivos `.gitkeep` conservan únicamente la
estructura de carpetas.

## Herramientas

Para revisar la estructura, detectar imágenes dañadas, archivos no admitidos y
duplicados exactos:

```cmd
python training/validate_dataset.py
```

Para dividir una carpeta fuente que contenga las cuatro categorías:

```cmd
python training\split_dataset.py --source C:\ruta\a\imagenes_clasificadas --dry-run
```

La simulación no modifica archivos. Cuando el resultado sea correcto, puede
reconstruirse la división existente:

```cmd
python training\split_dataset.py --source C:\ruta\a\imagenes_clasificadas --replace
```

La división usa una semilla fija, conserva `dataset/desconocido` y respalda la
división anterior en `training/artifacts/backups`. Descarta automáticamente:

- imágenes con lado corto menor a 128 píxeles;
- duplicados exactos dentro de una misma categoría;
- imágenes idénticas que aparecen en categorías diferentes.

Los archivos originales de la carpeta fuente nunca se modifican.

## Entrenamiento

Instala las dependencias de entrenamiento en un entorno virtual separado:

```cmd
pip install -r training/requirements.txt
```

Después de llenar y validar el dataset:

```cmd
python training/train.py
```

El entrenamiento utiliza MobileNetV3 Small con pesos preentrenados, aumentos
de datos y ajuste fino. Guarda el mejor modelo según el F1 macro de validación
en `training/artifacts/best_model.pt`, junto con:

- `history.json`: métricas por época.
- `test_metrics.json`: precisión, recall y F1 final.
- `confusion_matrix.csv`: valores de la matriz de confusión.
- `confusion_matrix.png`: visualización de la matriz.

Puedes cambiar épocas, tamaño de lote y otros parámetros en
`training/config.json`. El orden de categorías está fijado para coincidir con
la API y no debe modificarse sin actualizar también el backend.

## Exportación a ONNX

Cuando exista `training/artifacts/best_model.pt`, genera un modelo candidato:

```cmd
python training/export_onnx.py
```

El exportador comprueba la estructura ONNX, sus metadatos, la entrada
`[batch, 3, 224, 224]` y las cuatro salidas.

Después compara PyTorch y ONNX con imágenes del conjunto de prueba:

```cmd
python training/test_onnx.py
```

Esta prueba elimina duplicados exactos de `dataset/desconocido`, separa esas
imágenes en dos grupos y usa uno para calibrar automáticamente el umbral de
`No identificado`. El segundo grupo se reserva para medir la tasa real de
rechazo. El umbral seleccionado se guarda dentro del propio ONNX.

Si todas las predicciones y valores numéricos coinciden, puedes instalar el
modelo validado en la API después de completar la adaptación del
preprocesamiento del backend:

```cmd
python training/test_onnx.py --install
```

La instalación no ocurre si falla una comprobación. Si ya existe un modelo en
`api/`, primero se crea una copia de respaldo con fecha y hora.

El backend lee del propio ONNX el tamaño de entrada, redimensión, recorte,
espacio de color, normalización y orden de categorías. Un modelo sin estos
metadatos será rechazado como incompatible, evitando procesar imágenes con una
configuración distinta de la usada durante el entrenamiento.
