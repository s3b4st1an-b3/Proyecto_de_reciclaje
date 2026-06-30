# Pruebas

Instala las dependencias:

```powershell
pip install -r requirements-test.txt
```

Ejecuta la suite:

```powershell
pytest -q
```

Las pruebas unitarias utilizan una sesión ONNX simulada y no requieren el
modelo entrenado. Comprueban:

- imágenes JPEG y PNG válidas;
- datos Base64 e imágenes dañadas;
- límite de solicitud de 4 MB;
- contrato completo de respuesta;
- las cuatro categorías;
- respuesta `No identificado`;
- forma y normalización del tensor.

Cuando exista el modelo entrenado, ejecuta también:

```powershell
python training/test_onnx.py
```

Para una comprobación local completa, inicia `python run_local.py`, visita
`http://localhost:3000/api/classify` y confirma que el estado sea `ok`.

También puedes automatizar el diagnóstico y una clasificación:

```powershell
python tests/smoke_deployment.py --image C:\ruta\foto.jpg
```

Después del despliegue en Vercel, repite la comprobación en:

```powershell
python tests/smoke_deployment.py `
  --base-url https://TU-DOMINIO `
  --image C:\ruta\foto.jpg
```

Finalmente prueba desde la interfaz una imagen nueva de cada categoría y una
imagen ajena al dataset. Esta comprobación manual valida cámara, HTTPS,
frontend, función serverless y modelo en conjunto.
