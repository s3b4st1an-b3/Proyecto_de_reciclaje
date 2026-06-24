# Clasificador de reciclaje

Aplicación web que captura una imagen desde la cámara y la clasifica con un
modelo ONNX en cuatro categorías: papel/cartón, vidrio, plástico y orgánico.

## Ejecutar en local (Windows)

Requisitos:

- Python 3.12
- Una cámara web

Desde PowerShell, en la raíz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_local.py
```

Abre <http://localhost:3000>. El navegador permite usar la cámara en
`localhost`; no abras `index.html` directamente como un archivo.

Para comprobar solamente la API y la carga del modelo, visita:
<http://localhost:3000/api/classify>.

## Desplegar en Vercel

El proyecto ya incluye `vercel.json`, la versión de Python y las dependencias.
La cámara funciona en Vercel porque el sitio se publica con HTTPS.

Opción recomendada:

1. Sube el repositorio a GitHub, GitLab o Bitbucket.
2. En Vercel, crea un proyecto e importa el repositorio.
3. Deja vacío el campo **Build Command** y usa la raíz del repositorio como
   **Root Directory**.
4. Pulsa **Deploy**.

También puedes usar la CLI:

```powershell
npm install -g vercel
vercel
vercel --prod
```

Después del despliegue, comprueba `https://TU-DOMINIO/api/classify` antes de
probar la cámara.

## Estructura

- `public/`: interfaz web servida por el CDN de Vercel.
- `app.py`: punto de entrada Flask detectado por Vercel.
- `api/index.py`: API Flask compatible con Vercel.
- `api/modelo_reciclaje.onnx*`: modelo y pesos externos ONNX.
- `run_local.py`: servidor local para frontend y API.
