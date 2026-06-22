import os
from flask import send_from_directory
# Importamos tu servidor backend que ya programaste
from api.index import app

# Decirle a Python que busque los archivos visuales en la carpeta 'public'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, 'public')

@app.route('/')
def serve_index():
    return send_from_directory(PUBLIC_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(PUBLIC_DIR, path)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("SERVIDOR LOCAL ACTIVO")
    print("ingresa a: http://localhost:3000")
    print("="*50 + "\n")
    app.run(port=3000, debug=True)