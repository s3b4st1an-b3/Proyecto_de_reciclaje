import os

print("=" * 60)
print("BASE_DIR:", BASE_DIR)
print("PUBLIC_DIR:", PUBLIC_DIR)
print("Contenido de BASE_DIR:", os.listdir(BASE_DIR))

if os.path.exists(PUBLIC_DIR):
    print("Contenido de PUBLIC:", os.listdir(PUBLIC_DIR))
else:
    print("La carpeta PUBLIC NO existe")

print("=" * 60)