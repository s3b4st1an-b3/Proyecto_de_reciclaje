import os

from flask import send_from_directory

from api.index import app

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")


@app.route("/")
def serve_index():
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(PUBLIC_DIR, path)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("SERVIDOR LOCAL ACTIVO")
    print("Ingresa a: http://localhost:3000")
    print("=" * 50 + "\n")
    app.run(host="127.0.0.1", port=3000, debug=True)
