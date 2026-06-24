"""Punto de entrada que Vercel detecta para desplegar Flask."""

from api.index import app

__all__ = ["app"]
