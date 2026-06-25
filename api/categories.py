"""Configuración central de categorías y respuestas del clasificador."""

UMBRAL_CONFIANZA = 0.65

CLASE_NO_IDENTIFICADA = "No identificado"
RECOMENDACION_NO_IDENTIFICADA = (
    "Toma otra foto con buena iluminación, acerca el residuo y procura que "
    "aparezca un solo objeto sobre un fondo sencillo."
)

CATEGORIAS = {
    "Papel/Cartón": {
        "recomendacion": (
            "Deposítalo limpio y seco. Retira restos de comida, cinta adhesiva "
            "y otros materiales antes de reciclarlo."
        ),
    },
    "Vidrio": {
        "recomendacion": (
            "Vacía y enjuaga el recipiente. Separa tapas y evita mezclarlo con "
            "cerámica, espejos o bombillos."
        ),
    },
    "Plástico": {
        "recomendacion": (
            "Vacía, limpia y seca el envase. Revisa el símbolo del plástico y "
            "las reglas de reciclaje de tu localidad."
        ),
    },
    "Orgánico": {
        "recomendacion": (
            "Sepáralo de envases y materiales reciclables. Si es posible, "
            "úsalo para compostaje."
        ),
    },
}

IDENTIFICADORES_CLASE = (
    "papel_carton",
    "vidrio",
    "plastico",
    "organico",
)
CLASES = tuple(CATEGORIAS)


def obtener_recomendacion(clase):
    """Devuelve la recomendación asociada a una clase conocida."""
    if clase == CLASE_NO_IDENTIFICADA:
        return RECOMENDACION_NO_IDENTIFICADA

    categoria = CATEGORIAS.get(clase)
    if categoria is None:
        return RECOMENDACION_NO_IDENTIFICADA

    return categoria["recomendacion"]
