import numpy as np

""" Todos aquellas funciones que sean de utilidad para implementar los métodos de clase que se mencionan anteriormente"""

def check_types(*expected_types) -> None:
    """Verifica si los valores coinciden con los tipos esperados.

    Parámetros:
        expected_types (tuplas): Cada tupla contiene un valor y el tipo (o tipos) esperados.

    Excepciones:
        TypeError: Si algún valor no es del tipo esperado."""
        
    for value, expected in expected_types:
        if not isinstance(value, expected):
            raise TypeError(f"Error: Se esperaba tipo '{expected}' pero se recibió '{type(value)}' con valor '{value}'")

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normaliza una señal a un rango estándar (media 0, desviación estándar 1).

    Parámetros:
        signal (np.ndarray): Señal que será normalizada.

    Retorna:
        np.ndarray: Señal normalizada."""
        
    return (signal - np.mean(signal)) / np.std(signal)

def time_to_samples(time: float, sfreq: int) -> int:
    """Convierte el tiempo en segundos a muestras.

    Parámetros:
        time (float): Tiempo en segundos.
        sfreq (int): Frecuencia de muestreo de la señal.

    Retorna:
        int: Índice de la muestra correspondiente al tiempo dado."""
        
    return int(time * sfreq)