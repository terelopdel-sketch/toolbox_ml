import pandas as pd
import numpy as np

# Funcion 1: describe_df
def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen descriptivo de un DataFrame con informacion clave
    sobre los tipos de datos, valores nulos y cardinalidad de cada columna.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.

    Retorna:
        pd.DataFrame: DataFrame con una fila por columna del input y las
        siguientes columnas:
            - 'tipo': tipo de dato de la columna (object, int64, float64, bool, etc.)
            - 'porcentaje_nulos': porcentaje de valores nulos sobre el total de filas.
            - 'valores_unicos': numero de valores unicos distintos (sin contar NaN).
            - 'porcentaje_cardinalidad': porcentaje que representa el numero de
              valores unicos sobre el total de filas.
        Retorna None si el input no es un DataFrame valido.
    """
    if not isinstance(df, pd.DataFrame):
        print(f"[ERROR] describe_df: el argumento 'df' debe ser un pd.DataFrame. "
              f"Se recibio: {type(df).__name__}.")
        return None

    total_filas = len(df)

    resultados = {}
    for col in df.columns:

        # Tipo de dato de la columna
        tipo = df[col].dtype

        # Numero de nulos y su porcentaje sobre el total de filas
        num_nulos = df[col].isna().sum()
        pct_nulos = round((num_nulos / total_filas) * 100, 2) if total_filas > 0 else 0.0

        # Numero de valores unicos (nunique excluye NaN por defecto)
        valores_unicos = df[col].nunique()

        # Porcentaje de cardinalidad: valores unicos / total filas
        pct_cardinalidad = round((valores_unicos / total_filas) * 100, 2) if total_filas > 0 else 0.0

        resultados[col] = {
            "tipo": tipo,
            "porcentaje_nulos": pct_nulos,
            "valores_unicos": valores_unicos,
            "porcentaje_cardinalidad": pct_cardinalidad,
        }

    # --- Crear DataFrame de resultado con el nombre de columna como indice ---
    resultado_df = pd.DataFrame(resultados).T
    resultado_df.index.name = None

    return resultado_df

# Funcion 2: tipifica_variables
def tipifica_variables(
    df: pd.DataFrame,
    umbral_categoria: int,
    umbral_continua: float
) -> pd.DataFrame:
    """
    Sugiere el tipo estadistico de cada columna de un DataFrame basandose
    en su cardinalidad y el porcentaje de valores unicos.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.
        umbral_categoria (int): Numero minimo de valores unicos para que una
            variable sea considerada numerica en lugar de categorica.
            Debe ser un entero positivo.
        umbral_continua (float): Porcentaje minimo de cardinalidad (0-100)
            para distinguir variables numericas continuas de discretas.
            Debe ser un float entre 0 y 100.

    Retorna:
        pd.DataFrame: DataFrame con dos columnas:
            - 'nombre_variable': nombre de cada columna del DataFrame de entrada.
            - 'tipo_sugerido': tipo estadistico sugerido para esa columna.
        Retorna None si alguno de los argumentos no supera las validaciones.
    """

    if not isinstance(df, pd.DataFrame):
        print(f"[ERROR] tipifica_variables: 'df' debe ser un pd.DataFrame. "
              f"Se recibio: {type(df).__name__}.")
        return None

    if not isinstance(umbral_categoria, int) or umbral_categoria <= 0:
        print(f"[ERROR] tipifica_variables: 'umbral_categoria' debe ser un entero positivo. "
              f"Se recibio: {umbral_categoria} ({type(umbral_categoria).__name__}).")
        return None

    if not isinstance(umbral_continua, (int, float)) or not (0 <= umbral_continua <= 100):
        print(f"[ERROR] tipifica_variables: 'umbral_continua' debe ser un float entre 0 y 100. "
              f"Se recibio: {umbral_continua}.")
        return None

    total_filas = len(df)
    nombres = []
    tipos_sugeridos = []

    for col in df.columns:

        # Cardinalidad: numero de valores unicos distintos (sin NaN)
        cardinalidad = df[col].nunique()

        # Porcentaje de cardinalidad sobre el total de filas
        pct_cardinalidad = (cardinalidad / total_filas * 100) if total_filas > 0 else 0.0

        if cardinalidad == 2:
            tipo = "Binaria"

        elif cardinalidad < umbral_categoria:
            tipo = "Categórica"

        elif pct_cardinalidad >= umbral_continua:
            tipo = "Numérica Continua"

        else:
            tipo = "Numérica Discreta"

        nombres.append(col)
        tipos_sugeridos.append(tipo)

    # --- Construir DataFrame de resultado ---
    resultado_df = pd.DataFrame({
        "nombre_variable": nombres,
        "tipo_sugerido": tipos_sugeridos,
    })
    return resultado_df
