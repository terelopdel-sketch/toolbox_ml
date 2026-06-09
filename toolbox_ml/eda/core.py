
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns



# Funcion 1: describe_df
def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen descriptivo de un DataFrame con informacion clave
    sobre los tipos de datos, valores nulos y cardinalidad de cada columna.
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

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

def get_features_num_regression(
    df: pd.DataFrame,
    target_col: str,
    umbral_corr: float,
    pvalue: float = None
) -> list:
    """
    Devuelve las columnas numéricas del DataFrame cuya correlación de Pearson
    con target_col supera en valor absoluto el umbral indicado.

    Argumentos:
        df (pd.DataFrame): DataFrame de entrada.
        target_col (str): Nombre de la columna target (debe ser numérica).
        umbral_corr (float): Umbral mínimo de correlación en valor absoluto (entre 0 y 1).
        pvalue (float): Si se indica, filtra además por significancia estadística.
                        Solo se incluyen columnas con p-valor < pvalue. Default None.

    Retorna:
        list: Lista con los nombres de las columnas que cumplen los criterios.
              Retorna None si alguna comprobación de entrada falla.
    """

    # Comprobación 1: df debe ser un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' no es un pd.DataFrame.")
        return None

    # Comprobación 2: target_col debe existir en df
    if target_col not in df.columns:
        print(f"Error: '{target_col}' no existe en el DataFrame.")
        return None

    # Comprobación 3: target_col debe ser numérica
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: '{target_col}' no es una columna numérica.")
        return None

    # Comprobación 4: umbral_corr debe ser float entre 0 y 1
    if not isinstance(umbral_corr, (int, float)) or not (0 <= umbral_corr <= 1):
        print("Error: 'umbral_corr' debe ser un número entre 0 y 1.")
        return None

    # Comprobación 5: pvalue, si se indica, debe ser float entre 0 y 1
    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 < pvalue <= 1):
            print("Error: 'pvalue' debe ser un número entre 0 y 1.")
            return None

    # Seleccionar columnas numéricas excluyendo el target
    columnas_numericas = df.select_dtypes(include=np.number).columns.tolist()
    columnas_candidatas = [col for col in columnas_numericas if col != target_col]

    columnas_seleccionadas = []

    for col in columnas_candidatas:
        # Eliminar filas con nulos en el par de columnas antes de calcular
        datos_validos = df[[col, target_col]].dropna()

        # Calcular correlación de Pearson y p-valor
        corr, p = stats.pearsonr(datos_validos[col], datos_validos[target_col])

        # Aplicar filtro de correlación
        if abs(corr) > umbral_corr:
            # Aplicar filtro de p-valor si se ha indicado
            if pvalue is None or p < pvalue:
                columnas_seleccionadas.append(col)

    return columnas_seleccionadas
    pass

def plot_features_num_regression(
    df: pd.DataFrame,
    target_col: str = "",
    columns: list = [],
    umbral_corr: float = 0,
    pvalue: float = None
) -> list:
    """
    Pinta pairplots de target_col frente a las columnas numéricas que superen
    los criterios de correlación. Si hay más de 5 columnas, divide en grupos.

    Argumentos:
        df (pd.DataFrame): DataFrame de entrada.
        target_col (str): Nombre de la columna target (debe ser numérica).
        columns (list): Lista de columnas candidatas. Si está vacía, usa todas
                        las columnas numéricas del DataFrame.
        umbral_corr (float): Umbral mínimo de correlación en valor absoluto (entre 0 y 1).
        pvalue (float): Si se indica, filtra además por significancia estadística. Default None.

    Retorna:
        list: Lista de columnas que cumplen los criterios y se han representado.
              Retorna None si alguna comprobación de entrada falla.
    """

    # Comprobación 1: df debe ser un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' no es un pd.DataFrame.")
        return None

    # Comprobación 2: target_col debe existir en df
    if target_col not in df.columns:
        print(f"Error: '{target_col}' no existe en el DataFrame.")
        return None

    # Comprobación 3: target_col debe ser numérica
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: '{target_col}' no es una columna numérica.")
        return None

    # Comprobación 4: umbral_corr debe ser float entre 0 y 1
    if not isinstance(umbral_corr, (int, float)) or not (0 <= umbral_corr <= 1):
        print("Error: 'umbral_corr' debe ser un número entre 0 y 1.")
        return None

    # Comprobación 5: pvalue, si se indica, debe ser float entre 0 y 1
    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 < pvalue <= 1):
            print("Error: 'pvalue' debe ser un número entre 0 y 1.")
            return None

    # Si columns está vacía, usar todas las columnas numéricas menos el target
    if len(columns) == 0:
        columns = df.select_dtypes(include=np.number).columns.tolist()
        columns = [col for col in columns if col != target_col]

    # Filtrar las columnas candidatas por correlación (y p-valor si aplica)
    columnas_seleccionadas = []

    for col in columns:
        # Solo procesar columnas que existan y sean numéricas
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        datos_validos = df[[col, target_col]].dropna()
        corr, p = stats.pearsonr(datos_validos[col], datos_validos[target_col])

        if abs(corr) > umbral_corr:
            if pvalue is None or p < pvalue:
                columnas_seleccionadas.append(col)

    # Si no hay columnas que cumplan los criterios, avisar y salir
    if len(columnas_seleccionadas) == 0:
        print("Ninguna columna supera los criterios indicados.")
        return columnas_seleccionadas

    # Dividir en grupos de máximo 5 columnas y pintar un pairplot por grupo
    tamanio_grupo = 5

    for i in range(0, len(columnas_seleccionadas), tamanio_grupo):
        grupo = columnas_seleccionadas[i:i + tamanio_grupo]
        columnas_plot = [target_col] + grupo

        sns.pairplot(df[columnas_plot].dropna(), diag_kind="kde")
        plt.suptitle(f"Pairplot grupo {i // tamanio_grupo + 1}", y=1.02)
        plt.show()

    return columnas_seleccionadas
    pass


def get_features_cat_regression(
    df: pd.DataFrame,
    target_col: str,
    pvalue: float = 0.05
) -> list:
    """
    Devuelve las columnas categóricas del DataFrame cuya relación estadística
    con target_col sea significativa al nivel indicado.

    Usa Mann-Whitney U si la variable tiene 2 categorías, o ANOVA si tiene más.

    Argumentos:
        df (pd.DataFrame): DataFrame de entrada.
        target_col (str): Nombre de la columna target (debe ser numérica).
        pvalue (float): Nivel de significancia. Solo se incluyen columnas con
                        p-valor < pvalue. Default 0.05.

    Retorna:
        list: Lista con los nombres de las columnas categóricas significativas.
              Retorna None si alguna comprobación de entrada falla.
    """

    # Comprobación 1: df debe ser un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' no es un pd.DataFrame.")
        return None

    # Comprobación 2: target_col debe existir en df
    if target_col not in df.columns:
        print(f"Error: '{target_col}' no existe en el DataFrame.")
        return None

    # Comprobación 3: target_col debe ser numérica
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: '{target_col}' no es una columna numérica.")
        return None

    # Comprobación 4: pvalue debe ser float entre 0 y 1
    if not isinstance(pvalue, (int, float)) or not (0 < pvalue <= 1):
        print("Error: 'pvalue' debe ser un número entre 0 y 1.")
        return None

    # Seleccionar columnas categóricas (object o category) excluyendo el target
    columnas_categoricas = df.select_dtypes(include=["object", "category"]).columns.tolist()
    columnas_candidatas = [col for col in columnas_categoricas if col != target_col]

    columnas_seleccionadas = []

    for col in columnas_candidatas:
        # Eliminar filas con nulos en el par de columnas
        datos_validos = df[[col, target_col]].dropna()

        # Obtener las categorías únicas de la columna
        categorias = datos_validos[col].unique()

        # Agrupar los valores del target por cada categoría
        grupos = [datos_validos[target_col][datos_validos[col] == cat] for cat in categorias]

        # Seleccionar el test según el número de categorías
        if len(categorias) == 2:
            # Mann-Whitney U para variables binarias
            _, p = stats.mannwhitneyu(grupos[0], grupos[1], alternative="two-sided")
        elif len(categorias) > 2:
            # ANOVA de un factor para variables con más de 2 categorías
            _, p = stats.f_oneway(*grupos)
        else:
            # Si solo hay 1 categoría no tiene sentido hacer el test
            continue

        # Si el p-valor es significativo, añadir la columna
        if p < pvalue:
            columnas_seleccionadas.append(col)

    return columnas_seleccionadas
    pass


def plot_features_cat_regression(
    df: pd.DataFrame,
    target_col: str = "",
    columns: list = [],
    pvalue: float = 0.05,
    with_individual_plot: bool = False
) -> list:
    """
    Para cada columna categórica que supere el test estadístico, pinta
    histogramas agrupados de target_col por cada valor de esa variable.

    Argumentos:
        df (pd.DataFrame): DataFrame de entrada.
        target_col (str): Nombre de la columna target (debe ser numérica).
        columns (list): Lista de columnas candidatas. Si está vacía, usa todas
                        las columnas categóricas del DataFrame.
        pvalue (float): Nivel de significancia para el test estadístico. Default 0.05.
        with_individual_plot (bool): Si es False, pinta todas las variables en
                                     una sola figura. Si es True, una figura por variable.

    Retorna:
        list: Lista de columnas categóricas que han superado el test y se han representado.
              Retorna None si alguna comprobación de entrada falla.
    """

    # Comprobación 1: df debe ser un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' no es un pd.DataFrame.")
        return None

    # Comprobación 2: target_col debe existir en df
    if target_col not in df.columns:
        print(f"Error: '{target_col}' no existe en el DataFrame.")
        return None

    # Comprobación 3: target_col debe ser numérica
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: '{target_col}' no es una columna numérica.")
        return None

    # Comprobación 4: pvalue debe ser float entre 0 y 1
    if not isinstance(pvalue, (int, float)) or not (0 < pvalue <= 1):
        print("Error: 'pvalue' debe ser un número entre 0 y 1.")
        return None

    # Si columns está vacía, usar todas las columnas categóricas menos el target
    if len(columns) == 0:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
        columns = [col for col in columns if col != target_col]

    # Filtrar columnas por significancia estadística (misma lógica que get_features_cat_regression)
    columnas_seleccionadas = []

    for col in columns:
        if col not in df.columns:
            continue

        datos_validos = df[[col, target_col]].dropna()
        categorias = datos_validos[col].unique()
        grupos = [datos_validos[target_col][datos_validos[col] == cat] for cat in categorias]

        if len(categorias) == 2:
            _, p = stats.mannwhitneyu(grupos[0], grupos[1], alternative="two-sided")
        elif len(categorias) > 2:
            _, p = stats.f_oneway(*grupos)
        else:
            continue

        if p < pvalue:
            columnas_seleccionadas.append(col)

    # Si no hay columnas significativas, avisar y salir
    if len(columnas_seleccionadas) == 0:
        print("Ninguna columna supera el nivel de significancia indicado.")
        return columnas_seleccionadas

    # Pintar los histogramas
    if with_individual_plot:
        # Una figura independiente por cada variable
        for col in columnas_seleccionadas:
            fig, ax = plt.subplots(figsize=(8, 4))
            for cat in df[col].dropna().unique():
                datos = df[target_col][df[col] == cat].dropna()
                ax.hist(datos, alpha=0.5, label=str(cat), bins=20)
            ax.set_title(f"Distribución de '{target_col}' por '{col}'")
            ax.set_xlabel(target_col)
            ax.set_ylabel("Frecuencia")
            ax.legend()
            plt.tight_layout()
            plt.show()
    else:
        # Todas las variables en una sola figura con subplots
        n = len(columnas_seleccionadas)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))

        # Si solo hay una columna, axes no es lista, lo convertimos
        if n == 1:
            axes = [axes]

        for ax, col in zip(axes, columnas_seleccionadas):
            for cat in df[col].dropna().unique():
                datos = df[target_col][df[col] == cat].dropna()
                ax.hist(datos, alpha=0.5, label=str(cat), bins=20)
            ax.set_title(f"'{target_col}' por '{col}'")
            ax.set_xlabel(target_col)
            ax.set_ylabel("Frecuencia")
            ax.legend()

        plt.tight_layout()
        plt.show()

    return columnas_seleccionadas
    pass