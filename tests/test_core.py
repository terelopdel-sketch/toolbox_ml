
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

from toolbox_ml.eda.core import describe_df, tipifica_variables
from toolbox_ml.eda.core import (
    get_features_num_regression,
    plot_features_num_regression,
    get_features_cat_regression,
    plot_features_cat_regression
)


# Tests de describe_df

def test_describe_df_caso_correcto():
    """Caso correcto: input valido: retorna DataFrame con columnas esperadas y valores correctos."""
    df = pd.DataFrame({'a': [1, None, None, None], 'b': ['x', 'y', 'z', 'w']})
    resultado = describe_df(df)

    assert isinstance(resultado, pd.DataFrame)
    assert set(resultado.columns) == {'tipo', 'porcentaje_nulos', 'valores_unicos', 'porcentaje_cardinalidad'}
    assert resultado.loc['a', 'porcentaje_nulos'] == pytest.approx(75.0, abs=0.01)
    assert resultado.loc['b', 'valores_unicos'] == 4


def test_describe_df_caso_limite():
    """Caso limite: columna con todos los valores nulos: porcentaje_nulos=100, valores_unicos=0."""
    df = pd.DataFrame({'vacia': [None, None, None], 'normal': [1, 2, 3]})
    resultado = describe_df(df)

    assert resultado.loc['vacia', 'porcentaje_nulos'] == pytest.approx(100.0, abs=0.01)
    assert resultado.loc['vacia', 'valores_unicos'] == 0


def test_describe_df_caso_error():
    """Caso de error: input no es DataFrame: retorna None."""
    assert describe_df("esto no es un dataframe") is None
    assert describe_df([1, 2, 3]) is None
    assert describe_df(None) is None

# Tests de tipifica_variables

def test_tipifica_variables_caso_correcto():
    """Caso correcto: clasifica correctamente los cuatro tipos posibles."""
    df = pd.DataFrame({
        'binaria':   [0, 1] * 50,                             
        'categorica': ['A', 'B', 'C'] * 33 + ['A'],           
        'continua':  [round(x * 1.1, 2) for x in range(100)],
        'discreta':  list(range(20)) * 5,
    })
    resultado = tipifica_variables(df, umbral_categoria=10, umbral_continua=50.0)

    assert isinstance(resultado, pd.DataFrame)
    assert set(resultado.columns) == {'nombre_variable', 'tipo_sugerido'}

    tipos = dict(zip(resultado['nombre_variable'], resultado['tipo_sugerido']))
    assert tipos['binaria']    == 'Binaria'
    assert tipos['categorica'] == 'Categórica'
    assert tipos['continua']   == 'Numérica Continua'
    assert tipos['discreta']   == 'Numérica Discreta'


def test_tipifica_variables_caso_limite():
    """Caso limite: DataFrame con una sola columna binaria."""
    df = pd.DataFrame({'flag': [0, 1] * 10})
    resultado = tipifica_variables(df, umbral_categoria=5, umbral_continua=50.0)

    assert len(resultado) == 1
    assert resultado.iloc[0]['tipo_sugerido'] == 'Binaria'


def test_tipifica_variables_caso_error():
    """Caso de error: argumentos invalidos: retorna None."""
    df = pd.DataFrame({'a': [1, 2, 3]})

    assert tipifica_variables("no soy un df", 5, 30.0) is None   # df inválido
    assert tipifica_variables(df, 5.5, 30.0)           is None   # umbral_categoria float
    assert tipifica_variables(df, 5, 110.0)            is None   # umbral_continua fuera de rango

# ─────────────────────────────────────────────
# Fixtures (datos de prueba)
# ─────────────────────────────────────────────

@pytest.fixture
def df_numerico():
    """DataFrame con columnas numéricas para tests de regresión numérica."""
    np.random.seed(42)
    n = 100
    target = np.random.rand(n) * 10
    return pd.DataFrame({
        "target": target,
        "alta_corr": target * 2 + np.random.rand(n),
        "baja_corr": np.random.rand(n),
        "con_nulos": np.where(np.arange(n) % 5 == 0, np.nan, target + np.random.rand(n))
    })

@pytest.fixture
def df_categorico():
    """DataFrame con columnas categóricas para tests de regresión categórica."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "target": np.concatenate([np.random.normal(0, 1, 50), np.random.normal(5, 1, 50)]),
        "binaria": ["A"] * 50 + ["B"] * 50,
        "multicat": ["X"] * 33 + ["Y"] * 33 + ["Z"] * 34,
        "no_sig": np.random.choice(["M", "N"], n).tolist()
    })


# ─────────────────────────────────────────────
# Tests de get_features_num_regression
# ─────────────────────────────────────────────

def test_get_num_retorna_lista(df_numerico):
    """Caso correcto: retorna una lista."""
    resultado = get_features_num_regression(df_numerico, "target", umbral_corr=0.5)
    assert isinstance(resultado, list)

def test_get_num_detecta_columna_correlada(df_numerico):
    """La columna muy correlada debe aparecer en el resultado."""
    resultado = get_features_num_regression(df_numerico, "target", umbral_corr=0.5)
    assert "alta_corr" in resultado

def test_get_num_excluye_columna_no_correlada(df_numerico):
    """La columna poco correlada no debe aparecer con umbral alto."""
    resultado = get_features_num_regression(df_numerico, "target", umbral_corr=0.5)
    assert "baja_corr" not in resultado

def test_get_num_con_pvalue(df_numerico):
    """Con pvalue definido, el resultado sigue siendo una lista."""
    resultado = get_features_num_regression(df_numerico, "target", umbral_corr=0.5, pvalue=0.05)
    assert isinstance(resultado, list)

def test_get_num_input_invalido_no_dataframe():
    """Input no válido: no es DataFrame → retorna None."""
    assert get_features_num_regression("no_es_df", "target", umbral_corr=0.5) is None

def test_get_num_target_no_existe(df_numerico):
    """Target que no existe en el DataFrame → retorna None."""
    assert get_features_num_regression(df_numerico, "no_existe", umbral_corr=0.5) is None

def test_get_num_target_no_numerico(df_numerico):
    """Target no numérico → retorna None."""
    df_numerico["texto"] = "abc"
    assert get_features_num_regression(df_numerico, "texto", umbral_corr=0.5) is None

def test_get_num_umbral_invalido(df_numerico):
    """umbral_corr fuera de rango → retorna None."""
    assert get_features_num_regression(df_numerico, "target", umbral_corr=1.5) is None

def test_get_num_pvalue_invalido(df_numerico):
    """pvalue fuera de rango → retorna None."""
    assert get_features_num_regression(df_numerico, "target", umbral_corr=0.5, pvalue=2.0) is None

def test_get_num_maneja_nulos(df_numerico):
    """Columna con nulos no debe causar error."""
    resultado = get_features_num_regression(df_numerico, "target", umbral_corr=0.5)
    assert isinstance(resultado, list)


# ─────────────────────────────────────────────
# Tests de plot_features_num_regression
# ─────────────────────────────────────────────

def test_plot_num_retorna_lista(df_numerico):
    """Caso correcto: retorna una lista."""
    resultado = plot_features_num_regression(df_numerico, target_col="target", umbral_corr=0.5)
    assert isinstance(resultado, list)

def test_plot_num_columnas_correctas(df_numerico):
    """Las columnas devueltas deben ser las que superan el umbral."""
    resultado = plot_features_num_regression(df_numerico, target_col="target", umbral_corr=0.5)
    assert "alta_corr" in resultado
    assert "baja_corr" not in resultado

def test_plot_num_input_invalido():
    """Input no válido → retorna None."""
    assert plot_features_num_regression("no_es_df", target_col="target") is None

def test_plot_num_target_no_existe(df_numerico):
    """Target inexistente → retorna None."""
    assert plot_features_num_regression(df_numerico, target_col="no_existe") is None

def test_plot_num_umbral_invalido(df_numerico):
    """umbral_corr fuera de rango → retorna None."""
    assert plot_features_num_regression(df_numerico, target_col="target", umbral_corr=2.0) is None

def test_plot_num_lista_vacia_usa_todas(df_numerico):
    """Con columns=[] usa todas las columnas numéricas automáticamente."""
    resultado = plot_features_num_regression(df_numerico, target_col="target", umbral_corr=0.5, columns=[])
    assert isinstance(resultado, list)


# ─────────────────────────────────────────────
# Tests de get_features_cat_regression
# ─────────────────────────────────────────────

def test_get_cat_retorna_lista(df_categorico):
    """Caso correcto: retorna una lista."""
    resultado = get_features_cat_regression(df_categorico, "target")
    assert isinstance(resultado, list)

def test_get_cat_detecta_variable_significativa(df_categorico):
    """La variable binaria con grupos distintos debe ser significativa."""
    resultado = get_features_cat_regression(df_categorico, "target", pvalue=0.05)
    assert "binaria" in resultado

def test_get_cat_excluye_variable_no_significativa(df_categorico):
    """La variable aleatoria no debe ser significativa."""
    resultado = get_features_cat_regression(df_categorico, "target", pvalue=0.05)
    assert "no_sig" not in resultado

def test_get_cat_multicat_significativa(df_categorico):
    """Variable con más de 2 categorías y grupos distintos debe ser significativa."""
    resultado = get_features_cat_regression(df_categorico, "target", pvalue=0.05)
    assert "multicat" in resultado

def test_get_cat_input_invalido():
    """Input no válido → retorna None."""
    assert get_features_cat_regression("no_es_df", "target") is None

def test_get_cat_target_no_existe(df_categorico):
    """Target inexistente → retorna None."""
    assert get_features_cat_regression(df_categorico, "no_existe") is None

def test_get_cat_target_no_numerico(df_categorico):
    """Target no numérico → retorna None."""
    assert get_features_cat_regression(df_categorico, "binaria") is None

def test_get_cat_pvalue_invalido(df_categorico):
    """pvalue fuera de rango → retorna None."""
    assert get_features_cat_regression(df_categorico, "target", pvalue=1.5) is None


# ─────────────────────────────────────────────
# Tests de plot_features_cat_regression
# ─────────────────────────────────────────────

def test_plot_cat_retorna_lista(df_categorico):
    """Caso correcto: retorna una lista."""
    resultado = plot_features_cat_regression(df_categorico, target_col="target")
    assert isinstance(resultado, list)

def test_plot_cat_columnas_significativas(df_categorico):
    """Las columnas devueltas deben ser las significativas."""
    resultado = plot_features_cat_regression(df_categorico, target_col="target", pvalue=0.05)
    assert "binaria" in resultado
    assert "no_sig" not in resultado

def test_plot_cat_input_invalido():
    """Input no válido → retorna None."""
    assert plot_features_cat_regression("no_es_df", target_col="target") is None

def test_plot_cat_target_no_existe(df_categorico):
    """Target inexistente → retorna None."""
    assert plot_features_cat_regression(df_categorico, target_col="no_existe") is None

def test_plot_cat_pvalue_invalido(df_categorico):
    """pvalue fuera de rango → retorna None."""
    assert plot_features_cat_regression(df_categorico, target_col="target", pvalue=2.0) is None

def test_plot_cat_with_individual_plot(df_categorico):
    """Con with_individual_plot=True también retorna lista correctamente."""
    resultado = plot_features_cat_regression(df_categorico, target_col="target", with_individual_plot=True)
    assert isinstance(resultado, list)
