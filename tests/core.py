import pytest
import pandas as pd
import numpy as np
from toolbox_ml.eda.core import (
    get_features_num_regression,
    plot_features_num_regression,
    get_features_cat_regression,
    plot_features_cat_regression
)

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