
import pytest
import pandas as pd
from toolbox_ml.eda.core import describe_df, tipifica_variables

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
