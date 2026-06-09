# toolbox_ml🧰

Paquete de Python para análisis exploratorio de datos (EDA) y selección
de features para proyectos de Machine Learning.

---

## 👥 Equipo

| Nombre | Rol | Responsabilidad |
|---|---|---|
| Mª Teresa López | Scrum Master | Setup, integración y notebook demo |
| Raquel Martínez | Desarrollador 1 | `describe_df`, `tipifica_variables` |
| Ali Madriz | Desarrollador 2 | `get/plot_features_num_regression`, `get/plot_features_cat_regression` |

---

## 📁 Estructura del repositorio

```
toolbox_ml/
├── __init__.py
└── eda/
    ├── __init__.py
    └── core.py
tests/
├── __init__.py
└── test_core.py
notebooks/
└── demo.ipynb
.gitignore
README.md
requirements.txt
setup.py
```
---

## ⚙️ Instrucciones de instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/vuestro-grupo/toolbox_ml.git
```

2. Entrar en la carpeta del proyecto:
```bash
cd toolbox_ml
```

3. Crear el entorno virtual:
```bash
python -m venv venv
```

4. Activar el entorno virtual:
```bash
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

5. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

6. Instalar el paquete en modo desarrollo:
```bash
pip install -e .
```

---

## 🔧 Funciones disponibles

### `describe_df`
Genera un resumen estadístico del DataFrame con el tipo de dato, porcentaje
de nulos, valores únicos y porcentaje de cardinalidad de cada columna.

```python
from toolbox_ml.eda.core import describe_df
import seaborn as sns

df = sns.load_dataset('diamonds')
describe_df(df)
```

---

### `tipifica_variables`
Clasifica automáticamente cada variable en: Binaria, Categórica,
Numérica Discreta o Numérica Continua.

```python
from toolbox_ml.eda.core import tipifica_variables

tipifica_variables(df, umbral_categoria=10, umbral_continua=30.0)
```

---

### `get_features_num_regression`
Devuelve las columnas numéricas con correlación significativa con el target.

```python
from toolbox_ml.eda.core import get_features_num_regression

get_features_num_regression(df, target_col='price', umbral_corr=0.5, pvalue=0.05)
```

---

### `plot_features_num_regression`
Pinta pairplots de las columnas numéricas que superan el umbral de correlación
con el target.

```python
from toolbox_ml.eda.core import plot_features_num_regression

plot_features_num_regression(df, target_col='price', umbral_corr=0.5, pvalue=0.05)
```

---

### `get_features_cat_regression`
Devuelve las columnas categóricas con relación estadísticamente significativa
con el target.

```python
from toolbox_nl.eda.core import get_features_cat_regression

get_features_cat_regression(df, target_col='price', pvalue=0.05)
```

---

### `plot_features_cat_regression`
Pinta histogramas agrupados de las columnas categóricas con relación
estadísticamente significativa con el target.

```python
from toolbox_ml.eda.core import plot_features_cat_regression

plot_features_cat_regression(df, target_col='price', pvalue=0.05, with_individual_plot=False)
```

---

## 🔬 Tests

Los tests unitarios están escritos con `pytest` y se encuentran en `tests/test_core.py`.

Comprueban que cada función del paquete funciona correctamente cubriendo tres tipos de casos:

- ✅ **Caso correcto:** input válido → output esperado
- ⚠️ **Caso límite:** DataFrame vacío, columna con todos los valores nulos, etc.
- ❌ **Caso de error:** input incorrecto → la función retorna `None`

Para ejecutar todos los tests:

```bash
pytest tests/ -v
```

---

## 💎 Dataset utilizado en la demo

Diamond dataset de seaborn — precios de diamantes con variables numéricas
y categóricas.

```python
import seaborn as sns
df = sns.load_dataset('diamonds')
```