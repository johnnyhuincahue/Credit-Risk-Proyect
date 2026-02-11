# Credit Risk Scoring Model 🏦

Descripción del Proyecto

Este proyecto implementa un sistema completo de Scorecard de Riesgo de Crédito (Credit Scoring) utilizando Python. Simula el ciclo de vida real de un modelo de riesgo en una institución financiera, desde la ingesta de datos hasta la generación de una tarjeta de puntuación escalada (estilo FICO).

El objetivo es predecir la probabilidad de incumplimiento (default) de un cliente y traducir esa probabilidad en un puntaje de crédito interpretable.

Metodología Aplicada

El proyecto sigue los estándares de la industria bancaria (Basilea II/III):

Exploración de Datos (EDA): Análisis de distribuciones.

Ingeniería de Variables (WoE & IV):

Binning: Discretización de variables continuas.

Weight of Evidence (WoE): Transformación de variables para linearizar relaciones no monótonas y manejar outliers.

Information Value (IV): Selección de características basada en su poder predictivo.

Modelado: Uso de Regresión Logística. Aunque existen modelos más potentes (XGBoost), la Regresión Logística sigue siendo el estándar en riesgo crediticio debido a su alta interpretabilidad regulatoria.

Evaluación: Métricas AUC-ROC y Gini.

Scorecard Scaling: Conversión de log-odds a un sistema de puntos (ej. 300-850) usando PDO (Points to Double the Odds).

Estructura del Código (credit_scoring_model.py)

generate_credit_data(): Crea un dataset sintético realista con distribución log-normal para ingresos y relaciones no lineales.

calculate_woe_iv(): Función core que calcula los valores WoE y IV manualmente, esencial para entender la matemática detrás de las cajas negras.

train_model(): Entrenamiento y cálculo de métricas Gini/AUC.

create_scorecard(): Genera la tabla final que asigna puntos a cada rango de variables (ej. "Edad 25-30" = +15 puntos).

Resultados Clave

El modelo genera salidas como:

Information Value (IV) por variable para ranking de importancia.

Curva ROC para medir la discriminación del modelo.

Distribución de Scores: Visualización de la separación entre "Buenos" y "Malos" pagadores basada en el puntaje calculado.

Cómo ejecutar

Clonar el repositorio.

Instalar dependencias:

pip install pandas numpy matplotlib seaborn scikit-learn


Ejecutar el script:

python credit_scoring_model.py


Stack Tecnológico

Lenguaje: Python 3.9+

Librerías: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn.

Este proyecto fue creado con fines educativos para demostrar competencias en Risk Analytics.
