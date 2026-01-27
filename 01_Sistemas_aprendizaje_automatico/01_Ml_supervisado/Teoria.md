# Aprendizaje Automático Supervisado: Fundamentos Teóricos y Metodológicos
## Sistemas de Aprendizaje Automático - Bloque 1

## Resumen

El aprendizaje supervisado constituye el paradigma fundamental del aprendizaje automático moderno, caracterizado por la utilización de conjuntos de datos etiquetados para entrenar modelos predictivos. Este documento examina sistemáticamente los fundamentos teóricos, algoritmos principales, metodologías de evaluación y consideraciones prácticas que sustentan este enfoque. Se presenta un análisis riguroso de las técnicas de clasificación y regresión, incluyendo sus fundamentos matemáticos, propiedades estadísticas y aplicaciones en contextos reales.

## 1. Fundamentos del Aprendizaje Supervisado

### 1.1 Definición Formal y Marco Teórico

El aprendizaje supervisado se define formalmente como el problema de aproximar una función de mapeo $f: \mathcal{X} \rightarrow \mathcal{Y}$ que relaciona un espacio de entrada $\mathcal{X}$ con un espacio de salida $\mathcal{Y}$, basándose en un conjunto finito de pares de entrenamiento $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{n}$, donde $x_i \in \mathcal{X}$ representa las características observadas e $y_i \in \mathcal{Y}$ denota las etiquetas o valores objetivo correspondientes.

El objetivo fundamental consiste en inducir una hipótesis $h \in \mathcal{H}$ (donde $\mathcal{H}$ representa el espacio de hipótesis) que minimice el riesgo esperado:

$$R(h) = \mathbb{E}_{(x,y) \sim P(\mathcal{X}, \mathcal{Y})}[\mathcal{L}(h(x), y)]$$

donde $\mathcal{L}$ representa una función de pérdida apropiada y $P(\mathcal{X}, \mathcal{Y})$ denota la distribución conjunta subyacente de los datos.

### 1.2 Taxonomía de Problemas

**Clasificación**: Cuando el espacio de salida $\mathcal{Y}$ es discreto y finito, el problema se categoriza como clasificación. Puede subdividirse en:
- **Clasificación binaria**: $\mathcal{Y} = \{0, 1\}$ o $\{-1, +1\}$
- **Clasificación multiclase**: $|\mathcal{Y}| = K$ con $K > 2$
- **Clasificación multilabel**: Cada instancia puede pertenecer a múltiples clases simultáneamente

**Regresión**: Cuando $\mathcal{Y} \subseteq \mathbb{R}^d$, se trata de un problema de regresión, donde el objetivo es predecir valores continuos.

### 1.3 Requisitos y Consideraciones Metodológicas

**Conjunto de Datos Etiquetados**: La disponibilidad de ejemplos de entrenamiento con anotaciones correctas constituye el requisito fundamental. La calidad, cantidad y representatividad de estos datos determinan fundamentalmente el rendimiento del modelo resultante.

**Partición de Datos**: División estratégica del conjunto de datos en subconjuntos complementarios:
- **Conjunto de entrenamiento** (60-80%): Utilizado para ajustar parámetros del modelo
- **Conjunto de validación** (10-20%): Empleado para ajuste de hiperparámetros y selección de modelos
- **Conjunto de prueba** (10-20%): Reservado exclusivamente para evaluación final imparcial

**Independencia e Identidad Distribucional (i.i.d.)**: Los ejemplos de entrenamiento deben ser muestras independientes extraídas de una distribución común, garantizando validez de inferencias estadísticas.

## 2. Algoritmos de Clasificación: Análisis Comparativo

### 2.1 Árboles de Decisión

Los árboles de decisión constituyen modelos jerárquicos de decisión que particionan recursivamente el espacio de características mediante reglas if-then, formando una estructura arbórea donde nodos internos representan decisiones sobre atributos y nodos hoja asignan etiquetas de clase.

**Fundamento Teórico**: La construcción del árbol se basa en medidas de impureza como la entropía de Shannon o el índice de Gini:

$$H(S) = -\sum_{c \in C} p_c \log_2(p_c)$$

$$\text{Gini}(S) = 1 - \sum_{c \in C} p_c^2$$

donde $p_c$ representa la proporción de ejemplos de clase $c$ en el conjunto $S$.

**Ventajas**:
- Interpretabilidad excepcional mediante visualización directa del proceso de decisión
- Capacidad para manejar características numéricas y categóricas sin transformación
- No requiere normalización de datos
- Identificación automática de interacciones no lineales

**Limitaciones**:
- Alta varianza: pequeñas variaciones en datos pueden generar árboles significativamente diferentes
- Tendencia al sobreajuste en ausencia de poda o restricciones de profundidad
- Sesgo hacia características con múltiples valores
- Dificultad para capturar relaciones lineales simples

**Técnicas de Regularización**: Poda previa (pre-pruning) mediante restricción de profundidad máxima, mínimo de muestras por nodo, o poda posterior (post-pruning) mediante validación cruzada.

### 2.2 Máquinas de Vectores de Soporte (Support Vector Machines)

Las SVM constituyen algoritmos de aprendizaje basados en teoría de optimización convexa que buscan el hiperplano de separación óptimo maximizando el margen geométrico entre clases.

**Formulación Matemática**: Para el caso linealmente separable, el problema de optimización se formula como:

$$\min_{w,b} \frac{1}{2}\|w\|^2$$
$$\text{sujeto a: } y_i(w^T x_i + b) \geq 1, \forall i$$

donde $w$ representa el vector de pesos normal al hiperplano y $b$ el término de sesgo.

**Kernel Trick**: Para problemas no linealmente separables, se emplea el truco del kernel, mapeando implícitamente datos a espacios de mayor dimensionalidad mediante funciones kernel $K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$:

- **Kernel lineal**: $K(x_i, x_j) = x_i^T x_j$
- **Kernel polinomial**: $K(x_i, x_j) = (\gamma x_i^T x_j + r)^d$
- **Kernel RBF (Gaussiano)**: $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$

**Propiedades**:
- Garantía teórica de convergencia a óptimo global
- Efectividad en espacios de alta dimensionalidad
- Robustez frente a outliers mediante formulación de margen blando
- Eficiencia mediante representación mediante vectores de soporte

**Consideraciones Prácticas**: Requiere normalización de características y selección cuidadosa de hiperparámetros ($C$, parámetros de kernel) mediante búsqueda sistemática.

### 2.3 Clasificadores Bayesianos

Fundamentados en el Teorema de Bayes, estos clasificadores modelan la probabilidad posterior de cada clase dada una observación:

$$P(C_k|x) = \frac{P(x|C_k)P(C_k)}{P(x)}$$

**Naive Bayes**: Bajo la suposición (naive) de independencia condicional de características:

$$P(C_k|x_1, \ldots, x_n) \propto P(C_k) \prod_{i=1}^{n} P(x_i|C_k)$$

**Variantes**:
- **Gaussian Naive Bayes**: Asume distribuciones gaussianas para características continuas
- **Multinomial Naive Bayes**: Apropiado para conteos discretos (clasificación de texto)
- **Bernoulli Naive Bayes**: Para características binarias

**Características**:
- Entrenamiento extremadamente eficiente (lineal en cantidad de datos)
- Rendimiento sorprendentemente robusto pese a violación de suposición de independencia
- Excelente para clasificación de texto y problemas de alta dimensionalidad
- Provee estimaciones probabilísticas calibradas

### 2.4 Métodos Ensemble: Random Forests y Gradient Boosting

**Random Forests**: Conjunto de árboles de decisión entrenados en subconjuntos bootstrap de datos con selección aleatoria de características, combinando predicciones mediante votación mayoritaria.

**Gradient Boosting**: Construcción secuencial de modelos débiles donde cada nuevo modelo corrige errores residuales de modelos previos, optimizando directamente función de pérdida.

## 3. Algoritmos de Regresión

### 3.1 Regresión Lineal

El modelo fundamental de regresión establece una relación lineal entre variables predictoras y respuesta:

$$y = w_0 + w_1x_1 + \cdots + w_dx_d + \epsilon = w^Tx + \epsilon$$

**Estimación por Mínimos Cuadrados**: Los parámetros óptimos se obtienen minimizando la suma de errores cuadráticos:

$$\hat{w} = \arg\min_w \sum_{i=1}^{n}(y_i - w^Tx_i)^2$$

La solución en forma cerrada viene dada por: $\hat{w} = (X^TX)^{-1}X^Ty$

**Suposiciones del Modelo**:
- Linealidad en parámetros
- Independencia de errores
- Homocedasticidad (varianza constante de errores)
- Normalidad de residuos
- Ausencia de multicolinealidad perfecta

### 3.2 Regresión Polinomial y No Lineal

Extensión mediante transformación de características originales en términos polinomiales:

$$y = w_0 + w_1x + w_2x^2 + \cdots + w_dx^d$$

Permite capturar relaciones no lineales manteniendo linealidad en parámetros, aunque con riesgo incrementado de sobreajuste para grados elevados.

### 3.3 Métodos de Regularización

La regularización introduce penalizaciones sobre complejidad del modelo para prevenir sobreajuste:

**Ridge Regression (L2)**:
$$\min_w \sum_{i=1}^{n}(y_i - w^Tx_i)^2 + \lambda\|w\|_2^2$$

Penaliza magnitud de coeficientes pero no los reduce exactamente a cero.

**Lasso (L1)**:
$$\min_w \sum_{i=1}^{n}(y_i - w^Tx_i)^2 + \lambda\|w\|_1$$

Induce sparsity, efectivamente realizando selección de características al forzar coeficientes exactamente a cero.

**Elastic Net**:
$$\min_w \sum_{i=1}^{n}(y_i - w^Tx_i)^2 + \lambda_1\|w\|_1 + \lambda_2\|w\|_2^2$$

Combinación convexa que hereda ventajas de ambos métodos.

## 4. Evaluación y Validación de Modelos

### 4.1 Métricas de Clasificación

**Matriz de Confusión**: Tabla de contingencia que confronta predicciones con etiquetas verdaderas.

**Métricas Derivadas**:

- **Exactitud (Accuracy)**: $\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}$
  
- **Precisión (Precision)**: $P = \frac{TP}{TP + FP}$ (proporción de predicciones positivas correctas)

- **Sensibilidad (Recall/Sensitivity)**: $R = \frac{TP}{TP + FN}$ (proporción de positivos correctamente identificados)

- **Especificidad**: $\text{Spec} = \frac{TN}{TN + FP}$

- **F1-Score**: $F_1 = 2 \cdot \frac{P \cdot R}{P + R}$ (media armónica de precisión y recall)

**Curva ROC y AUC**: La curva Receiver Operating Characteristic representa sensibilidad versus (1-especificidad) para diversos umbrales de decisión. El área bajo la curva (AUC) proporciona medida agregada de rendimiento independiente del umbral.

### 4.2 Métricas de Regresión

**Error Absoluto Medio (MAE)**:
$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

Robusto ante outliers, interpretación directa en unidades originales.

**Error Cuadrático Medio (MSE)**:
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

Penaliza desproporcionadamente errores grandes.

**Raíz del Error Cuadrático Medio (RMSE)**:
$$\text{RMSE} = \sqrt{\text{MSE}}$$

Interpretable en unidades originales de la variable respuesta.

**Coeficiente de Determinación (R²)**:
$$R^2 = 1 - \frac{\sum_{i}(y_i - \hat{y}_i)^2}{\sum_{i}(y_i - \bar{y})^2}$$

Proporción de varianza explicada por el modelo; $R^2 \in (-\infty, 1]$.

## 5. Validación Cruzada y Selección de Modelos

### 5.1 K-Fold Cross-Validation

Partición del conjunto de datos en $K$ subconjuntos (folds) de tamaño aproximadamente igual. Se entrena $K$ modelos, utilizando cada vez $K-1$ folds para entrenamiento y el fold restante para validación. El rendimiento final se estima como promedio de las $K$ evaluaciones.

**Ventajas**: Utilización eficiente de datos, estimación robusta con cuantificación de incertidumbre.

### 5.2 Estratificación

En presencia de desbalance de clases, la validación cruzada estratificada mantiene proporciones de clases en cada fold, garantizando estimaciones representativas.

### 5.3 Leave-One-Out Cross-Validation (LOOCV)

Caso especial con $K=n$, donde cada observación sirve secuencialmente como conjunto de validación. Proporciona estimación casi imparcial pero computacionalmente costosa y con alta varianza.

## Referencias

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
