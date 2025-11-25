# Contenido Teórico: Machine Learning Supervisado
## Sistemas de Aprendizaje Automático - Bloque 1

## 1. Introducción al Aprendizaje Supervisado

### 1.1 Definición
Algoritmos que aprenden de datos etiquetados donde se conoce tanto la entrada como la salida esperada.

### 1.2 Requisitos
- Dataset con ejemplos etiquetados
- División en conjuntos de entrenamiento y prueba
- Métricas de evaluación

## 2. Algoritmos de Clasificación

### 2.1 Árboles de Decisión
- Estructura jerárquica de decisiones
- Interpretables y visualizables
- Propensos a sobreajuste

### 2.2 Máquinas de Vectores de Soporte (SVM)
- Encuentran el hiperplano óptimo
- Efectivas en espacios de alta dimensionalidad
- Requieren ajuste de hiperparámetros

### 2.3 Clasificadores Bayesianos
- Basados en probabilidad condicional
- Algoritmo Naive Bayes asume independencia
- Rápidos y efectivos para texto

## 3. Algoritmos de Regresión

### 3.1 Regresión Lineal
Modelo: Y = W*X + b
- Simple e interpretable
- Asume relación lineal

### 3.2 Regresión Polinomial
- Captura relaciones no lineales
- Requiere selección de grado
- Riesgo de sobreajuste

### 3.3 Métodos de Regularización
- Ridge (L2): Penaliza magnitud de pesos
- Lasso (L1): Puede eliminar características
- Elastic Net: Combinación de Ridge y Lasso

## 4. Evaluación de Modelos

### 4.1 Métricas de Clasificación
- Matriz de confusión
- Precisión y Recall
- F1-Score
- ROC-AUC

### 4.2 Métricas de Regresión
- Error Absoluto Medio (MAE)
- Error Cuadrático Medio (MSE)
- Raíz del Error Cuadrático Medio (RMSE)
- Coeficiente de Determinación (R²)

## 5. Validación Cruzada
- K-Fold Cross-Validation
- Estratificada para datos desbalanceados
- Leave-One-Out (LOO)
