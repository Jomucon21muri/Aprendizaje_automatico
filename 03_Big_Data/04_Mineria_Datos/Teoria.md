# Contenido Teórico: Minería de Datos
## Big Data - Bloque 4

## 1. Introducción a Minería de Datos

### 1.1 Definición
Proceso de descubrimiento de patrones e información significativa en grandes conjuntos de datos usando técnicas estadísticas, matemáticas e informáticas.

### 1.2 Diferencia con BI y Análisis
- **BI (Business Intelligence)**: Datos históricos, reportes
- **Analytics**: Interpretación exploratorio y causal
- **Data Mining**: Descubrimiento automático de patrones

### 1.3 Aplicaciones Empresariales
- Segmentación de clientes
- Predicción de churn
- Recomendaciones de productos
- Detección de fraude
- Análisis de cesta de compras

## 2. Proceso CRISP-DM

### 2.1 Fase 1: Business Understanding
- Definir objetivos del negocio (5-10% del tiempo)
- Traducir a problemas de datos
- Identificar éxito vs fracaso
- Planificar cronograma y recursos

### 2.2 Fase 2: Data Understanding
- Recopilar datos relevantes
- EDA: Exploración inicial
- Hipótesis sobre patrones
- Identificar problemas de calidad

### 2.3 Fase 3: Data Preparation
- Limpieza: Valores faltantes, duplicados, inconsistencias
- Integración: Consolidar múltiples fuentes
- Transformación: Normalización, encoding
- Feature engineering: Crear características relevantes
- **Nota**: 50-80% del tiempo

### 2.4 Fase 4: Modeling
- Seleccionar técnica(s) de minería
- Diseñar experimentos
- Entrenamiento del modelo
- Ajuste de parámetros
- Validación

### 2.5 Fase 5: Evaluation
- Evaluar performance técnico
- Interpretar resultados de negocio
- Comparar con baselines
- Revisar proceso completo
- Decisión: Proceder o iterar

### 2.6 Fase 6: Deployment
- Implementar en producción
- Monitoreo de performance
- Reentrenamiento periódico
- Mantener documentación
- Obtener feedback

## 3. Tareas de Minería Predictiva

### 3.1 Clasificación
- Predicción de categoría (target discreto)
- Algoritmos: Árboles, RF, SVM, Neural Networks
- Métricas: Exactitud, Precision, Recall, F1, ROC-AUC
- Casos: Diagnóstico médico, spam detection, evaluación crediticia

### 3.2 Regresión
- Predicción numérica (target continuo)
- Algoritmos: Lineal, polinomial, SVR, Gradient Boosting
- Métricas: MAE, MSE, RMSE, R²
- Casos: Predicción de precios, demanda, temperatura

### 3.3 Forecasting/Series Temporales
- Predicción de valores futuros
- Algoritmos: ARIMA, SARIMA, Prophet, LSTM
- Métricas: MAE, MAPE, RMSE
- Casos: Ventas, weather, precios de bolsa

## 4. Tareas de Minería Descriptiva

### 4.1 Clustering
- Agrupar datos similares (no supervisado)
- Algoritmos: K-Means, Hierarchical, DBSCAN, GMM
- Métricas: Silhouette, Davies-Bouldin, Inertia
- Casos: Segmentación de clientes, patrones biológicos

### 4.2 Análisis de Asociación
- Reglas de co-ocurrencia
- Algoritmo: Apriori, Eclat
- Métricas: Support, Confidence, Lift
- Casos: Market basket (qué productos se compran juntos)

### 4.3 Detección de Anomalías
- Identificar datos anómalos
- Algoritmos: Isolation Forest, LOF, One-Class SVM
- Métricas: Precision, Recall, F1 (si hay etiquetas)
- Casos: Detección de fraude, intrusiones, defectos

## 5. Data Quality y Preparación

### 5.1 Problemas Comunes
- Valores faltantes (missingness patterns)
- Outliers y ruido
- Inconsistencias y duplicados
- Valores codificados incorrectamente
- Desbalance de clases

### 5.2 Técnicas de Limpieza
- Imputación: Media, mediana, KNN, forward fill
- Detección de outliers: Z-score, IQR, Mahalanobis
- Normalización: Min-Max, Z-score, Robust Scaling
- Encodings: One-hot, Label, Target encoding

### 5.3 Feature Engineering
- Creación de características nuevas
- Selección: Filtros, wrappers, embedding
- Reducción: PCA, t-SNE, Autoencoders
- Características temporales: lag, rolling stats

## 6. Interpretabilidad y Explicabilidad

### 6.1 Modelos Interpretables
- Árboles de decisión: Visualizables
- Regresión lineal: Coeficientes
- Reglas de asociación: Explícitas

### 6.2 Explicabilidad Post-hoc
- **SHAP**: Explicaciones de adiciones de características
- **LIME**: Aproximación local interpretable
- **Permutation Importance**: Importancia de características
- **PDP/ICE Plots**: Relación característica-predicción

### 6.3 Fairness y Bias
- Auditar modelos para sesgo
- Desempenho desigual entre grupos
- Técnicas de mitigación
- Regulación: GDPR right to explanation

## 7. Monitoreo y Mantenimiento

### 7.1 Drift de Datos
- Data Drift: Cambio en distribución X
- Label Drift: Cambio en distribución Y
- Concept Drift: Cambio en relación X→Y
- Detección: Statistical tests, visualización

### 7.2 Degradación del Modelo
- Monitoreo de performance en producción
- Comparación con baseline
- Alertas para reentrenamiento
- Versioning de modelos

### 7.3 Reentrenamiento
- Planificación: Cadencia óptima
- Validación: Garantizar mejora
- Rollback: Plan de contingencia
- A/B testing: Validar en producción