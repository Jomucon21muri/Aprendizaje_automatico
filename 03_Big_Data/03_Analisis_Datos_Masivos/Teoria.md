# Contenido Teórico: Análisis de Datos Masivos
## Big Data - Bloque 3

## 1. Preparación de Datos a Escala

### 1.1 ETL vs ELT
- **ETL**: Extract → Transform → Load (tradicional)
- **ELT**: Extract → Load → Transform (Data Lake)
- Cambio paradigma en Big Data

### 1.2 Limpieza en Escala
- Valores faltantes: imputación distribuida
- Outliers: métodos robustos
- Deduplicación: expensive en masivo
- Normalización: por particiones

### 1.3 Feature Engineering Distribuido
- Derivar variables significativas
- Encoding de categóricas
- Estadísticas por grupos
- Ventanas temporales

## 2. Análisis Exploratorio de Datos Masivos

### 2.1 Estadística Descriptiva
- Media, mediana, desviación en paralelo
- Percentiles aproximados
- Histogramas distribuidos

### 2.2 Correlación y Relaciones
- Cálculo distribuido de covarianza
- Detección de multicolinealidad
- Feature importance

### 2.3 Detección de Anomalías
- Isolation Forest distribuido
- LOF (Local Outlier Factor)
- Basadas en distancia
- Time series anomaly detection

## 3. Machine Learning Distribuido

### 3.1 Algoritmos Escalables
- Regresión logística distribuida
- Tree-based: Random Forest, Gradient Boosting
- K-Means escalable
- Matrix Factorization para recomendaciones

### 3.2 Librerías de ML Distribuido
- **Spark MLlib**: Integrado en Spark
- **XGBoost distribuido**: Para gradient boosting
- **H2O**: Machine Learning distribuida
- **Dask-ML**: ML con Dask

### 3.3 Consideraciones de Escala
- Convergencia con datos masivos
- Comunicación entre nodos
- Memory efficiency
- Sampling strategies

## 4. Análisis Temporal y Series

### 4.1 Time Series Processing
- Window operations: tumbling, sliding
- Agregaciones por período
- Trend detection
- Seasonality analysis

### 4.2 Stream Analytics
- Estadísticas en tiempo real
- Detección de eventos
- Correlaciones tiempo-real
- CEP (Complex Event Processing)

## 5. Visualización y Business Intelligence

### 5.1 Dashboards Interactivos
- Kibana (ELK Stack)
- Tableau + Spark
- Grafana
- Apache Superset

### 5.2 Reporting Automatizado
- Alertas basadas en umbrales
- Reports programados
- Data storytelling
- Exportación de insights

### 5.3 Escalabilidad de Visualización
- Pre-agregación de datos
- Sampling para interactividad
- Worst-case design
- Progressive loading
