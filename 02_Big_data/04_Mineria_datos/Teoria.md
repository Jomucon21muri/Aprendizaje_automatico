# Minería de Datos: Metodologías, Algoritmos y Aplicaciones Empresariales
## Big Data - Bloque 4

## Resumen

La minería de datos constituye el proceso sistemático de descubrir patrones, correlaciones y conocimiento accionable en conjuntos de datos masivos mediante técnicas de estadística, machine learning y gestión de bases de datos. Este documento proporciona análisis exhaustivo de metodologías estructuradas (CRISP-DM, SEMMA), técnicas de minería predictiva (clasificación, regresión) y descriptiva (clustering, reglas de asociación, patrones secuenciales), algoritmos fundamentales (Apriori, FP-Growth, K-Means), aplicaciones sectoriales, métricas de evaluación, y la integración contemporánea con big data y deep learning. Se enfatiza la transformación de datos raw en insights estratégicos para toma de decisiones empresariales.

## 1. Fundamentos de Minería de Datos

### 1.1 Definición y Objetivos

**Minería de Datos (Data Mining)**: Proceso de extraer conocimiento útil, no trivial, implícito y potencialmente útil de grandes volúmenes de datos (Fayyad et al., 1996).

**Características**:
- **No trivial**: Patrones no evidentes mediante inspección simple
- **Válidos**: Generalizables a nuevos datos
- **Útiles**: Accionables para decisiones de negocio
- **Comprensibles**: Interpretables por stakeholders

**Objetivos**:
1. **Descripción**: Caracterizar propiedades generales de datos
2. **Predicción**: Inferir valores futuros o desconocidos
3. **Clasificación**: Asignar categorías a instancias
4. **Clustering**: Agrupar objetos similares
5. **Asociación**: Descubrir relaciones entre variables
6. **Detección de Anomalías**: Identificar outliers

### 1.2 Proceso KDD (Knowledge Discovery in Databases)

**Fases**:
1. **Selección**: Identificar conjuntos de datos relevantes
2. **Preprocesamiento**: Limpieza, manejo de missing values
3. **Transformación**: Normalización, discretización, feature engineering
4. **Minería de Datos**: Aplicación de algoritmos
5. **Interpretación/Evaluación**: Validación de patrones, visualización
6. **Despliegue**: Integración en sistemas operacionales

**Minería de Datos** es un subproceso dentro del KDD más amplio.

### 1.3 Tipos de Minería

**Minería Predictiva**:
- **Clasificación**: Variable objetivo categórica (fraude sí/no)
- **Regresión**: Variable objetivo continua (precio vivienda)
- **Forecasting**: Series temporales (demanda futura)

**Minería Descriptiva**:
- **Clustering**: Segmentación sin etiquetas previas
- **Reglas de Asociación**: Relaciones item → item
- **Patrones Secuenciales**: Secuencias temporales frecuentes
- **Resumen**: Descripciones compactas de subconjuntos de datos

## 2. Metodología CRISP-DM

### 2.1 Introducción

**CRISP-DM** (Cross-Industry Standard Process for Data Mining) es el estándar de facto, desarrollado en 1996, usado en ~60% de proyectos de minería de datos (KDnuggets polls).

**Características**:
- **Iterativo**: Fases se repiten según necesidad
- **Flexible**: Adaptable a cualquier industria
- **No prescriptivo**: Guía, no camisa de fuerza
- **Centrado en negocio**: Comienza con objetivos empresariales

### 2.2 Fases CRISP-DM

#### Fase 1: Business Understanding

**Objetivos**:
- Definir objetivos de negocio (aumentar retención 15%)
- Traducir a objetivos de minería de datos (predecir churn con F1 > 0.80)
- Evaluar situación (recursos, restricciones, riesgos)
- Crear plan de proyecto

**Actividades**:
- Entrevistas con stakeholders
- Análisis de KPIs existentes
- Identificación de data sources
- Criterios de éxito cuantificables

**Ejemplo**: E-commerce quiere reducir churn.
- Objetivo negocio: Retener 20% más clientes en Q1 2024
- Objetivo minería: Modelo predicción churn con precision ≥ 0.85, recall ≥ 0.75

#### Fase 2: Data Understanding

**Objetivos**:
- Recolectar datos iniciales
- Explorar datos (EDA - Exploratory Data Analysis)
- Verificar calidad
- Hipótesis preliminares

**Actividades**:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_csv('customer_data.csv')

# Descripción general
print(df.info())
print(df.describe())

# Missing values
print(df.isnull().sum())

# Distribuciones
df['age'].hist(bins=50)
plt.show()

# Correlaciones
sns.heatmap(df.corr(), annot=True)
plt.show()

# Clase desbalanceada?
print(df['churn'].value_counts())  # 15% churn, 85% no churn
```

**Descubrimientos**:
- 15% tasa de churn (desbalance de clases)
- 20% missing values en "income"
- Correlación negativa entre "loyalty_years" y "churn"
- Distribución "purchase_amount" sesgada (outliers)

#### Fase 3: Data Preparation

**Objetivos**:
- Seleccionar datos relevantes
- Limpiar datos (missing values, outliers, duplicados)
- Construir nuevas features
- Formatear para algoritmos

**Actividades**:

**Limpieza**:
```python
# Missing values
df['income'].fillna(df['income'].median(), inplace=True)
df.dropna(subset=['email'], inplace=True)  # Email crítico

# Outliers (IQR method)
Q1 = df['purchase_amount'].quantile(0.25)
Q3 = df['purchase_amount'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['purchase_amount'] < (Q1 - 1.5 * IQR)) | 
          (df['purchase_amount'] > (Q3 + 1.5 * IQR)))]

# Duplicados
df.drop_duplicates(subset=['customer_id'], inplace=True)
```

**Feature Engineering**:
```python
# Recency, Frequency, Monetary (RFM)
df['days_since_last_purchase'] = (pd.Timestamp.now() - df['last_purchase_date']).dt.days
df['purchase_frequency'] = df['total_purchases'] / df['customer_tenure_days']
df['avg_order_value'] = df['total_spent'] / df['total_purchases']

# Ratios
df['discount_usage_rate'] = df['discounts_used'] / df['total_purchases']

# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 45, 65, 100], 
                          labels=['18-25', '26-45', '46-65', '65+'])
```

**Encoding**:
```python
# One-Hot Encoding para categóricas
df = pd.get_dummies(df, columns=['region', 'subscription_type'], drop_first=True)

# Label Encoding para ordinales
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['satisfaction_level'] = le.fit_transform(df['satisfaction'])  # low, medium, high → 0,1,2
```

**Normalización**:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numeric_features = ['age', 'income', 'total_spent', 'days_since_last_purchase']
df[numeric_features] = scaler.fit_transform(df[numeric_features])
```

**Train/Test Split**:
```python
from sklearn.model_selection import train_test_split

X = df.drop(['customer_id', 'churn'], axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y  # Stratify mantiene proporción de clases
)
```

#### Fase 4: Modeling

**Objetivos**:
- Seleccionar técnicas de modelado
- Generar test design
- Construir modelos
- Evaluar modelos

**Técnicas para Clasificación**:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, scale_pos_weight=5),  # Para desbalance
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
}

# Entrenar y evaluar
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name}: Accuracy={accuracy_score(y_test, y_pred):.3f}, F1={f1_score(y_test, y_pred):.3f}")
```

**Hyperparameter Tuning**:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best F1: {grid_search.best_score_:.3f}")

best_model = grid_search.best_estimator_
```

**Ensemble Methods**:
```python
from sklearn.ensemble import VotingClassifier

# Voting ensemble (combina predicciones de múltiples modelos)
ensemble = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('gb', GradientBoostingClassifier(n_estimators=100)),
        ('xgb', XGBClassifier(n_estimators=100))
    ],
    voting='soft'  # Usa probabilidades
)

ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)
```

#### Fase 5: Evaluation

**Objetivos**:
- Evaluar resultados contra objetivos de negocio
- Revisar proceso completo
- Determinar próximos pasos

**Métricas de Clasificación**:

**Confusion Matrix**:
```
                Predicted
                No Churn  Churn
Actual No Churn    TN       FP
       Churn       FN       TP
```

**Métricas**:
- **Accuracy**: (TP + TN) / Total - Útil si clases balanceadas
- **Precision**: TP / (TP + FP) - De los predichos como churn, cuántos realmente churn
- **Recall (Sensitivity)**: TP / (TP + FN) - De los churn reales, cuántos detectamos
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall) - Media armónica
- **AUC-ROC**: Área bajo curva ROC (trade-off FPR vs TPR)

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")

# ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'AUC={roc_auc_score(y_test, y_proba):.3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

**Feature Importance**:
```python
importances = best_model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))
```

**Business Impact**:
- Modelo detecta 75% de churners (Recall=0.75)
- 85% de predicciones de churn son correctas (Precision=0.85)
- Campaña de retención cuesta $50 por cliente
- Valor de cliente retenido: $500
- ROI = (0.75 * 10000 churners * $500) - (0.75 * 10000 * $50) = $3.375M

#### Fase 6: Deployment

**Objetivos**:
- Planificar despliegue
- Monitoreo y mantenimiento
- Documentación
- Revisión final

**Estrategias de Despliegue**:

**Batch Scoring**:
```python
# Diariamente, predecir churn en todos los clientes activos
import schedule

def daily_scoring():
    customers = load_active_customers()
    customers_features = preprocess(customers)
    predictions = model.predict_proba(customers_features)[:, 1]
    
    high_risk = customers[predictions > 0.7]
    send_to_retention_team(high_risk)

schedule.every().day.at("02:00").do(daily_scoring)
```

**Real-Time API**:
```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('churn_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = preprocess_single(data)
    prediction = model.predict_proba([features])[0, 1]
    return jsonify({'churn_probability': float(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Monitoring**:
```python
# Detectar data drift
from scipy.stats import ks_2samp

def check_drift(feature_name, training_data, production_data):
    statistic, p_value = ks_2samp(training_data[feature_name], production_data[feature_name])
    if p_value < 0.05:
        alert(f"Data drift detected in {feature_name}: p={p_value}")

# Performance decay
def monitor_performance(y_true, y_pred):
    current_f1 = f1_score(y_true, y_pred)
    if current_f1 < 0.70:  # Threshold original era 0.75
        alert(f"Model performance degraded: F1={current_f1}")
        trigger_retraining()
```

### 2.3 Metodología SEMMA

**SEMMA** (SAS): Sample, Explore, Modify, Model, Assess
- Similar a CRISP-DM pero más técnica, menos business-oriented
- Sample: Muestreo de datos
- Explore: EDA
- Modify: Preparación y transformación
- Model: Construcción de modelos
- Assess: Evaluación

## 3. Técnicas de Minería Predictiva

### 3.1 Clasificación

**Algoritmos Principales**:

**Decision Trees**:
- **Ventajas**: Interpretables, manejo nativo de categóricas, no paramétricos
- **Desventajas**: Overfitting, inestables (pequeños cambios en datos → árbol diferente)
- **Criterios**: Gini impurity, Information Gain (Entropy)

$$\text{Gini}(D) = 1 - \sum_{i=1}^{k} p_i^2$$

donde $p_i$ es proporción de clase $i$ en dataset $D$.

**Random Forest**:
- Ensemble de árboles decorrelacionados (bagging + feature randomness)
- Reduce overfitting vs single tree
- OOB (Out-of-Bag) error para validación

**Gradient Boosting** (XGBoost, LightGBM, CatBoost):
- Ensemble secuencial: cada árbol corrige errores del anterior
- Altamente predictivo, ganador de competencias Kaggle
- Requiere tuning cuidadoso (learning rate, max depth, regularización)

**Support Vector Machines (SVM)**:
- Encuentra hiperplano óptimo que maximiza margen entre clases
- Kernel trick: mapea datos a dimensión superior
- Efectivo en alta dimensionalidad

**Naive Bayes**:
- Basado en Teorema de Bayes con independencia condicional (asunción "naive")
- Rápido, funciona bien con textos (spam detection)
- Robusto a features irrelevantes

**K-Nearest Neighbors (KNN)**:
- Clasifica basado en mayoría de K vecinos más cercanos
- No hay fase de entrenamiento (lazy learning)
- Sensible a escala de features y dimensionalidad

### 3.2 Regresión

**Linear Regression**:
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon$$

**Ridge Regression (L2 regularization)**:
$$\text{Loss} = \text{MSE} + \alpha \sum_{i=1}^{n} \beta_i^2$$

**Lasso Regression (L1 regularization)**:
$$\text{Loss} = \text{MSE} + \alpha \sum_{i=1}^{n} |\beta_i|$$

Lasso realiza feature selection (algunos $\beta_i$ → 0).

**Regresión para Predicción**:
```python
from sklearn.ensemble import RandomForestRegressor

# Predecir precio de vivienda
model = RandomForestRegressor(n_estimators=100, max_depth=15)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print(f"MAE: ${mean_absolute_error(y_test, y_pred):,.0f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")
```

## 4. Técnicas de Minería Descriptiva

### 4.1 Clustering

**K-Means**:
```python
from sklearn.cluster import KMeans

# Segmentación de clientes
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(df[['recency', 'frequency', 'monetary']])

# Visualización (PCA para reducir a 2D)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df[['recency', 'frequency', 'monetary']])

plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['cluster'], cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Customer Segments')
plt.colorbar(label='Cluster')
plt.show()
```

**Selección de K**:
- **Elbow Method**: Plot inertia vs K, buscar "codo"
- **Silhouette Score**: Mide cohesión y separación

```python
from sklearn.metrics import silhouette_score

inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, labels))

plt.plot(K_range, inertias, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

**Hierarchical Clustering**:
- Agrupa iterativamente pares más similares
- Produce dendrogram (árbol jerárquico)
- No requiere especificar K a priori

**DBSCAN**:
- Density-Based Spatial Clustering
- Identifica clusters de forma arbitraria
- Detecta outliers (noise points)
- Parámetros: eps (radio), min_samples

### 4.2 Reglas de Asociación

**Conceptos**:
- **Itemset**: Conjunto de items {leche, pan, huevos}
- **Regla**: $X \Rightarrow Y$ (si compra pan y leche, compra huevos)
- **Support**: Proporción de transacciones que contienen X ∪ Y
  $$\text{supp}(X \Rightarrow Y) = \frac{|\{T: X \cup Y \subseteq T\}|}{|T|}$$

- **Confidence**: Probabilidad condicional P(Y|X)
  $$\text{conf}(X \Rightarrow Y) = \frac{\text{supp}(X \cup Y)}{\text{supp}(X)}$$

- **Lift**: Indica si X y Y son independientes
  $$\text{lift}(X \Rightarrow Y) = \frac{\text{conf}(X \Rightarrow Y)}{\text{supp}(Y)}$$
  - Lift > 1: Correlación positiva
  - Lift = 1: Independientes
  - Lift < 1: Correlación negativa

**Algoritmo Apriori**:

**Pseudocódigo**:
```
1. L1 = {frequent 1-itemsets con supp ≥ min_supp}
2. for k = 2 to n:
3.     Ck = candidate k-itemsets generados de Lk-1
4.     for cada transacción T:
5.         Ct = subconjuntos de T que están en Ck
6.         incrementar conteo de itemsets en Ct
7.     Lk = itemsets en Ck con supp ≥ min_supp
8. return ∪k Lk
```

**Principio Apriori**: Si itemset es infrecuente, todos sus supersets también lo son (poda de búsqueda).

**Implementación Python**:
```python
from mlxtend.frequent_patterns import apriori, association_rules

# Datos: transacciones en formato one-hot
basket = pd.get_dummies(df.groupby(['transaction_id', 'item'])['item'].count().unstack().reset_index().fillna(0).set_index('transaction_id'))
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# Frequent itemsets
frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)

# Reglas de asociación
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules = rules[rules['lift'] > 1]  # Solo reglas con correlación positiva

print(rules.sort_values('lift', ascending=False).head(10))
```

**FP-Growth**:
- Más eficiente que Apriori (no genera candidates explícitamente)
- Construye FP-Tree (Frequent Pattern Tree)
- Mining recursivo sobre estructura compacta

**Aplicaciones**:
- **Market Basket Analysis**: Recomendaciones de productos
- **Cross-selling**: "Customers who bought X also bought Y"
- **Store Layout**: Ubicar productos relacionados cerca

### 4.3 Patrones Secuenciales

**Sequential Pattern Mining**: Descubrir secuencias frecuentes en datos temporales.

**Ejemplo**: E-commerce
- Secuencia: (Login) → (Browse Category: Electronics) → (Add to Cart: Laptop) → (Purchase)
- Patrón frecuente: 60% usuarios que navegan Electronics compran laptop dentro de 7 días

**Algoritmos**:
- **GSP** (Generalized Sequential Pattern): Extensión de Apriori para secuencias
- **PrefixSpan**: Projection-based, más eficiente

**Aplicaciones**:
- **Navegación web**: Optimizar flujo de usuario
- **Análisis de logs**: Detectar secuencias de errores que preceden fallos
- **Bioinformática**: Motifs en secuencias de ADN

## 5. Aplicaciones Empresariales

### 5.1 Retail y E-Commerce

**Customer Segmentation** (RFM Analysis):
- **Recency**: Días desde última compra
- **Frequency**: Número de compras
- **Monetary**: Valor total gastado

**Segmentos**:
- **Champions**: Alta F, alta M, baja R → Marketing de fidelización
- **At Risk**: Alta F y M históricamente, alta R → Campañas de reactivación
- **Lost**: Muy alta R → Descuentos agresivos o ignorar

**Churn Prediction**:
- Identificar clientes con alta probabilidad de abandonar
- Intervención proactiva (descuentos, atención personalizada)

**Recommendation Systems**:
- **Collaborative Filtering**: "Users similar to you liked..."
- **Content-Based**: Basado en atributos de productos
- **Hybrid**: Combinación

### 5.2 Finanzas

**Credit Scoring**:
- Predecir probabilidad de default en préstamos
- Features: Ingresos, historial crediticio, deuda actual, empleo
- Modelos: Logistic Regression, Gradient Boosting

**Fraud Detection**:
- Detección de transacciones fraudulentas en tiempo real
- Techniques: Anomaly Detection, clasificación desbalanceada
- Features: Monto, ubicación, hora, patrón de gasto del usuario

**Algorithmic Trading**:
- Patrones en series temporales de precios
- Features: Moving averages, RSI, MACD, volumen
- Predicción de tendencias

### 5.3 Telecomunicaciones

**Churn Prediction**:
- Predecir cancelaciones de suscripciones
- Features: Uso de datos, llamadas de atención al cliente, quejas, competencia local

**Network Optimization**:
- Clustering de torres celulares por patrones de tráfico
- Asignación eficiente de recursos

**Upselling/Cross-selling**:
- Recomendar planes premium o servicios adicionales
- Basado en uso actual y perfil demográfico

### 5.4 Healthcare

**Disease Prediction**:
- Clasificación de riesgo de enfermedades (diabetes, cardiovascular)
- Features: Edad, IMC, historial familiar, biomarcadores

**Readmission Prediction**:
- Predecir reingresos hospitalarios post-alta
- Optimizar seguimiento y recursos

**Drug Discovery**:
- Mining de datasets moleculares para identificar candidatos a fármacos
- Patrones en interacciones proteína-ligando

### 5.5 Marketing

**Customer Lifetime Value (CLV)**:
- Predecir valor total que generará un cliente
- Optimizar inversión en adquisición y retención

**Campaign Response Prediction**:
- Probabilidad de responder a campañas marketing
- Segmentación para targeting

**Sentiment Analysis**:
- Minería de texto en redes sociales
- Detectar percepciones de marca

## 6. Evaluación y Validación

### 6.1 Métricas por Tarea

**Clasificación**:
- Accuracy, Precision, Recall, F1, AUC-ROC (ya discutidas)
- **Matthews Correlation Coefficient (MCC)**: Robusto a desbalance
  $$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

**Regresión**:
- **MAE** (Mean Absolute Error): Promedio de errores absolutos
- **RMSE** (Root Mean Squared Error): Penaliza errores grandes
- **R²**: Proporción de varianza explicada (0 a 1, mayor es mejor)
- **MAPE** (Mean Absolute Percentage Error): Error relativo

**Clustering**:
- **Silhouette Score**: -1 a 1, mayor es mejor
- **Davies-Bouldin Index**: Menor es mejor
- **Adjusted Rand Index**: Compara con ground truth si existe

### 6.2 Técnicas de Validación

**Holdout**: Train 70-80%, Test 20-30%
- Simple, pero estimador de performance tiene alta varianza

**K-Fold Cross-Validation**:
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"F1 scores: {scores}")
print(f"Mean F1: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

**Stratified K-Fold**: Mantiene proporción de clases en cada fold (crítico para datos desbalanceados).

**Leave-One-Out (LOO)**: K = N (cada instancia es un fold)
- Bajo bias, alto variance, computacionalmente costoso

**Time Series Validation**:
- No aleatorizar (violaría temporalidad)
- **Rolling Window**: Entrenar en [t-n, t], predecir t+1
- **Expanding Window**: Entrenar en [0, t], predecir t+1

### 6.3 Overfitting y Regularización

**Overfitting**: Modelo memoriza datos de entrenamiento, pobre generalización.

**Detección**:
- Train accuracy >> Test accuracy
- Learning curves: Error training baja, error validation estancado

**Mitigación**:
- **Más datos**: Reduce overfitting
- **Feature Selection**: Eliminar features irrelevantes
- **Regularización**: L1, L2, Dropout (redes neuronales)
- **Early Stopping**: Detener entrenamiento cuando validation error empieza a aumentar
- **Ensemble Methods**: Random Forest, Bagging

## 7. Integración con Big Data

### 7.1 Minería de Datos a Escala

**Desafíos**:
- Datos no caben en memoria de una máquina
- Algoritmos tradicionales no escalan (O(n²), O(n³))

**Soluciones**:

**Spark MLlib**:
```python
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler

# Datos distribuidos en cluster Spark
df_spark = spark.read.parquet("hdfs://data/customers.parquet")

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_spark = assembler.transform(df_spark)

rf = RandomForestClassifier(featuresCol="features", labelCol="churn", numTrees=100)
model = rf.fit(df_spark)

predictions = model.transform(df_spark)
```

**Sampling Inteligente**:
- Entrenar en muestra representativa (stratified sampling)
- Validar en datos completos

**Approximate Algorithms**:
- MinHash para similitud
- Bloom filters para membership queries
- Sketches (Count-Min, HyperLogLog)

### 7.2 Feature Stores

**Feast** (Open-source):
- Repositorio centralizado de features
- Consistencia entre training y serving
- Versionado y lineage

```python
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# Online serving (baja latencia)
features = store.get_online_features(
    features=["user_rfm:recency", "user_rfm:frequency", "user_rfm:monetary"],
    entity_rows=[{"user_id": 12345}]
).to_dict()

# Offline (training)
training_df = store.get_historical_features(
    entity_df=entities,
    features=["user_rfm:recency", "user_rfm:frequency", "user_rfm:monetary"]
).to_df()
```

## 8. Tendencias Contemporáneas

### 8.1 AutoML

**Automated Machine Learning**: Automatización de pipeline completo.

**Componentes**:
- **Automated Feature Engineering**: Featuretools, tsfresh
- **Neural Architecture Search (NAS)**: Diseño automático de redes neuronales
- **Hyperparameter Optimization**: Bayesian Optimization, Hyperband
- **Model Selection**: Evaluación automática de múltiples algoritmos

**Herramientas**:
- **Auto-sklearn**: Basado en scikit-learn
- **TPOT**: Genetic Programming para pipelines
- **H2O AutoML**: Comercial/open-source
- **Google Cloud AutoML**: Servicio cloud

```python
from tpot import TPOTClassifier

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)

print(f"Test Accuracy: {tpot.score(X_test, y_test):.3f}")
tpot.export('best_pipeline.py')  # Exporta código del mejor pipeline
```

### 8.2 Deep Learning para Data Mining

**Autoencoders** para detección de anomalías:
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Autoencoder
input_layer = Input(shape=(X_train.shape[1],))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(X_train.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, validation_split=0.2)

# Anomalías: instancias con alto reconstruction error
X_test_pred = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)
anomalies = X_test[mse > threshold]
```

**Graph Neural Networks (GNNs)**:
- Minería de datos en grafos (redes sociales, knowledge graphs)
- Predicción de enlaces, clasificación de nodos

**Transformers para Series Temporales**:
- Modelos de atención para forecasting
- Capturan dependencias de largo plazo

### 8.3 Explainable AI (XAI) en Data Mining

**SHAP** (SHapley Additive exPlanations):
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Importancia global
shap.summary_plot(shap_values, X_test)

# Explicación individual
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

**LIME** (Local Interpretable Model-agnostic Explanations):
- Explica predicción individual mediante modelo local interpretable

**Motivación**:
- Regulaciones (GDPR - derecho a explicación)
- Confianza en decisiones críticas (salud, finanzas)
- Debugging de modelos

### 8.4 Privacy-Preserving Data Mining

**Federated Learning**:
- Entrenar modelo sin centralizar datos
- Cada nodo entrena localmente, comparte gradientes
- Aplicación: Hospitals colaboran sin compartir datos de pacientes

**Differential Privacy**:
- Añadir ruido a datos o resultados
- Garantías matemáticas de privacidad

**Homomorphic Encryption**:
- Computación sobre datos encriptados
- Predicciones sin descifrar datos

## 9. Best Practices

**Entender el Negocio**:
- Minería de datos es medio, no fin
- Alineación constante con objetivos empresariales

**Data Quality**:
- "Garbage in, garbage out"
- Invertir tiempo en limpieza y validación

**Feature Engineering**:
- A menudo más impactante que selección de algoritmo
- Conocimiento del dominio es clave

**Simplicidad**:
- Modelo simple interpretable > modelo complejo marginalmente mejor
- Navaja de Occam

**Validación Rigurosa**:
- Cross-validation, múltiples métricas
- Evitar data leakage (usar info que no estará en producción)

**Monitoreo Post-Despliegue**:
- Performance decay por data drift, concept drift
- Re-entrenamiento periódico

**Documentación**:
- Decisiones, experimentos, resultados
- Reproducibilidad

**Ética**:
- Sesgos en datos → sesgos en modelos
- Fairness, accountability, transparency

## Referencias

- Fayyad, U., Piatetsky-Shapiro, G., & Smyth, P. (1996). From Data Mining to Knowledge Discovery in Databases. *AI Magazine*, 17(3), 37-54.
- Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann.
- Witten, I. H., Frank, E., Hall, M. A., & Pal, C. J. (2016). *Data Mining: Practical Machine Learning Tools and Techniques* (4th ed.). Morgan Kaufmann.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Chapman, P., et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS Inc.
- Agrawal, R., & Srikant, R. (1994). Fast Algorithms for Mining Association Rules. *VLDB*, 487-499.
- Tan, P. N., Steinbach, M., & Kumar, V. (2018). *Introduction to Data Mining* (2nd ed.). Pearson.
- Provost, F., & Fawcett, T. (2013). *Data Science for Business*. O'Reilly.
