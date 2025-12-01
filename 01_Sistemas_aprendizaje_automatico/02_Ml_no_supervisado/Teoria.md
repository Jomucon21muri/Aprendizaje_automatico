# Contenido teórico: machine learning no supervisado
## Sistemas de aprendizaje automático - Bloque 2

## 1. Introducción al aprendizaje no supervisado

### 1.1 Definición
Algoritmos que descubren patrones en datos sin etiquetas predefinidas.

### 1.2 Aplicaciones
- Segmentación de clientes
- Descubrimiento de patrones
- Compresión de datos
- Detección de fraudes

## 2. Algoritmos de clustering

### 2.1 K-Means
- Particiona datos en k clusters
- Minimiza varianza dentro de clusters
- Sensible a inicialización
- Requiere especificar k

### 2.2 Clustering jerárquico
- Dendrograma muestra relaciones
- Aglomerativo (bottom-up) o divisivo (top-down)
- No requiere especificar número de clusters

### 2.3 DBSCAN
- Basado en densidad
- No requiere especificar número de clusters
- Identifica puntos de ruido
- Parámetros: eps, min_samples

### 2.4 Modelos gaussianos mixtos (GMM)
- Mezcla de distribuciones gaussianas
- Probabilístico
- EM (Expectation-Maximization)

## 3. Reducción de dimensionalidad

### 3.1 Análisis de componentes principales (PCA)
- Transforma a nuevo espacio
- Maximiza varianza explicada
- Componentes ortogonales

### 3.2 t-SNE
- Para visualización en 2D/3D
- Preserva estructuras locales
- Computacionalmente intensivo

### 3.3 AutoEncoders
- Redes neuronales para compresión
- Reconstruye entrada desde representación comprimida
- Flexibles y poderosos

## 4. Evaluación sin supervisión

### 4.1 Índice de silhueta
Mide cuán similar es un punto a su cluster respecto a otros.

### 4.2 Davies-Bouldin
Razón promedio entre dispersión intra-cluster e inter-cluster.

### 4.3 Índice Calinski-Harabasz
Relación de dispersión entre clusters vs dentro de clusters.

## 5. Detección de anomalías
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
