# Aprendizaje Automático No Supervisado: Análisis de Patrones Latentes
## Sistemas de Aprendizaje Automático - Bloque 2

## Resumen

El aprendizaje no supervisado constituye un paradigma fundamental del aprendizaje automático caracterizado por el descubrimiento de estructuras inherentes en datos sin etiquetas predefinidas. Este documento presenta un análisis exhaustivo de los principales algoritmos y metodologías, incluyendo técnicas de agrupamiento (clustering), reducción de dimensionalidad y detección de anomalías. Se examinan fundamentos teóricos, propiedades matemáticas, consideraciones algorítmicas y aplicaciones prácticas de cada método, proporcionando una base rigurosa para el análisis exploratorio de datos complejos.

## 1. Fundamentos del Aprendizaje No Supervisado

### 1.1 Marco Conceptual y Definición Formal

El aprendizaje no supervisado aborda el problema de extraer estructuras significativas de un conjunto de datos $\mathcal{D} = \{x_i\}_{i=1}^{n}$ donde $x_i \in \mathbb{R}^d$ representa observaciones sin etiquetas asociadas. A diferencia del aprendizaje supervisado, no existe función objetivo explícita a optimizar, sino que se buscan representaciones, agrupamientos o transformaciones que revelen organización subyacente de los datos.

Los objetivos principales incluyen:

**Descubrimiento de Estructura**: Identificar agrupamientos naturales, jerarquías o patrones de similitud en los datos.

**Reducción de Dimensionalidad**: Encontrar representaciones compactas que preserven información relevante mientras eliminan redundancia.

**Detección de Anomalías**: Identificar observaciones que se desvían significativamente de patrones normales.

**Aprendizaje de Representaciones**: Descubrir características latentes que faciliten tareas posteriores.

### 1.2 Desafíos Metodológicos

El aprendizaje no supervisado presenta desafíos únicos:

- **Ausencia de Ground Truth**: Sin etiquetas verdaderas, la evaluación se vuelve fundamentalmente subjetiva o dependiente de criterios indirectos
- **Definición de Similitud**: La noción de similitud entre observaciones puede ser específica del dominio y no única
- **Escalabilidad**: Muchos algoritmos tienen complejidad computacional prohibitiva para conjuntos de datos masivos
- **Interpretabilidad**: Los resultados requieren validación mediante conocimiento del dominio

## 2. Algoritmos de Agrupamiento (Clustering)

### 2.1 K-Means Clustering

K-Means constituye el algoritmo de agrupamiento más ampliamente utilizado, basado en particionamiento iterativo del espacio de datos en $K$ clusters disjuntos.

**Formulación Matemática**: El objetivo es minimizar la inercia intra-cluster (within-cluster sum of squares):

$$J = \sum_{k=1}^{K}\sum_{x_i \in C_k}\|x_i - \mu_k\|^2$$

donde $\mu_k$ representa el centroide del cluster $C_k$.

**Algoritmo de Lloyd**:
1. Inicialización: Seleccionar $K$ centroides iniciales (aleatorio, K-Means++, etc.)
2. Asignación: Asignar cada punto al centroide más cercano
3. Actualización: Recalcular centroides como media de puntos asignados
4. Iterar pasos 2-3 hasta convergencia

**Propiedades y Limitaciones**:

*Ventajas*:
- Eficiencia computacional: $O(nKdi)$ donde $i$ es número de iteraciones
- Escalabilidad a conjuntos de datos grandes
- Simplicidad conceptual y de implementación
- Convergencia garantizada a mínimo local

*Limitaciones*:
- Requiere especificar $K$ a priori
- Sensibilidad a inicialización (mitigado por K-Means++)
- Asume clusters esféricos de tamaño similar
- Sensible a outliers
- Limitado a distancia Euclidiana (aunque variantes como K-Medoids usan otras métricas)

**Selección de K**: Métodos incluyen elbow method (punto de inflexión en inercia), silhouette score, gap statistic y información teórico-estadística (BIC, AIC).

### 2.2 Clustering Jerárquico

Construye jerarquía de agrupamientos representada mediante dendrograma, permitiendo visualización de estructura a múltiples escalas de resolución.

**Enfoques Algorítmicos**:

**Agglomerative (Bottom-Up)**:
1. Inicializar cada punto como cluster individual
2. Iterativamente fusionar los dos clusters más similares
3. Continuar hasta obtener un único cluster o criterio de parada

**Divisive (Top-Down)**:
1. Inicializar todos los puntos en un único cluster
2. Recursivamente dividir clusters en subclusters
3. Continuar hasta que cada punto forme cluster individual

**Criterios de Enlace (Linkage)**:

- **Single Linkage**: $d(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x,y)$ (puede producir efecto encadenamiento)
- **Complete Linkage**: $d(C_i, C_j) = \max_{x \in C_i, y \in C_j} d(x,y)$ (clusters más compactos)
- **Average Linkage**: $d(C_i, C_j) = \frac{1}{|C_i||C_j|}\sum_{x \in C_i}\sum_{y \in C_j} d(x,y)$
- **Ward's Method**: Minimiza incremento en varianza intra-cluster

**Ventajas**:
- No requiere especificar número de clusters a priori
- Dendrograma proporciona visualización de estructura jerárquica
- Determinístico (no depende de inicialización aleatoria)

**Limitaciones**:
- Complejidad computacional: $O(n^2\log n)$ para enlace average/complete
- No escala bien a conjuntos de datos muy grandes
- Decisiones de fusión son irrevocables

### 2.3 DBSCAN (Density-Based Spatial Clustering)

DBSCAN identifica clusters como regiones de alta densidad separadas por regiones de baja densidad, permitiendo descubrimiento de clusters de forma arbitraria.

**Conceptos Fundamentales**:

- **ε-vecindad**: $N_\epsilon(x) = \{y \in \mathcal{D} : d(x,y) \leq \epsilon\}$
- **Core point**: Punto con al menos `min_samples` vecinos en $N_\epsilon(x)$
- **Border point**: No es core point pero está en vecindad de core point
- **Noise point**: Ni core ni border point

**Algoritmo**:
1. Para cada punto no visitado, determinar si es core point
2. Si es core point, iniciar nuevo cluster y añadir todos puntos density-reachable
3. Puntos no alcanzables desde core points se clasifican como ruido

**Propiedades Distintivas**:
- No requiere especificar número de clusters
- Identifica clusters de forma arbitraria (no limitado a formas convexas)
- Robusto ante ruido y outliers
- Identifica explícitamente puntos anómalos

**Consideraciones**:
- Requiere selección cuidadosa de $\epsilon$ y `min_samples`
- Dificultad con clusters de densidades variables
- Complejidad $O(n\log n)$ con estructuras de indexación espacial

### 2.4 Modelos de Mezclas Gaussianas (Gaussian Mixture Models)

GMM constituye un enfoque probabilístico que modela distribución de datos como superposición de $K$ distribuciones gaussianas multivariadas:

$$p(x) = \sum_{k=1}^{K}\pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$$

donde $\pi_k$ representa pesos de mezcla ($\sum_k \pi_k = 1$), $\mu_k$ son medias y $\Sigma_k$ matrices de covarianza.

**Algoritmo Expectation-Maximization (EM)**:

**E-step**: Calcular responsabilidades (probabilidades posteriores de pertenencia):
$$\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i|\mu_k, \Sigma_k)}{\sum_{j=1}^{K}\pi_j \mathcal{N}(x_i|\mu_j, \Sigma_j)}$$

**M-step**: Actualizar parámetros maximizando verosimilitud esperada:
$$\mu_k = \frac{\sum_i \gamma_{ik}x_i}{\sum_i \gamma_{ik}}, \quad \Sigma_k = \frac{\sum_i \gamma_{ik}(x_i - \mu_k)(x_i - \mu_k)^T}{\sum_i \gamma_{ik}}$$

**Ventajas**:
- Proporciona pertenencia probabilística (soft clustering)
- Modelo generativo permite muestreo de nuevos puntos
- Flexible: diferentes formas de clusters mediante covarianzas
- Fundamento teórico riguroso

**Limitaciones**:
- Sensible a inicialización
- Puede converger a máximos locales
- Requiere especificar $K$
- Asume forma gaussiana (limitante para distribuciones multimodales complejas)

## 3. Reducción de Dimensionalidad

### 3.1 Análisis de Componentes Principales (PCA)

PCA identifica direcciones de máxima varianza en datos, proyectando observaciones a subespacio de menor dimensión que preserva máxima variabilidad.

**Fundamento Matemático**: Dada matriz de datos centrados $X \in \mathbb{R}^{n \times d}$, PCA busca proyección ortogonal $W \in \mathbb{R}^{d \times k}$ que maximiza varianza:

$$\max_W \text{tr}(W^T\Sigma W) \quad \text{sujeto a } W^TW = I_k$$

donde $\Sigma = \frac{1}{n}X^TX$ es matriz de covarianza muestral.

**Solución**: Los vectores propios de $\Sigma$ correspondientes a los $k$ mayores valores propios forman las columnas de $W$.

**Propiedades**:
- Transformación lineal óptima bajo criterio de reconstrucción de mínimos cuadrados
- Componentes principales son ortogonales (no correlacionados)
- Descomposición: $X = UDV^T$ (SVD)
- Varianza explicada por componente $k$: $\lambda_k / \sum_i \lambda_i$

**Aplicaciones**:
- Compresión de datos
- Visualización mediante proyección a 2D/3D
- Eliminación de multicolinealidad
- Pre-procesamiento para aceleración de algoritmos

**Limitaciones**:
- Asume linealidad de relaciones
- Sensible a escala de variables (requiere normalización)
- Componentes pueden ser difíciles de interpretar
- No preserva distancias locales necesariamente

### 3.2 t-Distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE constituye técnica no lineal de reducción de dimensionalidad especialmente efectiva para visualización, preservando estructura local de datos mediante optimización de distribuciones de probabilidad.

**Fundamento**: Convierte distancias Euclidianas en probabilidades condicionales en espacio original:

$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$

En espacio de menor dimensión, usa distribución t de Student:

$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|y_k - y_l\|^2)^{-1}}$$

Minimiza divergencia KL entre $P$ y $Q$: $\text{KL}(P\|Q) = \sum_{i \neq j} p_{ij}\log\frac{p_{ij}}{q_{ij}}$

**Características**:
- Excelente preservación de estructura local
- Revela clusters y manifolds no evidentes en PCA
- Hiperparámetro perplexity controla balance local/global

**Limitaciones**:
- Computacionalmente intensivo: $O(n^2)$ (variantes como Barnes-Hut reducen a $O(n\log n)$)
- No determinístico (diferentes ejecuciones producen resultados diferentes)
- Distancias en espacio reducido no son interpretables cuantitativamente
- No adecuado como paso de pre-procesamiento general

### 3.3 Autoencoders para Reducción de Dimensionalidad

Autoencoders constituyen arquitecturas de redes neuronales que aprenden representaciones comprimidas mediante reconstrucción:

**Arquitectura**:
- **Encoder**: $h = f_\theta(x)$ mapea entrada a representación latente
- **Decoder**: $\hat{x} = g_\phi(h)$ reconstruye desde representación

**Objetivo**: Minimizar error de reconstrucción: $\mathcal{L} = \|x - g_\phi(f_\theta(x))\|^2$

**Ventajas**:
- Capacidad de capturar no-linealidades complejas
- Flexible: arquitecturas profundas pueden modelar transformaciones complejas
- Variantes (VAE) proporcionan representaciones probabilísticas

**Variantes**:
- **Denoising Autoencoders**: Robustez mediante reconstrucción desde entrada corrupta
- **Variational Autoencoders (VAE)**: Representación latente como distribución probabilística
- **Sparse Autoencoders**: Regularización mediante sparsity en capa latente

## 4. Métricas de Evaluación para Clustering

### 4.1 Índice de Silhouette

Cuantifica cohesión intra-cluster y separación inter-cluster para cada observación:

$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$

donde $a_i$ es distancia promedio a puntos en mismo cluster y $b_i$ es mínima distancia promedio a puntos de otros clusters. Valores en $[-1, 1]$; valores altos indican asignación apropiada.

### 4.2 Índice Davies-Bouldin

Promedia razón entre dispersión intra-cluster y separación inter-cluster:

$$DB = \frac{1}{K}\sum_{k=1}^{K}\max_{k' \neq k}\frac{\sigma_k + \sigma_{k'}}{d(c_k, c_{k'})}$$

Valores menores indican clusters más compactos y separados.

### 4.3 Índice Calinski-Harabasz

Razón entre dispersión inter-cluster e intra-cluster:

$$CH = \frac{\text{tr}(B_K)}{\text{tr}(W_K)} \cdot \frac{n-K}{K-1}$$

Valores mayores indican clustering mejor definido.

## 5. Detección de Anomalías

### 5.1 Isolation Forest

Algoritmo basado en árboles que explota la propiedad de que anomalías son "fáciles de aislar":

- Construye ensemble de árboles de aislamiento mediante splits aleatorios
- Anomalías requieren menos splits para aislamiento (menor profundidad promedio)
- Complejidad $O(n\log n)$, escalable a grandes conjuntos de datos

### 5.2 Local Outlier Factor (LOF)

Cuantifica grado de anomalía basándose en densidad local relativa:

$$LOF_k(x) = \frac{\sum_{y \in N_k(x)} \frac{lrd_k(y)}{lrd_k(x)}}{|N_k(x)|}$$

donde $lrd_k$ es densidad local de accesibilidad. Valores >> 1 indican outliers.

### 5.3 One-Class SVM

Aprende región de alta densidad en espacio de características, clasificando observaciones fuera de esta región como anómalas. Utiliza kernel trick para flexibilidad en formas de decisión.

## Referencias

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
- MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press.
