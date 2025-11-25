# Contenido Teórico: Redes Neuronales
## Inteligencia Artificial - Bloque 3

## 1. Fundamentos Biológicos

### 1.1 La Neurona Biológica
- Soma (cuerpo celular)
- Dendritas (reciben señales)
- Axón (transmite señales)
- Sinapsis (conexiones entre neuronas)

### 1.2 Principio de Funcionamiento
Las neuronas se comunican mediante impulsos eléctricos que se fortalecen o debilitan según la actividad.

## 2. La Neurona Artificial

### 2.1 Modelo Matemático
```
Y = f(∑(Wi × Xi) + b)
```
Donde:
- Wi = pesos
- Xi = entradas
- b = sesgo (bias)
- f = función de activación

### 2.2 Funciones de Activación
- ReLU (Rectified Linear Unit)
- Sigmoid
- Tanh
- Softmax

## 3. Arquitecturas de Redes Neuronales

### 3.1 Redes de Alimentación Adelante (Feedforward)
Información fluye unidireccionalmente desde entrada a salida.

### 3.2 Redes Convolucionales (CNN)
Especializadas en procesamiento de imágenes mediante filtros convolucionales.

### 3.3 Redes Recurrentes (RNN)
Tienen retroalimentación, ideales para datos secuenciales y series temporales.

## 4. Entrenamiento de Redes Neuronales

### 4.1 Algoritmo de Backpropagation
- Propagación hacia adelante
- Cálculo de error
- Propagación hacia atrás
- Actualización de pesos

### 4.2 Optimizadores Comunes
- Descenso de Gradiente Estocástico (SGD)
- Adam
- RMSprop
- Adagrad

## 5. Regularización y Validación
- Dropout
- Normalización por lotes (Batch Normalization)
- Validación cruzada
- Early stopping
