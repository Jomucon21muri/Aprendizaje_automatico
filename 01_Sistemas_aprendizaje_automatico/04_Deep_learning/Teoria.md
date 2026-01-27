# Aprendizaje Profundo: Arquitecturas Neuronales y Representaciones Jerárquicas
## Sistemas de Aprendizaje Automático - Bloque 4

## Resumen

El aprendizaje profundo (Deep Learning) representa un paradigma transformador dentro del aprendizaje automático, caracterizado por el uso de redes neuronales artificiales con múltiples capas de procesamiento. Este documento examina los fundamentos teóricos, arquitecturas principales, metodologías de entrenamiento y aplicaciones del aprendizaje profundo. Se analizan redes convolucionales (CNNs), redes recurrentes (RNNs/LSTMs), autoencoders y redes generativas adversariales (GANs), proporcionando una comprensión integral de las técnicas que han revolucionado campos como visión computacional, procesamiento del lenguaje natural y generación de contenido.

## 1. Fundamentos Teóricos del Aprendizaje Profundo

### 1.1 Distinción entre Aprendizaje Superficial y Profundo

El aprendizaje profundo se distingue del aprendizaje superficial (shallow learning) mediante características arquitectónicas y capacidades representacionales:

**Aprendizaje Superficial**:
- Arquitecturas con 1-2 capas ocultas
- Requiere ingeniería manual de características (feature engineering)
- Representaciones relativamente simples
- Limitación en captura de abstracciones complejas

**Aprendizaje Profundo**:
- Arquitecturas con 3 o más capas ocultas (típicamente decenas o cientos)
- Aprendizaje automático de representaciones jerárquicas
- Capacidad para abstracciones de alto nivel
- Eliminación progresiva de necesidad de feature engineering manual

**Teorema de Aproximación Universal**: Mientras que redes de una capa pueden aproximar cualquier función continua dado suficientes neuronas, redes profundas logran representaciones más eficientes con menor cantidad de parámetros para funciones complejas.

### 1.2 Aprendizaje de Representaciones Jerárquicas

La arquitectura profunda permite construcción automática de jerarquías de características:

**Capas Iniciales (Lower Layers)**:
- Detectan características primitivas: bordes, gradientes, texturas básicas
- Filtros simples, localizados espacialmente
- Alta frecuencia de activación
- Transferibles entre dominios

**Capas Intermedias**:
- Composiciones de características primitivas: esquinas, contornos, patrones texturales complejos
- Mayor campo receptivo
- Representaciones más abstractas

**Capas Superiores (Higher Layers)**:
- Conceptos de alto nivel: partes de objetos, escenas completas
- Máxima abstracción semántica
- Específicas al dominio y tarea
- Directamente relacionadas con categorías de salida

Esta organización jerárquica emula procesamiento visual cortical en sistemas biológicos, donde información progresa de áreas V1 (bordes simples) hacia áreas superiores (reconocimiento de objetos).

### 1.3 Retropropagación y Optimización en Redes Profundas

**Algoritmo de Retropropagación (Backpropagation)**:
Método fundamental para entrenar redes neuronales mediante cálculo eficiente de gradientes usando regla de la cadena:

$$\frac{\partial \mathcal{L}}{\partial w_{ij}^{(l)}} = \frac{\partial \mathcal{L}}{\partial z_j^{(l)}} \cdot \frac{\partial z_j^{(l)}}{\partial w_{ij}^{(l)}} = \delta_j^{(l)} \cdot a_i^{(l-1)}$$

**Desafíos en Redes Profundas**:
- **Vanishing Gradient**: Gradientes se desvanecen exponencialmente en capas profundas (especialmente con sigmoides/tanh)
- **Exploding Gradient**: Gradientes crecen descontroladamente, causando inestabilidad numérica

**Soluciones Arquitectónicas**:
- Funciones de activación ReLU y variantes
- Normalización de batch/layer
- Conexiones residuales (skip connections)
- Inicialización cuidadosa de pesos (He, Xavier/Glorot)

## 2. Redes Neuronales Convolucionales (CNNs)

### 2.1 Operación de Convolución y Motivación Biológica

La operación de convolución constituye el núcleo de las CNNs, inspirada en campos receptivos visuales:

$$Y_{i,j} = \sum_{m}\sum_{n} X_{i+m, j+n} \cdot K_{m,n} + b$$

donde $K$ representa el kernel/filtro, $X$ la entrada, $Y$ el mapa de características resultante.

**Propiedades Fundamentales**:
- **Conectividad Local**: Cada neurona conecta solo a región local de entrada (campo receptivo)
- **Compartición de Parámetros**: Mismo filtro aplicado en toda la imagen
- **Invariancia Traslacional**: Detecta características independientemente de posición
- **Reducción Paramétrica**: Drásticamente menos parámetros que redes fully-connected

### 2.2 Componentes Arquitectónicos de CNNs

**Capas Convolucionales**:
- Múltiples filtros aprend ibles detectan características distintas
- Stride controla desplazamiento del filtro
- Padding preserva dimensiones espaciales
- Múltiples canales de entrada/salida

**Funciones de Activación**:
- **ReLU**: $f(x) = \max(0, x)$ - más común, mitiga vanishing gradient
- **Leaky ReLU**: $f(x) = \max(\alpha x, x)$ - permite gradientes para valores negativos
- **ELU, SELU**: Variantes con propiedades auto-normalizantes

**Capas de Pooling**:
- **Max Pooling**: Toma valor máximo en ventana local
- **Average Pooling**: Promedia valores en ventana
- Reduce dimensionalidad espacial
- Proporciona invarianza traslacional adicional
- Controla sobreajuste mediante downsampling

**Capas Fully-Connected (Dense)**:
- Típicamente al final de arquitectura
- Realizan clasificación/regresión final
- Todas las neuronas conectadas a capa previa

### 2.3 Arquitecturas Icónicas y su Evolución

**LeNet-5 (LeCun et al., 1998)**:
- Primera CNN exitosa para reconocimiento de dígitos
- 7 capas, ~60K parámetros
- Arquitectura: Conv → Pool → Conv → Pool → FC → FC
- Estableció paradigma convolucional

**AlexNet (Krizhevsky et al., 2012)**:
- Breakthrough en ImageNet 2012 (error top-5: 15.3%)
- 8 capas, 60M parámetros
- Innovaciones: ReLU, Dropout, Data Augmentation, GPU training
- Reignició investigación en Deep Learning

**VGGNet (Simonyan & Zisserman, 2014)**:
- Arquitectura simple y profunda (16-19 capas)
- Filtros pequeños (3×3) stacked repetidamente
- 138M parámetros (VGG-16)
- Demostró efectividad de profundidad

**ResNet (He et al., 2015)**:
- Conexiones residuales: $H(x) = F(x) + x$
- Permite entrenar redes extremadamente profundas (152+ capas)
- Mitigación definitiva de vanishing gradient
- Won ImageNet 2015 (error top-5: 3.6%, superando humanos)

**Inception/GoogLeNet**:
- Módulos Inception: procesamiento multi-escala paralelo
- Reducción de parámetros mediante convoluciones 1×1
- 22 capas con solo 7M parámetros

**EfficientNet**:
- Escalamiento compuesto (profundidad, ancho, resolución)
- Estado del arte en eficiencia parámetro/rendimiento

### 2.4 Aplicaciones de CNNs

**Clasificación de Imágenes**: Asignación de categorías a imágenes completas

**Detección de Objetos**:
- **R-CNN**: Region proposals + CNN classification
- **Fast/Faster R-CNN**: Mejoras en eficiencia
- **YOLO (You Only Look Once)**: Detección en tiempo real single-pass
- **SSD, RetinaNet**: Variantes modernas

**Segmentación Semántica**:
- **FCN (Fully Convolutional Networks)**: Predicción pixel-wise
- **U-Net**: Arquitectura encoder-decoder para imágenes médicas
- **DeepLab**: Dilated convolutions y CRF

**Reconocimiento Facial**: FaceNet, DeepFace - embeddings para verificación e identificación

## 3. Redes Neuronales Recurrentes y Arquitecturas Secuenciales

### 3.1 Fundamentos de RNNs

Las RNNs procesan secuencias mediante conexiones recurrentes que mantienen estado oculto:

$$h_t = f(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$
$$y_t = W_{hy}h_t + b_y$$

**Propiedades**:
- Compartición de parámetros temporalmente
- Procesamiento de secuencias de longitud variable
- Memoria de contexto pasado

**Limitaciones del RNN Básico**:
- **Vanishing Gradient en el Tiempo**: Gradientes decaen exponencialmente, dificultando aprendizaje de dependencias largas
- **Exploding Gradient**: Mitigado mediante gradient clipping

### 3.2 Long Short-Term Memory (LSTM)

Arquitectura diseñada específicamente para capturar dependencias a largo plazo mediante mecanismos de compuertas (gates):

**Componentes del LSTM**:

- **Forget Gate**: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ - decide qué información descartar
- **Input Gate**: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ - decide qué información nueva almacenar
- **Cell State Update**: $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$ - candidato para nuevo estado
- **Cell State**: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ - memoria a largo plazo
- **Output Gate**: $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ - controla salida
- **Hidden State**: $h_t = o_t \odot \tanh(C_t)$ - estado oculto actual

**Ventajas**:
- Captura dependencias largas efectivamente
- Mitigación de vanishing gradient mediante arquitectura de compuertas
- Estado celular proporciona memoria persistente

### 3.3 Gated Recurrent Unit (GRU)

Simplificación del LSTM con rendimiento comparable y menos parámetros:

- **Reset Gate**: Controla cuánto del estado previo considerar
- **Update Gate**: Balancea información nueva vs retenida
- Menos parámetros que LSTM, entrenamiento más rápido

### 3.4 Arquitecturas Bidireccionales y Seq2Seq

**Bidirectional RNN/LSTM**:
- Procesa secuencia forward y backward simultáneamente
- Acceso a contexto pasado y futuro
- Efectivo para tareas donde contexto completo está disponible

**Sequence-to-Sequence (Seq2Seq)**:
- **Encoder**: Comprime secuencia de entrada en representación fija
- **Decoder**: Genera secuencia de salida autore gresivamente
- Aplicaciones: traducción automática, resumen, diálogo

**Mecanismo de Atención**:
- Permite decoder atender selectivamente a partes relevantes de entrada
- Mitiga cuello de botella de vector de contexto fijo
- Fundamento para arquitecturas Transformer

### 3.5 Aplicaciones de RNNs

- **Modelado de Lenguaje**: Predicción de siguiente palabra
- **Traducción Automática**: Seq2Seq con atención
- **Reconocimiento de Voz**: Secuencias acústicas a texto
- **Generación de Texto**: Producción de narrativas coherentes
- **Series Temporales**: Predicción financiera, meteorológica

## 4. Autoencoders: Aprendizaje de Representaciones Comprimidas

### 4.1 Arquitectura y Principio de Funcionamiento

Autoencoders aprenden representaciones compactas mediante reconstrucción:

$$h = f_{\theta}(x) \quad \text{(encoder)}$$
$$\hat{x} = g_{\phi}(h) \quad \text{(decoder)}$$
$$\mathcal{L} = \|x - \hat{x}\|^2 \quad \text{(reconstrucción)}$$

**Objetivo**: Minimizar error de reconstrucción fuerza encoder a capturar características más salientes.

### 4.2 Variantes de Autoencoders

**Sparse Autoencoders**:
- Regularización de sparsity en representación latente
- Fuerza activación de pocas neuronas
- Representaciones más interpretables

**Denoising Autoencoders**:
- Entrenan con entrada corrupta pero reconstruyen versión limpia
- Robustez ante ruido
- Aprendizaje de representaciones más robustas

**Variational Autoencoders (VAE)**:
- Encoder produce distribución probabilística $q_\phi(z|x)$
- Loss incluyereconstrucción + divergencia KL: $\mathcal{L} = \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$
- Permite generación mediante muestreo del espacio latente
- Interpolación suave en espacio latente

### 4.3 Aplicaciones

- Reducción de dimensionalidad no lineal
- Compresión de datos (imágenes, audio)
- Pre-entrenamiento de redes profundas
- Detección de anomalías (errores de reconstrucción altos)
- Eliminación de ruido (denoising autoencoders)

## 5. Redes Generativas Adversariales (GANs)

### 5.1 Fundamento Teórico: Juego de Minimax

GANs entrenan dos redes en competencia:

**Generator**: $G(z; \theta_g)$ genera muestras sintéticas desde ruido $z \sim p_z$

**Discriminator**: $D(x; \theta_d)$ discrimina entre muestras reales y sintéticas

**Objetivo Minimax**:
$$\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

Generator busca engañar discriminator; discriminator busca detectar sintéticos.

### 5.2 Arquitecturas y Variantes Modernas

**DCGAN (Deep Convolutional GAN)**:
- Arquitectura totalmente convolucional
- Batch normalization en generator y discriminator
- ReLU en generator, LeakyReLU en discriminator
- Estabilización del entrenamiento

**Conditional GAN (cGAN)**:
- Condicionamiento en etiquetas de clase
- Generación controlada de categorías específicas

**CycleGAN**:
- Traducción imagen-a-imagen sin pares de entrenamiento
- Cycle consistency loss: $x \rightarrow y \rightarrow x \approx x$
- Aplicaciones: style transfer, estaciones del año, fotografía artística

**StyleGAN**:
- Control fino sobre atributos generados
- Mapping network para espacio latente disentangled
- Progressive growing para imágenes de alta resolución

**Diffusion Models**:
- Proceso iterativo de denoising
- Estado del arte en generación de imágenes (DALL-E 2, Stable Diffusion)
- Mayor estabilidad de entrenamiento que GANs tradicionales

### 5.3 Desafíos del Entrenamiento de GANs

- **Mode Collapse**: Generator colapsa a producir variedad limitada
- **Training Instability**: Oscilaciones, no-convergencia
- **Evaluación**: Métricas como Inception Score, FID para calidad

## 6. Técnicas Avanzadas de Entrenamiento

### 6.1 Transfer Learning

Utilización de conocimiento preentrenado en nuevas tareas:

**Estrategias**:
- **Feature Extraction**: Congelar capas preentrenadas, entrenar solo clasificador final
- **Fine-Tuning**: Ajustar finamente todas o últimas capas con learning rate bajo
- **Domain Adaptation**: Adaptar modelo a dominio objetivo diferente

**Modelos Preentrenados**:
- ImageNet pretrained models: ResNet, EfficientNet, Vision Transformers
- Lenguaje: BERT, GPT, T5

### 6.2 Data Augmentation

Expansión artificial del dataset mediante transformaciones:

**Técnicas Estándar**:
- Transformaciones geométricas: rotación, traslación, flip, zoom
- Ajustes de color: brillo, contraste, saturación
- Cropping aleatorio

**Técnicas Avanzadas**:
- **Mixup**: $\tilde{x} = \lambda x_i + (1-\lambda)x_j$, $\tilde{y} = \lambda y_i + (1-\lambda)y_j$
- **CutMix**: Reemplazo de regiones entre imágenes
- **AutoAugment**: Búsqueda automática de políticas de augmentation

### 6.3 Regularización

**Batch Normalization**:
- Normaliza activaciones por mini-batch
- Estabiliza entrenamiento, permite learning rates mayores
- Reduce dependencia de inicialización

**Dropout**:
- Desactivación aleatoria de neuronas durante entrenamiento
- Previene co-adaptación de características
- Ensemble implícito

**Layer Normalization, Group Normalization**:
- Alternativas para tamaños de batch pequeños

## Referencias

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
- Goodfellow, I., Pouget-Abadie, J., Mirza, M., et al. (2014). Generative adversarial nets. *NeurIPS*.
- Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. *arXiv preprint arXiv:1312.6114*.