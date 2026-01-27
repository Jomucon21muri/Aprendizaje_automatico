# Arquitecturas Transformer: Mecanismos de Atención y Modelos de Secuencia
## Sistemas de Aprendizaje Automático - Bloque 5

## Resumen

La arquitectura Transformer, introducida por Vaswani et al. (2017) en el trabajo seminal "Attention Is All You Need", ha revolucionado el procesamiento de secuencias y constituye la base de los modelos de lenguaje modernos más avanzados. Este documento examina exhaustivamente los fundamentos teóricos de los mecanismos de atención, la arquitectura Transformer completa, y los modelos derivados que han establecido nuevos estándares en procesamiento del lenguaje natural, visión computacional y tareas multimodales. Se proporciona un análisis riguroso de componentes arquitectónicos, estrategias de preentrenamiento y aplicaciones contemporáneas.

## 1. Motivación y Limitaciones de Arquitecturas Recurrentes

### 1.1 Problemáticas de RNN y LSTM

A pesar de los avances con redes recurrentes, persisten limitaciones fundamentales:

**Procesamiento Secuencial Inherente**:
- Computación debe proceder paso-a-paso temporalmente
- Imposibilidad de paralelización dentro de secuencias
- Cuello de botella computacional severo para secuencias largas
- Escalabilidad limitada con hardware moderno (GPUs, TPUs)

**Dependencias de Largo Alcance**:
- LSTM/GRU mejoran pero no eliminan completamente degradación gradiente
- Información debe propagarse a través de muchos pasos temporales
- Ruta de información crece linealmente con distancia en secuencia
- Dificultad para capturar dependencias globales efectivamente

**Cuello de Botella del Encoder**:
- En arquitecturas Seq2Seq, todo el contexto se comprime en vector fijo
- Pérdida de información para secuencias largas
- Atención básica mitiga pero no elimina problema fundamental

### 1.2 Solución Paradigmática: Self-Attention

El mecanismo de self-attention permite:
- **Paralelización Completa**: Todas las posiciones procesadas simultáneamente
- **Rutas de Información Constantes**: Cualquier par de posiciones conectado directamente (ruta O(1))
- **Dependencias Globales**: Cada token puede atender a todos los demás directamente
- **Escalabilidad**: Aprovechamiento óptimo de arquitecturas paralelas modernas

## 2. Mecanismo de Atención: Fundamentos Matemáticos

### 2.1 Atención Escal ada Producto-Punto (Scaled Dot-Product Attention)

El mecanismo fundamental de atención se define matemáticamente como:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Componentes**:
- **Query (Q)**: ¿Qué estoy buscando? Representación de posición que realiza consulta
- **Key (K)**: ¿Qué información tengo disponible? Representaciones para matchear con queries
- **Value (V)**: ¿Qué información devolver? Representaciones a agregar según scores de atención

**Proceso Computacional**:
1. Calcular scores de compatibilidad: $S = QK^T$ (producto matricial)
2. Escalar: $S' = S / \sqrt{d_k}$ (evita magnitudes excesivas para softmax)
3. Normalizar: $A = \text{softmax}(S')$ (distribuciónprobabilística sobre posiciones)
4. Agregar valores: $O = AV$ (combinación ponderada de valores)

**Escalamiento por $\sqrt{d_k}$**:
- Sin escalamiento, productos punto en dimensiones altas tienen magnitud $O(d_k)$
- Magnitudes grandes empujan softmax a regiones de gradientes vanishing
- Escalamiento mantiene varianza unitaria, mejorando estabilidad y aprendizaje

### 2.2 Multi-Head Attention: Múltiples Subespacios Representacionales

Multi-head attention permite al modelo atender a diferentes aspectos simultáneamente:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

donde cada head:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Dimensiones**:
- $W_i^Q, W_i^K \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$
- $W^O \in \mathbb{R}^{hd_v \times d_{model}}$

**Ventajas**:
- Cada head aprende diferentes tipos de relaciones
- Algunos heads capturan sintaxis, otros semántica, otros referencias anafóricas
- Diversidad representacional incrementa capacidad expresiva
- Paralelización eficiente de múltiples operaciones de atención

### 2.3 Interpretabilidad de Attention Weights

Los pesos de atención proporcionan insights sobre decisiones del modelo:
- Visualización de qué tokens atienden a cuáles
- Identificación de dependencias sintácticas capturadas
- Comprensión de flujo de información
- Herramienta diagnóstica para análisis de errores

**Limitaciones**:
- Attention weights no son necesariamente explicaciones causales
- Múltiples heads pueden distribuir información
- Interpretación puede ser no intuitiva en capas profundas

## 3. Arquitectura Transformer Completa

### 3.1 Positional Encoding: Inyección de Información Posicional

Transformers carecen de noción inherente de orden secuencial. Positional encodings añaden información posicional:

**Encoding Sinusoidal (Original)**:
$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

**Propiedades**:
- Determinístico, no aprendido
- Permite extrapolación a secuencias más largas que entrenamiento
- Distancias relativas representables mediante transformaciones lineales
- Diferentes frecuencias para diferentes dimensiones

**Alternativas Aprendidas**:
- Embeddings posicionales aprendidos
- Positional encodings relativos (mejor para secuencias largas)
- RoPE (Rotary Position Embeddings) en modelos modernos

### 3.2 Encoder Stack: Procesamiento Bidireccional

**Arquitectura de una Capa de Encoder**:
1. **Multi-Head Self-Attention**: Atención sobre toda la secuencia de entrada
2. **Add & Norm**: Conexión residual + Layer Normalization
3. **Feed-Forward Network**: $FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$ (dos capas lineales con ReLU)
4. **Add & Norm**: Segunda conexión residual + Layer Normalization

**Layer Normalization**:
$$\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sigma + \epsilon} + \beta$$

Normaliza a través de dimensión de características (no batch), estabilizando entrenamiento.

**Conexiones Residuales**:
$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

Facilitan flujo de gradientes, permiten entrenamiento de arquitecturas muy profundas.

### 3.3 Decoder Stack: Generación Autoregresiva

**Arquitectura de una Capa de Decoder**:
1. **Masked Multi-Head Self-Attention**: Atención solo a posiciones previas (autoregresivo)
2. **Add & Norm**
3. **Multi-Head Cross-Attention**: Atención a salida del encoder
4. **Add & Norm**
5. **Feed-Forward Network**
6. **Add & Norm**

**Masked Attention**:
- Enmascara posiciones futuras en secuencia objetivo
- Garantiza que predicción en posición $i$ solo depende de posiciones $<i$
- Preserva propiedad autoregresiva necesaria para generación

**Cross-Attention**:
- Queries del decoder
- Keys y Values del encoder
- Permite decoder atender selectivamente a entrada relevante

### 3.4 Complejidad Computacional

**Self-Attention**: $O(n^2 \cdot d)$ donde $n$ es longitud de secuencia
- Cuadrática en longitud de secuencia
- Cuello de botella para secuencias muy largas
- Motiva variantes eficientes (Linformer, Performer, Longformer)

**Feed-Forward**: $O(n \cdot d^2)$ lineal en longitud de secuencia

## 4. Modelos Basados en Transformer

### 4.1 BERT: Bidirectional Encoder Representations from Transformers

**Arquitectura**:
- Solo stack de encoders (12 para BERT-base, 24 para BERT-large)
- Procesamiento completamente bidireccional
- 110M parámetros (base), 340M (large)

**Objetivos de Preentrenamiento**:

**Masked Language Modeling (MLM)**:
- Enmascara 15% de tokens aleatoriamente
- Modelo predice tokens enmascarados basándose en contexto bidireccional
- Fuerza comprensión profunda de relaciones bidireccionales

**Next Sentence Prediction (NSP)**:
- Determina si oración B sigue genuinamente a oración A
- Captura relaciones entre oraciones
- Menos crítico (eliminado en RoBERTa)

**Fine-Tuning**:
- Añadir capa de clasificación específica a tarea
- Ajustar todos los parámetros con learning rate bajo
- Transfer learning extremadamente efectivo

**Variantes**:
- **RoBERTa**: Entrenamiento más largo, batch sizes mayores, sin NSP
- **ALBERT**: Factorización de embeddings, compartición de parámetros entre capas
- **DistilBERT**: Destilación de conocimiento, 40% menos parámetros, 97% rendimiento

### 4.2 GPT: Generative Pre-trained Transformer

**Arquitectura**:
- Solo stack de decoders (sin cross-attention)
- Procesamiento unidireccional (left-to-right)
- Diseñado específicamente para generación autoregresiva

**Preentrenamiento**:
- **Causal Language Modeling**: Predecir siguiente token dado contexto previo
- $\max_\theta \sum_i \log P(x_i | x_{<i}; \theta)$
- Entrenamiento en corpus masivo (WebText, Common Crawl)

**Evolución**:
- **GPT (2018)**: 117M parámetros, demostró viabilidad de preentrenamiento
- **GPT-2 (2019)**: 1.5B parámetros, generación de texto de calidad sorprendente
- **GPT-3 (2020)**: 175B parámetros, emergencia de few-shot learning
- **GPT-4 (2023)**: Multimodal, capacidades de razonamiento avanzadas

**In-Context Learning**:
- Capacidad de adaptarse a tareas nuevas mediante ejemplos en prompt
- Sin actualización de parámetros
- Emergencia con escala suficiente

### 4.3 T5: Text-to-Text Transfer Transformer

**Paradigma Unificado**:
- Toda tarea formulada como texto-a-texto
- Encoder-Decoder completo
- Versatilidad máxima

**Ejemplos de Formato**:
- Traducción: "translate English to German: That is good." → "Das ist gut."
- Resumen: "summarize: [article]" → "[summary]"
- Clasificación: "sentiment: This movie is great!" → "positive"

**Span Corruption Objective**:
- Corrompe spans contiguos de texto
- Modelo debe predecir spans enmascarados
- Generaliza MLM a secuencias

### 4.4 Vision Transformer (ViT)

**Adaptación a Imágenes**:
1. Dividir imagen en patches (e.g., 16×16)
2. Lineal projection de patches a embeddings
3. Añadir positional embeddings
4. Procesar con Transformer encoder estándar
5. Clasificación mediante token [CLS]

**Resultados**:
- Rendimiento comparable o superior a CNNs en ImageNet
- Requiere más datos de preentrenamiento que CNNs
- Escalabilidad excepcional
- Foundation para modelos multimodales (CLIP, DALL-E)

## 5. Estrategias de Preentrenamiento y Fine-Tuning

### 5.1 Paradigma de Transfer Learning

**Fase 1: Preentrenamiento**:
- Corpus masivo general (Books, Wikipedia, Common Crawl)
- Objetivos self-supervised (MLM, CLM, etc.)
- Costoso computacionalmente (semanas, miles de GPUs)
- Aprende representaciones lingüísticas generales

**Fase 2: Fine-Tuning**:
- Dataset específico a tarea target
- Ajuste de todos los parámetros o solo últimas capas
- Learning rate típicamente 10-100× menor que preentrenamiento
- Horas o días de entrenamiento
- Dramática mejora en data efficiency

### 5.2 Prompting y Few-Shot Learning

**Zero-Shot**:
- Sin ejemplos, solo descripción de tarea
- "Translate to French: Hello" → modelo infiere tarea

**Few-Shot**:
- Pocos ejemplos (1-50) en contexto
- Modelo adapta sin actualizar parámetros
- Efectivo en modelos de escala suficiente (>10B parámetros)

**Chain-of-Thought Prompting**:
- "Let's think step-by-step"
- Mejora dramática en razonamiento complejo
- Elicita razonamiento intermedio

### 5.3 Instruction Tuning

Fine-tuning en dataset masivo de (instrucción, salida) pares:
- Mejora seguimiento de instrucciones
- Generalización a instrucciones no vistas
- Ejemplos: FLAN, InstructGPT, Alpaca

## 6. Consideraciones de Eficiencia y Escalabilidad

### 6.1 Transformers Eficientes

**Linformer**: Aproximación de atención en $O(n)$ mediante proyecciones de bajo rango

**Performer**: Kernel methods para aproximar softmax attention

**Longformer**: Atención sparse con patrones locales y globales

**Big Bird**: Combinación de atención local, global y random

### 6.2 Cuantización y Compresión

- Cuantización de parámetros (FP32 → INT8)
- Poda de pesos menos importantes
- Destilación de conocimiento
- LoRA (Low-Rank Adaptation) para fine-tuning eficiente

## Referencias

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *NeurIPS*.
- Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers. *NAACL*.
- Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding with unsupervised learning. *Technical report, OpenAI*.
- Raffel, C., Shazeer, N., Roberts, A., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *JMLR*, 21, 1-67.
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An image is worth 16x16 words. *ICLR*.
