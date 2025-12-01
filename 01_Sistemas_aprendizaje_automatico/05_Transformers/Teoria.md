# Contenido teórico: transformers
## Sistemas de aprendizaje automático - Bloque 5

## 1. Limitaciones de RNN/LSTM

### 1.1 Problemas
- Procesamiento secuencial (no paralelizable)
- Dependencia a corto plazo pese a LSTM
- Cuello de botella para textos largos
- Difícil capturar dependencias globales

### 1.2 Solución: Attention Mechanism
- Permite relaciones directas entre todos los tokens
- Completamente paralelizable
- Escala a secuencias muy largas

## 2. Mecanismo de attention

### 2.1 Self-Attention escalado
```
Attention(Q, K, V) = softmax(QK^T / √dk) V
```
Donde:
- Q (Query): Qué buscar
- K (Key): Qué está disponible
- V (Value): Qué retornar
- dk: Dimensión para escala

### 2.2 Multi-Head Attention
- Múltiples mecanismos de atención en paralelo
- Cada head captura diferentes tipos de relaciones
- Concatenación y proyección final

### 2.3 Interpretabilidad
- Attention weights muestran dependencias
- Visualizable: cuál token atiende cuál
- Explainability mejorado

## 3. Arquitectura Transformer completa

### 3.1 Encoder Stack
- Múltiples bloques idénticos (típicamente 12-24)
- Cada bloque: Multi-Head Attention + Feed-Forward
- Layer Normalization y conexiones residuales
- Output: Representaciones contextuales

### 3.2 Decoder Stack
- Similar a encoder pero con atención modificada
- Masked Multi-Head Attention: Solo tokens previos
- Cross-Attention: Atiende a encoder output
- Autoregresivo: Genera token por token

### 3.3 Embeddings y posiciones
- Token Embeddings: representación densa
- Positional Encoding: información de posición (sin entrenable)
- Suma: Token + Position encoding

## 4. Modelos basados en transformers

### 4.1 BERT (Bidirectional Encoder Representations)
- Solo encoder
- Preentrenamiento bidireccional (MLM, NSP)
- Excelente para clasificación y análisis
- Variantes: RoBERTa, ALBERT, DistilBERT

### 4.2 GPT (Generative Pre-trained Transformer)
- Solo decoder
- Preentrenamiento autoregresivo
- Excelente para generación de texto
- Versiones: GPT-2, GPT-3, GPT-3.5, GPT-4

### 4.3 T5 (Text-to-Text Transfer Transformer)
- Encoder-Decoder completo
- Trata todo como generación texto-a-texto
- Versátil para múltiples tareas
- Versiones: T5 base, large, xl, xxl

### 4.4 Vision Transformer (ViT)
- Aplica Transformers a imágenes
- Divide imagen en patches
- Comparable con CNNs en ImageNet
- Escalable a imágenes de alta resolución

## 5. Preentrenamiento y fine-tuning

### 5.1 Preentrenamiento
- **MLM (Masked Language Modeling)**: Predecir palabras enmascaradas
- **NSP (Next Sentence Prediction)**: Predicción de siguiente oración
- **Causal Language Modeling**: Predecir siguiente token
- Realizado en corpus masivo (Wikipedia, Common Crawl)

### 5.2 Fine-tuning
- Ajusta pesos preentrenados en tarea específica
- Requiere mucho menos datos
- Transfer Learning efectivo
- Learning rate bajo para no olvidar

### 5.3 Prompting
- Con GPT-3+: Few-shot learning
- Chain-of-Thought prompting
- Role-based prompting
- Prompt engineering crítico

## 6. Eficiencia y escalabilidad

### 6.1 Complejidad computacional
- Self-attention: O(n²) en longitud secuencia
- Linformer, Performer: Approximaciones lineales
- Sparse attention para eficiencia

### 6.2 Compresión de modelos
- Quantization: reducir precisión
- Pruning: Eliminar pesos
- Destilación: Modelo pequeño aprende de grande
- LoRA: Adaptación con pocos parámetros

### 6.3 Instrumentos
- DeepSpeed, Megatron: Entrenamiento distribuido
- Hugging Face Transformers: Librería estándar
- ONNX, TensorFlow Lite: Inferencia optimizada