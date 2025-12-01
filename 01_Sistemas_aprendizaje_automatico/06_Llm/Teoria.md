# Contenido Teórico: Large Language Models (LLM)
## Sistemas de Aprendizaje Automático - Bloque 6

## 1. Fundamentos de LLMs

### 1.1 Definición y Características
- Modelos generativos basados en Transformers
- Entrenados en corpus de texto masivo
- Capacidad de predicción de siguiente token
- Emergencia de habilidades con escala

### 1.2 Escala y Emergencia
- Ley de Escala: Performance ∝ log(Parámetros)
- Habilidades emergen en ciertos umbrales
- Few-shot learning con tamaño suficiente
- In-context learning

## 2. Preentrenamiento a Escala Masiva

### 2.1 Datos de Entrenamiento
- Corpus: Wikipedia, Common Crawl, código, libros
- Volumen: Terabytes de texto
- Filtrado y limpieza
- Duplicación limitada

### 2.2 Objetivo de Preentrenamiento
- Causal Language Modeling (Next Token Prediction)
- Minimizar cross-entropy loss
- Entrenamiento distribuido en miles de GPUs
- Duración: Semanas a meses

### 2.3 Métricas de Evaluación
- Perplexity: Incertidumbre promedio por token
- Benchmarks: GLUE, SuperGLUE, MMLU, BigBench
- Task-specific metrics: BLEU, ROUGE, F1
- Human evaluation

## 3. Técnicas de Alineación Humana

### 3.1 Alineación (Alignment)
- Problema: Modelos preentrenados pueden ser peligrosos
- Objetivo: Alinear con valores humanos
- Trade-off: Helpfulness vs Harmlessness

### 3.2 RLHF (Reinforcement Learning from Human Feedback)
1. Collect human preferences
2. Train reward model
3. Fine-tune con PPO (Proximal Policy Optimization)
4. Iteración con nuevas evaluaciones

### 3.3 Instruction Tuning
- Fine-tuning en tareas con instrucciones
- Dataset: (instrucción, entrada, salida esperada)
- Ejemplo: FLAN, T0
- Mejora generalización a tareas nuevas

## 4. Familias Principales de LLMs

### 4.1 GPT Series (OpenAI)
- **GPT-2** (2019): 1.5B params, open source
- **GPT-3** (2020): 175B params, few-shot learning
- **GPT-3.5** (2022): ChatGPT, alineado
- **GPT-4** (2023): Multimodal, razonamiento mejorado

### 4.2 PaLM/Gemini (Google)
- **PaLM** (2022): 540B params
- **Gemini**: Multimodal, visión + texto
- **LaMDA**: Diálogo específico

### 4.3 Claude (Anthropic)
- Constitutional AI: Alineación con principios
- Versiones: Claude 1, 2, 3 (Opus, Sonnet, Haiku)
- Énfasis en seguridad y honestidad

### 4.4 LLaMA (Meta)
- Open source: 7B, 13B, 70B
- Entrenamiento eficiente
- Base para: Alpaca, Mistral, Nous

### 4.5 Mistral (Europa)
- Alternativa abierta y eficiente
- Versiones: 7B, 8x7B (Mixture of Experts)
- Enfoque comercial open source

## 5. Métodos de Aprovechamiento (Prompting)

### 5.1 Prompting Básico
- Zero-shot: Sin ejemplos
- Few-shot: Con ejemplos en contexto
- Instruction: Instrucciones explícitas

### 5.2 Chain-of-Thought (CoT)
- "Piensa paso-a-paso"
- Mejora razonamiento complejo
- Self-consistency: Múltiples caminos

### 5.3 Retrieval-Augmented Generation (RAG)
- Recupera documentos relevantes
- Aumenta contexto con información externa
- Reduce hallucinations
- Permite información actualizada

### 5.4 Agents y Tools
- LLM como razonador central
- Acceso a herramientas (búsqueda, calculadora, APIs)
- ReAct: Reasoning + Acting
- Ejemplo: AutoGPT, LangChain agents

## 6. Adaptación y Especialización

### 6.1 Fine-tuning Completo
- Reentrenamiento de todos los pesos
- Requiere GPU de alto rendimiento
- Riesgo: Catastrophic forgetting

### 6.2 LoRA (Low-Rank Adaptation)
- Adapta solo parámetros adicionales (~1% del modelo)
- Entrenable en una sola GPU
- Mantiene conocimiento base

### 6.3 Prompt Tuning
- Optimiza prefijo de tokens aprendidos
- No requiere reentrenamiento
- Eficiente pero menos potente

## 7. Desafíos y Consideraciones

### 7.1 Hallucinations
- Generación de información falsa
- Parece confiable pero es inventado
- Mitigación: Verificación externa, RAG

### 7.2 Sesgo y Fairness
- Heredado de datos de entrenamiento
- Puede amplificar estereotipos
- Evaluación: Benchmark de fairness

### 7.3 Costo Computacional
- Entrenamiento: Millones de dólares, semanas
- Inferencia: Caro en escala
- Optimizaciones: Quantization, distillation

### 7.4 Privacidad y Seguridad
- Memorización de datos de entrenamiento
- Ataques: Adversarial prompts
- Regulación: GDPR, IA Act (EU)

### 7.5 Ética
- Uso potencial para desinformación
- Impacto laboral
- Responsabilidad por errores
- Gobernanza de IA