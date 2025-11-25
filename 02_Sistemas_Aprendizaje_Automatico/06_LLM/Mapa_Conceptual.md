# Mapa Conceptual: Large Language Models (LLM)
## Sistemas de Aprendizaje Automático - Bloque 6

### Evolución de LLMs

```
2018: BERT (110M params)
      ↓
2019: GPT-2 (1.5B params)
      ↓
2020: GPT-3 (175B params) - Few-shot learning
      ↓
2022: ChatGPT (RLHF) - Breakthrough público
      ↓
2023+: GPT-4, Claude, LLaMA, Mistral
```

### Arquitectura y Escala

```
LLM = Transformer Decoder × 10^9+ parámetros

Componentes:
├── Vocabulary: 50K-100K tokens
├── Embedding: ~512-1024 dimensiones
├── Transformer Blocks: 12-96 capas
├── Attention Heads: 8-128 cabezas
└── Total Params: 1B-405B+
```

### Métodos de Adaptación
- **Fine-tuning**: Reentrenamiento de todos los pesos
- **LoRA**: Adaptación de bajo rango
- **Prompt Engineering**: Instrucciones específicas
- **Few-Shot Learning**: Ejemplos en contexto
- **Chain-of-Thought**: Razonamiento paso-a-paso
- **Retrieval-Augmented Generation (RAG)**: Con información externa

### Aplicaciones
- Chatbots conversacionales
- Resumen automático
- Traducción
- Programación automática
- Análisis de texto
- Generación creativa

### Consideraciones
- Hallucinations: Información inventada
- Sesgo: Datos de entrenamiento
- Costo computacional
- Privacidad y seguridad
- Ética y gobernanza