# Mapa Conceptual: Transformers
## Sistemas de Aprendizaje Automático - Bloque 5

### Arquitectura Transformer

```
Entrada de Texto
      ↓
Embedding + Positional Encoding
      ↓
┌─────────────────────────┐
│   Multi-Head Attention  │ (Encoder)
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│  Feed-Forward Network   │
└─────────────────────────┘
      ↓
Stack de N Bloques Encoder
      ↓
┌─────────────────────────┐
│   Multi-Head Attention  │ (Decoder)
└─────────────────────────┘
      ↓
Predicción de Siguiente Token
```

### Componentes Clave
- **Self-Attention**: Relaciones entre tokens
- **Multi-Head Attention**: Múltiples representaciones
- **Positional Encoding**: Información de posición
- **Feed-Forward**: Transformación no-lineal
- **Layer Normalization**: Normalización entre capas
- **Residual Connections**: Facilita gradientes profundos

### Variantes de Transformers
- **BERT**: Bidireccional (encoding)
- **GPT**: Autoregresivo (decoding)
- **T5**: Encoder-Decoder
- **Vision Transformer (ViT)**: Para imágenes
- **DistilBERT**: Comprimido
- **RoBERTa, ALBERT, ELECTRA**: Mejoras BERT

### Casos de Uso
- Traducción automática
- Análisis de sentimiento
- Respuesta a preguntas
- Generación de texto
- Clasificación de documentos