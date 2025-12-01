# Mapa conceptual: transformers
## Sistemas de aprendizaje automático - Bloque 5

### Arquitectura Transformer

```
Entrada de texto
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
Predicción de siguiente token
```

### Componentes clave
- **Self-Attention**: relaciones entre tokens
- **Multi-Head Attention**: múltiples representaciones
- **Positional Encoding**: información de posición
- **Feed-Forward**: transformación no-lineal
- **Layer Normalization**: normalización entre capas
- **Residual Connections**: facilita gradientes profundos

### Variantes de transformers
- **BERT**: Bidireccional (encoding)
- **GPT**: Autoregresivo (decoding)
- **T5**: Encoder-Decoder
- **Vision Transformer (ViT)**: Para imágenes
- **DistilBERT**: Comprimido
- **RoBERTa, ALBERT, ELECTRA**: Mejoras BERT

### Casos de uso
- Traducción automática
- Análisis de sentimiento
- Respuesta a preguntas
- Generación de texto
- Clasificación de documentos