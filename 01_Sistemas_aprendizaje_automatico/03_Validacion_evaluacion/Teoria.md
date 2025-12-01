# Contenido teórico: validación y evaluación de modelos
## Sistemas de aprendizaje automático - Bloque 3

## 1. Problemas de generalización

### 1.1 Overfitting (sobreajuste)
- Modelo aprende ruido y particularidades
- Rendimiento excelente en entrenamiento, pobre en prueba
- Causas: modelo complejo, pocos datos, bajo regularización

### 1.2 Underfitting (infraajuste)
- Modelo demasiado simple
- Rendimiento pobre en ambos conjuntos
- Causas: modelo simple, datos insuficientes, features pobres

### 1.3 Equilibrio sesgo-varianza
- Sesgo: error sistemático del modelo
- Varianza: Sensibilidad a fluctuaciones en datos
- Objetivo: Minimizar ambos

## 2. Estrategias de validación

### 2.1 Validación simple (Train/Test)
- 70-80% entrenamiento, 20-30% prueba
- Rápida pero con varianza alta

### 2.2 Validación cruzada (K-Fold)
- Divide datos en k folds
- Entrena k modelos, cada uno con un fold como prueba
- Más robusta que validación simple

### 2.3 K-Fold estratificada
- Para datos desbalanceados
- Mantiene proporción de clases en cada fold

### 2.4 Time Series Split
- Para datos secuenciales
- No mezcla datos futuros con pasados

## 3. Métricas de clasificación

### 3.1 Matriz de confusión
```
           Predicho Positivo  Predicho Negativo
Real Pos   TP                FN
Real Neg   FP                TN
```

### 3.2 Métricas derivadas
- **Exactitud** = (TP + TN) / Total
- **Precisión** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1-Score** = 2 * (Precisión * Recall) / (Precisión + Recall)
- **Especificidad** = TN / (TN + FP)

### 3.3 Curva ROC y AUC
- ROC: Sensibilidad vs (1 - Especificidad)
- AUC: Área bajo la curva (0.5 a 1.0)

## 4. Métricas de regresión

### 4.1 Error absoluto medio (MAE)
MAE = (1/n) * ∑|Yi - Ŷi|

### 4.2 Error cuadrático medio (MSE)
MSE = (1/n) * ∑(Yi - Ŷi)²

### 4.3 RMSE
RMSE = √MSE

### 4.4 Coeficiente de determinación (R²)
R² = 1 - (SS_residual / SS_total)

## 5. Hiperparámetros y ajuste

### 5.1 Grid Search
- Busca exhaustiva sobre grid de parámetros
- Computacionalmente intensivo

### 5.2 Random Search
- Muestreo aleatorio de parámetros
- Más eficiente en espacios grandes

### 5.3 Validación de hiperparámetros
- Usa validation curve
- Evita overfitting del modelo al conjunto de prueba
