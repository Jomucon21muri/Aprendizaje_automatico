# Contenido Teórico: Reinforcement Learning
## Sistemas de Aprendizaje Automático - Bloque 7

## 1. Fundamentos de RL

### 1.1 El Problema de RL
- Aprendizaje por interacción con ambiente
- Objetivo: Maximizar recompensa acumulada
- Exploración vs Explotación trade-off
- Retroalimentación retrasada

### 1.2 Procesos de Markov
- Propiedad de Markov: futuro depende solo del presente
- Estados tienen información suficiente
- Transiciones probabilísticas

### 1.3 Markov Decision Process (MDP)
- Tupla: (S, A, P, R, γ)
- S: Espacio de estados
- A: Espacio de acciones
- P(s'|s,a): Dinámica de transición
- R(s,a): Función de recompensa
- γ: Factor de descuento [0,1)

## 2. Conceptos Clave

### 2.1 Política (Policy)
- Mapeo Estado → Acción
- Determinística: π(s) = a
- Estocástica: π(a|s) = P(acción|estado)
- Objetivo: Encontrar política óptima

### 2.2 Funciones de Valor
- **Value Function V(s)**: Valor esperado desde estado s
- **Action-Value Q(s,a)**: Valor de acción a en estado s
- **Advantage A(s,a) = Q(s,a) - V(s)**

### 2.3 Ecuación de Bellman
```
V(s) = E[R(s,a) + γV(s') | π]
Q(s,a) = E[R(s,a) + γmax Q(s',a')]
```
Descomposición: valor = recompensa inmediata + valor futuro

### 2.4 Exploración-Explotación
- ε-greedy: Aleatorio con probabilidad ε
- Boltzmann: Probabilístico con temperatura
- UCB (Upper Confidence Bound)
- Thompson Sampling

## 3. Métodos Value-Based

### 3.1 Q-Learning
- Off-policy: Aprende política óptima mientras explora
- Bellman update: Q(s,a) ← Q(s,a) + α[r + γmax Q(s',a') - Q(s,a)]
- Convergencia garantizada con tabla finita
- Impracticable para espacios grandes

### 3.2 Deep Q-Networks (DQN)
- Aproxima Q-values con red neuronal
- Experience Replay: Almacena y muestrea transiciones
- Target Network: Red separada para estabilidad
- Double DQN, Dueling DQN: Mejoras

### 3.3 SARSA
- On-policy: Aprende política actual
- Más conservador que Q-Learning
- Útil para control en línea

## 4. Métodos Policy-Based

### 4.1 Policy Gradient
- Optimiza política directamente
- ∇J(θ) ∝ E[∇log π(a|s)Q(s,a)]
- Ventaja: Políticas estocásticas, espacios continuos

### 4.2 REINFORCE (Monte Carlo Policy Gradient)
- Usa retornos episódicos completos
- Alto varianza, bajo sesgo
- Simple de implementar

### 4.3 Actor-Critic
- Actor: Política (policy)
- Critic: Función de valor (baseline)
- Reduce varianza vs REINFORCE
- Ejemplo: A3C (Asynchronous Advantage Actor-Critic)

### 4.4 Métodos Avanzados
- **PPO (Proximal Policy Optimization)**: Clipping de ratio
- **TRPO (Trust Region)**: Restricción de región segura
- **A3C**: Actor-Critic asincrónico paralelo
- **SAC (Soft Actor-Critic)**: Máxima entropía

## 5. Model-Based RL

### 5.1 Aprendizaje de Dinámicas
- Aprende modelo: P(s'|s,a) y R(s,a)
- Planning: Usa modelo para decisiones
- Data-efficient pero requiere más cómputo

### 5.2 Métodos de Planning
- Dyna: Combina learning y planning
- Tree search: Monte Carlo Tree Search (MCTS)
- AlphaGo: Combinación redes neuronales + MCTS

## 6. Aplicaciones Prácticas

### 6.1 Juegos
- Atari: DQN, Rainbow
- Go: AlphaGo (policy + value networks)
- Ajedrez: AlphaZero (self-play, MCTS)

### 6.2 Robótica
- Manipulación: Aprehensión de objetos
- Navegación: Path planning
- Control motor: Movimientos fluidos

### 6.3 Sistemas Reales
- Conducción autónoma
- Optimización de energía
- Trading algorítmico
- Recomendaciones personalizadas

## 7. Desafíos

### 7.1 Muestra Ineficiencia
- Requiere millones de interacciones
- Impracticable para sistemas reales
- Solución: Transfer learning, simuladores

### 7.2 Reward Shaping
- Diseño de función de recompensa es crítico
- Pobre diseño → Comportamiento no deseado
- Aprendizaje inverso: Inferir recompensas

### 7.3 No-estacionariedad
- Ambiente puede cambiar
- Políticas viejas pueden fallar
- Adaptación continua requerida