# Aprendizaje por Refuerzo: Fundamentos y Algoritmos
## Sistemas de Aprendizaje Automático - Bloque 7

## Resumen

El Aprendizaje por Refuerzo (Reinforcement Learning) constituye un paradigma fundamental del aprendizaje automático donde un agente aprende a tomar decisiones óptimas mediante interacción con su entorno, recibiendo retroalimentación en forma de recompensas. Este documento examina rigurosamente los fundamentos teóricos basados en Procesos de Decisión de Markov, algoritmos value-based y policy-based, métodos actor-critic, y enfoques model-based. Se analizan aplicaciones desde juegos estratégicos hasta robótica y sistemas de control, proporcionando una comprensión integral de técnicas que han revolucionado campos como la IA para videojuegos y control autónomo.

## 1. Fundamentos Teóricos del Aprendizaje por Refuerzo

### 1.1 Problema Fundamental y Definición

El Aprendizaje por Refuerzo aborda el problema de cómo un agente debe actuar en un entorno para maximizar recompensa acumulada a largo plazo:

**Características Distintivas**:
- **Aprendizaje por Interacción**: No existe dataset etiquetado; el agente aprende mediante prueba y error
- **Retroalimentación Retrasada**: Consecuencias de acciones pueden manifestarse temporal mente distantes
- **Trade-off Exploración-Explotación**: Balance entre explorar nuevas estrategias y explotar conocimiento actual
- **Decisiones Secuenciales**: Acciones presentes afectan estados y oportunidades futuras

**Componentes del Sistema RL**:
- **Agente**: Entidad que toma decisiones y aprende
- **Entorno**: Sistema con el cual el agente interacta
- **Estado ($s$)**: Representación de configuración actual del entorno
- **Acción ($a$)**: Decisión tomada por el agente
- **Recompensa ($r$)**: Señal escalar de feedback del entorno
- **Política ($\pi$)**: Estrategia del agente, mapeo estado→acción

### 1.2 Procesos de Decisión de Markov (MDP)

Formalización matemática del problema de RL:

**Definición Formal**:
Un MDP se define mediante la tupla $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

- $\mathcal{S}$: Espacio de estados
- $\mathcal{A}$: Espacio de acciones
- $\mathcal{P}(s'|s,a)$: Función de transición (dinámica del entorno)
- $\mathcal{R}(s,a,s')$: Función de recompensa
- $\gamma \in [0,1)$: Factor de descuento temporal

**Propiedad de Markov**:
$$P(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, \ldots, s_0, a_0) = P(s_{t+1}|s_t, a_t)$$

El futuro depende únicamente del estado presente, no de la historia completa. Esta propiedad permite tractabilidad computacional.

**Retorno Descontado**:
$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty}\gamma^k r_{t+k}$$

El factor $\gamma$ modela preferencia temporal: recompensas inmediatas valen más que futuras.

### 1.3 Políticas y Funciones de Valor

**Política**:
- **Determinística**: $\pi(s) = a$
- **Estocástica**: $\pi(a|s) = P(\text{acción}=a|\text{estado}=s)$

**Función de Valor de Estado**:
$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k} \mid s_t=s\right]$$

Valor esperado de retorno comenzando en estado $s$ siguiendo política $\pi$.

**Función de Valor de Acción (Q-Function)**:
$$Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k} \mid s_t=s, a_t=a\right]$$

Valor esperado comenzando en $s$, tomando acción $a$, luego siguiendo $\pi$.

**Función de Ventaja**:
$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

Cuánto mejor es tomar acción $a$ comparado con seguir la política.

**Ecuaciones de Bellman**:
$$V^\pi(s) = \sum_a \pi(a|s)\sum_{s',r}P(s',r|s,a)[r + \gamma V^\pi(s')]$$
$$Q^\pi(s,a) = \sum_{s',r}P(s',r|s,a)[r + \gamma \sum_{a'}\pi(a'|s')Q^\pi(s',a')]$$

Descomposición recursiva: valor = recompensa inmediata + valor descontado del futuro.

### 1.4 Políticas Óptimas

**Política Óptima**: $\pi^* = \arg\max_\pi V^\pi(s), \forall s$

**Ecuación de Optimalidad de Bellman**:
$$V^*(s) = \max_a \sum_{s',r}P(s',r|s,a)[r + \gamma V^*(s')]$$
$$Q^*(s,a) = \sum_{s',r}P(s',r|s,a)[r + \gamma \max_{a'}Q^*(s',a')]$$

Dado $Q^*$, política óptima es: $\pi^*(s) = \arg\max_a Q^*(s,a)$

## 2. Trade-off Exploración-Explotación

Dilema fundamental: ¿explorar para descubrir potencialmente mejores acciones, o explotar conocimiento actual?

### 2.1 Estrategias de Exploración

**ε-Greedy**:
$$a = \begin{cases}
\arg\max_a Q(s,a) & \text{con probabilidad } 1-\epsilon \\
\text{acción aleatoria} & \text{con probabilidad } \epsilon
\end{cases}$$

Simple pero efectivo. $\epsilon$ típicamente decae durante entrenamiento.

**Boltzmann Exploration (Softmax)**:
$$P(a|s) = \frac{\exp(Q(s,a)/\tau)}{\sum_{a'}\exp(Q(s,a')/\tau)}$$

$\tau$ (temperatura) controla aleatoriedad. $\tau \to 0$: greedy, $\tau \to \infty$: uniforme.

**Upper Confidence Bound (UCB)**:
$$a_t = \arg\max_a \left[Q(s,a) + c\sqrt{\frac{\ln t}{N(s,a)}}\right]$$

Favorece acciones prometedoras pero poco exploradas.

**Thompson Sampling**:
Muestreo probabilístico desde distribuciones bayesianas sobre valores estimados.

## 3. Algoritmos Value-Based

### 3.1 Q-Learning

Algoritmo off-policy que aprende $Q^*$ directamente:

**Regla de Actualización**:
$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha[r_t + \gamma \max_{a'}Q(s_{t+1},a') - Q(s_t,a_t)]$$

**Propiedades**:
- **Off-policy**: Aprende política óptima mientras sigue política exploratoria
- **Convergencia**: Garantizada bajo condiciones estándar (visitas infinitas, decaimiento de learning rate)
- **Tabular**: Tabla Q para espacios discretos pequeños

**Limitaciones**:
- Impracticable para espacios de estados grandes/continuos
- Requiere discretización o aproximación de funciones

### 3.2 Deep Q-Networks (DQN)

Aproximación de Q-function con redes neuronales profundas:

**Innovaciones Clave**:

**Experience Replay**:
- Almacena transiciones $(s,a,r,s')$ en buffer de memoria
- Muestrea mini-batches aleatorios para entrenamiento
- Rompe correlaciones temporales, estabiliza aprendizaje
- Reutilización eficiente de datos

**Target Network**:
- Red separada $Q_{\theta^-}$ con parámetros $\theta^-$ actualizados periódicamente
- Loss: $\mathcal{L} = \mathbb{E}[(r + \gamma \max_{a'}Q_{\theta^-}(s',a') - Q_\theta(s,a))^2]$
- Estabiliza entrenamiento evitando moving targets

**Arquitectura**:
- Input: Estado (e.g., pixels de juego)
- Capas convolucionales para procesamiento visual
- Capas fully-connected
- Output: Q-values para cada acción

**Variantes Avanzadas**:
- **Double DQN**: Desacopla selección y evaluación de acción, reduce overestimation
- **Dueling DQN**: Arquitectura separada para $V(s)$ y $A(s,a)$, mejora aprendizaje
- **Prioritized Experience Replay**: Muestrea transiciones importantes más frecuentemente
- **Rainbow DQN**: Combinación de múltiples mejoras

### 3.3 SARSA (State-Action-Reward-State-Action)

Algoritmo on-policy que aprende la política que está siguiendo:

$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha[r_t + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)]$$

Diferencia con Q-Learning: usa $Q(s_{t+1},a_{t+1})$ (acción realmente tomada) en lugar de $\max_{a'}Q(s_{t+1},a')$.

**Características**:
- Más conservador que Q-Learning
- Aprende política incluyendo exploración
- Útil para control online seguro

## 4. Algoritmos Policy-Based

Optimizan política directamente sin función de valor intermediaria.

### 4.1 Policy Gradient

**Objetivo**: Maximizar retorno esperado
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]$$

donde $\tau$ es trayectoria completa.

**Policy Gradient Theorem**:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s,a)]$$

**Ventajas**:
- Funciona con espacios de acción continuos
- Puede aprender políticas estocásticas
- Convergencia a óptimo local (gradiente ascendente)

### 4.2 REINFORCE (Monte Carlo Policy Gradient)

Algoritmo básico de policy gradient:

**Actualización**:
$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) G_t$$

Usa retorno completo del episodio $G_t$. Alta varianza, bajo sesgo.

**Baseline**:
Para reducir varianza: $\nabla_\theta J \approx \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s)(G_t - b(s))]$

Baseline típico: $b(s) = V(s)$

### 4.3 Métodos Actor-Critic

Combinan policy-based (actor) con value-based (critic):

**Actor**: Política $\pi_\theta(a|s)$ que selecciona acciones
**Critic**: Función de valor $V_\phi(s)$ o $Q_\phi(s,a)$ que evalúa acciones

**Ventaja**:
- Actor proporciona policy gradient
- Critic reduce varianza como baseline
- Menor varianza que REINFORCE puro

**Actualización Actor**:
$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a|s) A(s,a)$$

donde $A(s,a) = r + \gamma V(s') - V(s)$ (TD error como estimador de ventaja)

**Actualización Critic**:
$$\phi \leftarrow \phi + \beta \nabla_\phi (r + \gamma V_\phi(s') - V_\phi(s))^2$$

### 4.4 Algoritmos Avanzados

**A3C (Asynchronous Advantage Actor-Critic)**:
- Múltiples agentes en paralelo explorando independientemente
- Actualizaciones asíncronas de parámetros globales
- Diversidad de experiencias estabiliza aprendizaje

**PPO (Proximal Policy Optimization)**:
- Restricción sobre magnitud de actualización de política
- Clipping de ratio de probabilidad:
$$L^{CLIP}(\theta) = \mathbb{E}[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$
- Estado del arte en muchos dominios
- Robusto, fácil de tunear

**TRPO (Trust Region Policy Optimization)**:
- Optimización con restricción KL-divergence
- Garantías teóricas de mejora monotónica
- Más complejo computacionalmente que PPO

**SAC (Soft Actor-Critic)**:
- Maximiza entropía además de recompensa
- Fomenta exploración
- Excelente para tareas continuas

## 5. Model-Based Reinforcement Learning

Aprende modelo del entorno para planificación:

### 5.1 Aprendizaje de Dinámicas

**Modelo de Transición**: $\hat{P}(s'|s,a)$
**Modelo de Recompensa**: $\hat{R}(s,a)$

**Métodos**:
- Redes neuronales para aproximar dinámicas
- Modelos probabilísticos (Gaussian Processes, Bayesian NNs)
- Ensemble de modelos para cuantificar incertidumbre

### 5.2 Planning con Modelos

**Dyna-Q**: Combina experiencia real con simulaciones del modelo
**MCTS (Monte Carlo Tree Search)**: Búsqueda en árbol guiada por simulaciones
**MuZero**: Aprende modelo latente del entorno, planning implícito

**Trade-offs**:
- **Eficiencia de datos**: Modelos permiten más aprendizaje por interacción real
- **Costo computacional**: Planning requiere más cómputo
- **Error del modelo**: Sesgos del modelo pueden perjudicar rendimiento

## 6. Aplicaciones Emblemáticas

### 6.1 Juegos

**AlphaGo / AlphaZero**:
- Dominio de Go, ajedrez, shogi
- MCTS + Deep RL
- Self-play para generar datos

**OpenAI Five (Dota 2)**:
- PPO a escala masiva
- Coordinación multiagente
- Horizonte temporal largo

**Atari Games**:
- DQN demostró aprendizaje end-to-end desde pixels
- Benchmark estándar para algoritmos RL

### 6.2 Robótica

- Manipulación de objetos
- Locomoción (caminar, correr)
- Control de drones
- Ensamblaje industrial

### 6.3 Sistemas de Control

- Optimización de data centers (Google)
- Control de fusión nuclear (DeepMind/TAE)
- Gestión de tráfico
- Sistemas de energía renovable

### 6.4 Finanzas y Trading

- Optimización de portafolios
- Market making
- Ejecución de órdenes

### 6.5 Salud

- Personalización de tratamientos
- Dosificación óptima
- Planificación quirúrgica

## 7. Desafíos y Direcciones Futuras

### 7.1 Sample Efficiency

RL típicamente requiere millones de interacciones. Mejoras:
- Model-based RL
- Meta-learning
- Transfer learning
- Curriculum learning

### 7.2 Sparse Rewards

Cuando recompensas son infrecuentes:
- Reward shaping
- Hindsight Experience Replay
- Intrinsic motivation

### 7.3 Partial Observability

Cuando agente no observa estado completo:
- POMDPs (Partially Observable MDPs)
- Recurrent policies (LSTM, GRU)
- Belief state tracking

### 7.4 Multi-Agent RL

Múltiples agentes interactuando:
- Competitivo, cooperativo, mixto
- Equilibrios de Nash
- Comunicación entre agentes

### 7.5 Seguridad y Robustez

- Constrained RL (satisfacción de restricciones)
- Safe exploration
- Robustez ante adversarios
- Verificación formal

## Referencias

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529-533.
- Schulman, J., et al. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
- Silver, D., et al. (2017). Mastering the game of Go without human knowledge. *Nature*, 550, 354-359.
- Haarnoja, T., et al. (2018). Soft actor-critic. *ICML*.