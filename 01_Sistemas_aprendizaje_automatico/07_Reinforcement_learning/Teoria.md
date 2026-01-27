# Aprendizaje por Refuerzo: Teoría de Decisiones Secuenciales y Agentes Autónomos
## Sistemas de Aprendizaje Automático - Bloque 7

## Resumen

El Aprendizaje por Refuerzo (Reinforcement Learning, RL) constituye un paradigma fundamental del aprendizaje automático donde agentes aprenden comportamientos óptimos mediante interacción con entornos, guiados por señales de recompensa. Este documento examina los fundamentos teóricos basados en Procesos de Decisión de Markov, algoritmos value-based y policy-based, técnicas de exploración-explotación, y aplicaciones contemporáneas. Se proporciona análisis riguroso de Q-Learning, Deep Q-Networks, Policy Gradients, Actor-Critic y métodos avanzados que han permitido logros sobresalientes en juegos, robótica y sistemas de control.

## 1. Fundamentos Teóricos del Aprendizaje por Refuerzo

### 1.1 Problema de RL y Formulación Conceptual

El aprendizaje por refuerzo aborda el problema de cómo un agente debe actuar en un entorno para maximizar recompensa acumulada a largo plazo:

**Componentes Fundamentales**:
- **Agente**: Entidad que toma decisiones y aprende
- **Entorno**: Sistema con el que el agente interacta
- **Estado ($s$)**: Representación de la situación actual
- **Acción ($a$)**: Decisión tomada por el agente
- **Recompensa ($r$)**: Señal escalar de feedback inmediato
- **Política ($\pi$)**: Estrategia de decisión del agente

**Ciclo de Interacción**:
1. Agente observa estado $s_t$
2. Agente selecciona acción $a_t$ según política $\pi$
3. Entorno transiciona a nuevo estado $s_{t+1}$
4. Agente recibe recompensa $r_{t+1}$
5. Proceso se repite

**Diferencias con Aprendizaje Supervisado**:
- No hay ejemplos etiquetados explícitos de "acciones correctas"
- Feedback retrasado: consecuencias de acciones se manifiestan en el futuro
- Acciones del agente afectan datos futuros observados
- Trade-off exploración vs explotación

### 1.2 Procesos de Decisión de Markov (MDPs)

Marco matemático formal para modelar problemas de RL:

**Definición**: Una tupla $(S, A, P, R, \gamma)$ donde:
- $S$: Conjunto finito de estados
- $A$: Conjunto finito de acciones
- $P(s'|s,a)$: Función de transición - probabilidad de $s'$ dado $s$ y $a$
- $R(s,a)$ o $R(s,a,s')$: Función de recompensa
- $\gamma \in [0,1)$: Factor de descuento

**Propiedad de Markov**:
$$P(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, \ldots, s_0, a_0) = P(s_{t+1}|s_t, a_t)$$

El futuro depende solo del presente, no de la historia completa. Esta suposición simplifica análisis pero puede ser restrictiva en problemas con información parcial.

**Retorno Descontado**:
$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

El descuento $\gamma$ modela preferencia temporal y garantiza convergencia para horizontes infinitos.

### 1.3 Políticas y Funciones de Valor

**Política $\pi$**:
- **Determinística**: $\pi: S \rightarrow A$, mapeo directo estado-acción
- **Estocástica**: $\pi(a|s)$, distribución probabilística sobre acciones

**Función de Valor de Estado**:
$$V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k+1} \Big| s_t = s\right]$$

Valor esperado del retorno al seguir política $\pi$ desde estado $s$.

**Función de Valor de Acción (Q-Function)**:
$$Q^\pi(s,a) = \mathbb{E}_\pi[G_t | s_t = s, a_t = a]$$

Valor esperado del retorno al tomar acción $a$ en estado $s$ y luego seguir $\pi$.

**Relación**:
$$V^\pi(s) = \sum_{a \in A} \pi(a|s) Q^\pi(s,a)$$

**Función de Ventaja**:
$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

Cuantifica cuánto mejor es acción $a$ respecto al promedio en estado $s$.

### 1.4 Ecuaciones de Bellman

**Ecuación de Bellman para $V^\pi$**:
$$V^\pi(s) = \sum_{a}\pi(a|s)\sum_{s'}P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

Descomposición recursiva: valor = recompensa inmediata + valor descontado del siguiente estado.

**Ecuación de Optimalidad de Bellman para $V^*$**:
$$V^*(s) = \max_a \sum_{s'}P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]$$

Define función de valor óptima que satisface cualquier política óptima.

**Ecuación de Optimalidad de Bellman para $Q^*$**:
$$Q^*(s,a) = \sum_{s'}P(s'|s,a)[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]$$

**Política Óptima**:
$$\pi^*(s) = \arg\max_a Q^*(s,a)$$

Dado $Q^*$, extracción de política óptima es directa mediante argmax.

## 2. Dilema Exploración-Explotación

### 2.1 Trade-off Fundamental

**Explotación**: Seleccionar acciones que actualmente parecen mejores según conocimiento actual (maximiza recompensa inmediata).

**Exploración**: Probar acciones subóptimas según conocimiento actual para potencialmente descubrir mejores opciones (maximiza información).

Sin exploración suficiente, el agente puede converger a políticas subóptimas. Sin explotación, el agente no capitaliza conocimiento adquirido.

### 2.2 Estrategias de Exploración

**$\epsilon$-Greedy**:
$$a_t = \begin{cases}
\arg\max_a Q(s_t, a) & \text{con probabilidad } 1-\epsilon \\
\text{acción aleatoria} & \text{con probabilidad } \epsilon
\end{cases}$$

Simple pero efectivo. $\epsilon$ típicamente decae con el tiempo: $\epsilon_t = \epsilon_0 \cdot \lambda^t$.

**Boltzmann/Softmax Exploration**:
$$\pi(a|s) = \frac{\exp(Q(s,a)/\tau)}{\sum_{a'}\exp(Q(s,a')/\tau)}$$

Parámetro de temperatura $\tau$ controla aleatoriedad. $\tau \rightarrow 0$ es greedy, $\tau \rightarrow \infty$ es uniforme.

**Upper Confidence Bound (UCB)**:
$$a_t = \arg\max_a \left[Q(s_t,a) + c\sqrt{\frac{\ln t}{N(s_t,a)}}\right]$$

Bonus de exploración basado en incertidumbre. Favorece acciones poco visitadas.

**Thompson Sampling**:
- Mantener distribución posterior sobre valores de acción
- Muestrear de posterior y actuar según muestra
- Exploración naturalmente probabilística

## 3. Métodos Value-Based

### 3.1 Q-Learning

Algoritmo off-policy para aprender $Q^*$ sin requerir modelo del entorno:

**Actualización Q-Learning**:
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[r_{t+1} + \gamma \max_{a'}Q(s_{t+1}, a') - Q(s_t, a_t)]$$

**Propiedades**:
- **Off-policy**: Aprende $Q^*$ mientras sigue política exploratoria
- **Model-free**: No requiere $P(s'|s,a)$
- **Convergencia**: Garantizada bajo condiciones estándar (visitar todos estados-acciones infinitamente, decaimiento apropiado de $\alpha$)

**Algoritmo**:
1. Inicializar $Q(s,a)$ arbitrariamente
2. Para cada episodio:
   - Inicializar $s$
   - Para cada paso:
     - Seleccionar $a$ usando política derivada de $Q$ (e.g., $\epsilon$-greedy)
     - Tomar acción $a$, observar $r, s'$
     - $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'}Q(s',a') - Q(s,a)]$
     - $s \leftarrow s'$

**Limitaciones**:
- Representación tabular impracticable para espacios de estados grandes
- No generaliza entre estados similares

### 3.2 Deep Q-Networks (DQN)

Extensión de Q-Learning usando redes neuronales profundas como aproximadores de función:

**Formulación**:
$$Q(s,a;\theta) \approx Q^*(s,a)$$

Red neuronal con parámetros $\theta$ aproxima función Q.

**Pérdida**:
$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim D}\left[\left(r + \gamma \max_{a'}Q(s',a';\theta^-) - Q(s,a;\theta)\right)^2\right]$$

**Innovaciones Clave**:

**Experience Replay**:
- Buffer almacena transiciones $(s,a,r,s')$
- Muestreo aleatorio de mini-batches para training
- Rompe correlaciones temporales
- Permite reutilización de experiencias

**Target Network**:
- Red separada $Q(s,a;\theta^-)$ con parámetros congelados
- Actualización periódica: $\theta^- \leftarrow \theta$ cada $C$ pasos
- Estabiliza training al fijar targets temporalmente

**Resultados**:
- DQN (Mnih et al., 2015): Nivel humano en 49 juegos Atari
- Uso de CNN para procesar píxeles directamente

**Variantes**:
- **Double DQN**: Desacopla selección y evaluación de acciones, reduce sobreestimación
- **Dueling DQN**: Arquitectura que separa $V(s)$ y ventajas $A(s,a)$
- **Prioritized Experience Replay**: Muestrea transiciones según TD-error
- **Rainbow**: Combina múltiples mejoras

### 3.3 SARSA (On-Policy TD Control)

Alternativa on-policy a Q-Learning:

**Actualización SARSA**:
$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha[r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t,a_t)]$$

Usa acción $a_{t+1}$ realmente tomada (según política actual), no $\max_{a'}Q(s_{t+1},a')$.

**Características**:
- On-policy: Aprende valor de política seguida
- Más conservador que Q-Learning
- Mejor para control online con exploración

## 4. Métodos Policy-Based

### 4.1 Policy Gradient

Optimización directa de política mediante gradiente ascendente en retorno esperado:

**Objetivo**:
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]$$

donde $\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \ldots)$ es trayectoria.

**Teorema de Policy Gradient**:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T}G_t \nabla_\theta \log \pi_\theta(a_t|s_t)\right]$$

**Intuición**: Incrementar probabilidad de acciones que condujeron a alto retorno.

**Ventajas sobre Value-Based**:
- Políticas estocásticas naturalmente
- Espacios de acción continuos
- Mejor para problemas con múltiples acciones óptimas

### 4.2 REINFORCE

Algoritmo Monte Carlo policy gradient:

**Actualización**:
$$\theta \leftarrow \theta + \alpha G_t \nabla_\theta \log \pi_\theta(a_t|s_t)$$

**Propiedades**:
- Estimador imparcial pero alta varianza
- Requiere episodios completos
- Convergencia a mínimo local garantizada

**Con Baseline**:
$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t (G_t - b(s_t))\nabla_\theta \log \pi_\theta(a_t|s_t)\right]$$

Baseline $b(s_t)$ reduce varianza sin introducir sesgo. Típicamente $b(s_t) = V(s_t)$.

### 4.3 Actor-Critic

Combina policy gradient (actor) con value function (critic):

**Componentes**:
- **Actor**: Política $\pi_\theta(a|s)$ actualizada con policy gradient
- **Critic**: Función de valor $V_\phi(s)$ estima retornos

**Actualización Actor**:
$$\theta \leftarrow \theta + \alpha \delta_t \nabla_\theta \log \pi_\theta(a_t|s_t)$$

**Actualización Critic**:
$$\phi \leftarrow \phi + \beta \delta_t \nabla_\phi V_\phi(s_t)$$

**TD-Error**:
$$\delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

**Ventajas**:
- Reduce varianza vs REINFORCE
- No requiere episodios completos
- Aprendizaje más estable

### 4.4 Métodos Avanzados de Policy Gradient

**A3C (Asynchronous Advantage Actor-Critic)**:
- Múltiples agentes en paralelo
- Exploraciones diversas
- Estabiliza training sin experience replay

**PPO (Proximal Policy Optimization)**:
- Restricción sobre tamaño de actualización de política
$$L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$
donde $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$
- Estado del arte en robustez y simplicidad

**TRPO (Trust Region Policy Optimization)**:
- Optimización con restricción KL
- Garantías teóricas de mejora monotónica
- Computacionalmente más costoso que PPO

**SAC (Soft Actor-Critic)**:
- Framework de máxima entropía
- Objetivo: $J(\pi) = \sum_t \mathbb{E}[r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))]$
- Exploración natural mediante regularización de entropía
- Estado del arte para control continuo

## 5. Model-Based Reinforcement Learning

### 5.1 Aprendizaje de Modelos del Entorno

Aprende dinámica $P(s'|s,a)$ y recompensa $R(s,a)$:

**Ventajas**:
- Sample efficiency: Puede planificar sin interacción adicional
- Interpretabilidad: Modelo puede ser analizado
- Transferencia: Modelo puede usarse para múltiples tareas

**Desafíos**:
- Errores de modelo se acumulan en planning
- Requiere más computación para planning
- Difícil modelar entornos complejos con precisión

### 5.2 Dyna-Q

Combina model-free learning con planning:
1. Interacción real: Actualizar $Q$ y aprender modelo
2. Planning: Simular experiencias con modelo, actualizar $Q$

Aprovecha ventajas de ambos enfoques.

### 5.3 Monte Carlo Tree Search (MCTS)

Algoritmo de búsqueda guiada por simulaciones:

**Fases**:
1. **Selection**: Descender árbol usando policy (e.g., UCT)
2. **Expansion**: Añadir nuevo nodo
3. **Simulation**: Rollout hasta terminal
4. **Backpropagation**: Propagar valor hacia raíz

Fundamental en AlphaGo, AlphaZero.

## 6. Aplicaciones Contemporáneas

### 6.1 Juegos

- **AlphaGo/AlphaZero**: Maestría sobrehumana en Go, ajedrez, shogi mediante self-play
- **OpenAI Five**: Dota 2 a nivel profesional
- **AlphaStar**: StarCraft II grandmaster level

### 6.2 Robótica

- Manipulación de objetos
- Locomoción (caminar, correr, saltar)
- Control adaptativo en entornos dinámicos
- Aprendizaje de habilidades complejas

### 6.3 Sistemas de Recomendación

- Optimización de recomendaciones a largo plazo
- Modelado de engagement de usuarios
- Balanceo exploración-explotación en contenido

### 6.4 Finanzas

- Trading algorítmico
- Gestión de portafolios
- Pricing dinámico
- Optimización de ejecución

### 6.5 Sistemas Autónomos

- Conducción autónoma
- Control de tráfico
- Gestión de recursos en data centers
- Optimización de redes

## 7. Desafíos y Direcciones Futuras

### 7.1 Sample Efficiency

RL típicamente requiere millones de interacciones. Direcciones:
- Meta-learning para transferencia
- Model-based methods
- Offline RL desde datasets existentes

### 7.2 Seguridad y Robustez

- Garantías de safety durante exploración
- Robustez ante adversarios
- Interpretabilidad de políticas aprendidas

### 7.3 Generalización

- Transfer learning entre tareas
- Multi-task RL
- Lifelong learning

### 7.4 Espacios de Acción Complejos

- Acciones jerárquicas
- Combinatorias
- Lenguaje natural

## Referencias

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529-533.
- Silver, D., et al. (2017). Mastering the game of Go without human knowledge. *Nature*, 550, 354-359.
- Schulman, J., et al. (2017). Proximal policy optimization algorithms. *arXiv*.
- Haarnoja, T., et al. (2018). Soft actor-critic. *ICML*.
