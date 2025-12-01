# Mapa conceptual: reinforcement learning
## Sistemas de aprendizaje automático - Bloque 7

### Marco MDP (Markov Decision Process)

```
       Estado (S)
           ↓
        Agente
           ↓
      Acción (A)
           ↓
      Ambiente
           ↓
   Recompensa (R) + Nuevo Estado (S')
           ↑
           └─ Iteración
```

### Elementos clave
- **Agente**: toma decisiones
- **Ambiente**: sistema dinámico
- **Estado**: configuración actual
- **Acción**: movimiento disponible
- **Recompensa**: retroalimentación inmediata
- **Política**: estrategia de decisión
- **Valor**: utilidad esperada

### Categorías de algoritmos

```
Reinforcement Learning
    ├── Value-Based
    │   ├── Q-Learning
    │   ├── SARSA
    │   ├── Deep Q-Networks (DQN)
    │   └── Distributional RL
    ├── Policy-Based
    │   ├── Policy Gradient
    │   ├── REINFORCE
    │   ├── Actor-Critic
    │   └── PPO, A3C, TRPO
    ├── Model-Based
    │   ├── Dinámica aprendida
    │   ├── Planning
    │   └── Imagination-augmented
    └── Métodos híbridos
        ├── A3C
        ├── SAC
        └── TD3
```

### Aplicaciones
- Juegos: Ajedrez, Go, Atari
- Robótica
- Conducción autónoma
- Optimización de recursos
- Finanzas
- Control de sistemas