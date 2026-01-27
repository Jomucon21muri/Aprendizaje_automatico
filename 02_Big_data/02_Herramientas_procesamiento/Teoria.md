# Herramientas de Procesamiento de Big Data: Ecosistemas Spark, Hadoop y Streaming
## Big Data - Bloque 2

## Resumen

El procesamiento eficiente de datos masivos requiere herramientas especializadas que abstraen complejidad de infraestructuras distribuidas mientras proveen APIs expresivas. Este documento examina en profundidad Apache Spark como motor unificado de procesamiento in-memory, ecosistema Hadoop con MapReduce y YARN, Apache Kafka para streaming distribuido, Apache Flink para procesamiento unificado, y herramientas complementarias como Hive, Pig y Presto. Se proporciona análisis riguroso de arquitecturas, modelos de programación, optimizaciones de rendimiento, y trade-offs entre diferentes frameworks para procesamiento batch, streaming y analíticas interactivas sobre petabytes de datos.

## 1. Apache Spark: Motor Unificado de Procesamiento

### 1.1 Motivación y Arquitectura

**Limitaciones de MapReduce**:
- I/O a disco excesivo entre stages
- Solo paradigma map-reduce (no DAGs arbitrarios)
- Iteraciones ineficientes (ML, graph processing)
- Latencia alta para consultas interactivas

**Spark**: Motor de procesamiento in-memory 100x más rápido que MapReduce para iteraciones.

**Características Principales**:
- **In-memory computing**: Caché intermedio en RAM
- **DAG execution engine**: Optimización de pipelines complejos
- **API unificada**: Batch, streaming, SQL, ML, graph
- **Lazy evaluation**: Construcción de plan de ejecución antes de ejecutar

**Arquitectura**:
- **Driver Program**: Ejecuta `main()`, crea SparkContext
- **Cluster Manager**: YARN, Mesos, Standalone, Kubernetes
- **Executors**: Procesos en worker nodes, ejecutan tareas y cachean datos
- **Tasks**: Unidades de trabajo enviadas a executors

**Flujo de Ejecución**:
1. Driver crea DAG de operaciones (transformaciones y acciones)
2. DAG Scheduler divide DAG en stages basado en shuffles
3. Task Scheduler asigna tasks a executors
4. Executors ejecutan tasks, retornan resultados

### 1.2 RDDs (Resilient Distributed Datasets)

**Definición**: Colección inmutable y particionada de elementos procesables en paralelo con tolerancia a fallos mediante lineage.

**Propiedades**:
- **Inmutabilidad**: Operaciones crean nuevos RDDs
- **Particionamiento**: Datos distribuidos en nodos
- **Lazy evaluation**: Transformaciones no se ejecutan hasta acción
- **Lineage**: Grafo de dependencias para recompute

**Creación**:
```python
# Paralelizar colección
rdd = sc.parallelize([1, 2, 3, 4, 5])

# Desde archivo HDFS/S3
rdd = sc.textFile("hdfs://path/to/file.txt")

# Desde otro RDD
rdd2 = rdd.map(lambda x: x * 2)
```

**Transformaciones (Lazy)**:
- **map(f)**: Aplica función a cada elemento
- **filter(f)**: Selecciona elementos que cumplen predicado
- **flatMap(f)**: Map que puede retornar múltiples elementos
- **union(other)**: Unión de dos RDDs
- **distinct()**: Elementos únicos
- **groupByKey()**: Agrupa (K,V) por clave
- **reduceByKey(f)**: Reduce valores por clave localmente antes de shuffle
- **sortByKey()**: Ordena por clave
- **join(other)**: Join por clave

**Acciones (Eager)**:
- **collect()**: Retorna todos los elementos al driver
- **count()**: Número de elementos
- **first()**: Primer elemento
- **take(n)**: Primeros n elementos
- **reduce(f)**: Agrega elementos con función asociativa
- **saveAsTextFile(path)**: Escribe a sistema de archivos

**Ejemplo Word Count**:
```python
text_file = sc.textFile("hdfs://...")
counts = (text_file
    .flatMap(lambda line: line.split(" "))
    .map(lambda word: (word, 1))
    .reduceByKey(lambda a, b: a + b))
counts.saveAsTextFile("hdfs://output")
```

**Persistencia**:
```python
rdd.cache()  # Equivalente a persist(MEMORY_ONLY)
rdd.persist(StorageLevel.MEMORY_AND_DISK)
```

**Niveles**:
- `MEMORY_ONLY`: RAM, recompute si no cabe
- `MEMORY_AND_DISK`: Spill a disco si no cabe en RAM
- `DISK_ONLY`: Solo disco
- `MEMORY_ONLY_SER`: Serializado en RAM (menos memoria, más CPU)
- `OFF_HEAP`: Fuera de JVM heap (experimental)

### 1.3 Spark SQL y DataFrames

**Motivación**: API más declarativa y optimizable que RDDs, inspirada en R y Pandas.

**DataFrame**: Distribución distribuida de datos organizada en columnas nombradas, con schema.

**Ventajas sobre RDDs**:
- **Catalyst Optimizer**: Optimización de queries
- **Tungsten**: Code generation, memory management optimizado
- **Schema**: Información de tipos permite optimizaciones
- **API más concisa**: Similar a SQL/Pandas

**Creación**:
```python
# Desde RDD
df = spark.createDataFrame(rdd, schema=["name", "age"])

# Desde archivo Parquet/JSON/CSV
df = spark.read.parquet("hdfs://path/to/data.parquet")

# Desde tabla Hive
df = spark.sql("SELECT * FROM hive_table")
```

**Operaciones**:
```python
# Selección
df.select("name", "age")

# Filtrado
df.filter(df["age"] > 25)

# Agregación
df.groupBy("department").agg({"salary": "avg"})

# Join
df1.join(df2, df1["id"] == df2["user_id"], "inner")

# SQL directo
df.createOrReplaceTempView("people")
spark.sql("SELECT AVG(age) FROM people WHERE age > 25")
```

**Catalyst Optimizer**:
1. **Analysis**: Resolución de nombres, tipos
2. **Logical Optimization**: Predicate pushdown, constant folding, projection pruning
3. **Physical Planning**: Selección de algoritmos físicos (broadcast join vs sort-merge join)
4. **Code Generation**: Generación de bytecode Java optimizado

**Tungsten**:
- Gestión de memoria off-heap
- Cache-aware computation
- Code generation para eliminar overhead de virtualización

### 1.4 Spark Streaming

**Modelo**: Micro-batching discretizado (DStreams).

**DStream (Discretized Stream)**: Secuencia continua de RDDs.

**Arquitectura**:
- **Receiver**: Recibe datos, divide en micro-batches
- **Batch Interval**: Típicamente segundos
- **Processing**: Cada batch se procesa como RDD

**Ejemplo**:
```python
from pyspark.streaming import StreamingContext

ssc = StreamingContext(sc, 1)  # Batch cada 1 segundo

lines = ssc.socketTextStream("localhost", 9999)
words = lines.flatMap(lambda line: line.split(" "))
pairs = words.map(lambda word: (word, 1))
wordCounts = pairs.reduceByKey(lambda a, b: a + b)
wordCounts.pprint()

ssc.start()
ssc.awaitTermination()
```

**Windowed Operations**:
```python
# Cuenta palabras en ventanas de 30s, deslizando cada 10s
windowedWordCounts = pairs.reduceByKeyAndWindow(
    lambda a, b: a + b, 
    lambda a, b: a - b,  # Función inversa para eficiencia
    windowDuration=30, 
    slideDuration=10
)
```

**Limitaciones**:
- Latencia mínima del batch interval (segundos)
- No true streaming record-at-a-time
- Complejidad en manejo de estado

### 1.5 Structured Streaming

**Evolución**: API unificada sobre DataFrames/Datasets para streaming.

**Modelo**: Tabla infinita en append-only, queries incrementales.

**Ventajas**:
- Event-time processing nativo
- Watermarking para late data
- Exactamente-once semantics
- API idéntica a batch

**Ejemplo**:
```python
# Streaming DataFrame
lines = spark.readStream.format("socket").load()

# Query idéntica a batch
wordCounts = (lines
    .groupBy("word")
    .count())

# Output
query = (wordCounts
    .writeStream
    .format("console")
    .outputMode("complete")
    .start())
```

**Event-Time y Watermarking**:
```python
events = spark.readStream.json("events")

windowed = (events
    .withWatermark("timestamp", "10 minutes")
    .groupBy(
        window(events.timestamp, "5 minutes"),
        events.userId
    )
    .count())
```

**Output Modes**:
- **Append**: Solo nuevas filas
- **Complete**: Toda tabla resultado
- **Update**: Solo filas actualizadas

### 1.6 MLlib

Librería distribuida de machine learning:

**Algoritmos**:
- **Classification**: Logistic Regression, Decision Trees, Random Forests, GBT, Naive Bayes
- **Regression**: Linear, Ridge, Lasso, Decision Trees
- **Clustering**: K-Means, GMM, LDA
- **Collaborative Filtering**: ALS
- **Dimensionality Reduction**: PCA, SVD

**Pipelines**:
```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

assembler = VectorAssembler(inputCols=["feature1", "feature2"], outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="label")

pipeline = Pipeline(stages=[assembler, lr])
model = pipeline.fit(train_df)
predictions = model.transform(test_df)
```

### 1.7 GraphX

Procesamiento de grafos distribuidos:

**Graph**: Vértices y aristas con propiedades.

**Operaciones**:
- **PageRank**: Importancia de vértices
- **Connected Components**: Componentes conectados
- **Triangle Counting**: Detección de triángulos
- **Pregel API**: Computación iterativa estilo BSP

##2. Ecosistema Hadoop

### 2.1 Apache Hive

**SQL sobre Hadoop**: Data warehouse para analíticas batch.

**Arquitectura**:
- **Metastore**: Esquemas de tablas (usualmente RDBMS)
- **HiveQL**: Lenguaje similar a SQL
- **Execution Engine**: MapReduce, Tez, Spark

**Ejemplo**:
```sql
CREATE TABLE users (
  id INT,
  name STRING,
  age INT
)
PARTITIONED BY (country STRING)
STORED AS PARQUET;

INSERT OVERWRITE TABLE users PARTITION(country='US')
SELECT id, name, age FROM raw_users WHERE country = 'US';

SELECT country, AVG(age) 
FROM users 
GROUP BY country
HAVING AVG(age) > 30;
```

**Partitioning**:
```sql
-- Partición por fecha
PARTITIONED BY (year INT, month INT, day INT)

-- Query con partition pruning
SELECT * FROM logs WHERE year=2024 AND month=1;
```

**Bucketing**:
```sql
CLUSTERED BY (user_id) INTO 256 BUCKETS
```

Distribuye datos en número fijo de archivos, optimiza joins.

**UDFs**:
```python
# Python UDF
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

def square(x):
    return x * x

square_udf = udf(square, IntegerType())
df.select(square_udf("value"))
```

### 2.2 Apache Pig

**Dataflow Language**: Pig Latin para ETL.

**Características**:
- Procedural vs declarative (SQL)
- Optimizado para exploración de datos
- Extensible con UDFs

**Ejemplo**:
```pig
users = LOAD 'hdfs://users.txt' AS (id:int, name:chararray, age:int);
filtered = FILTER users BY age > 25;
grouped = GROUP filtered BY age;
counts = FOREACH grouped GENERATE group AS age, COUNT(filtered) AS count;
ordered = ORDER counts BY count DESC;
top10 = LIMIT ordered 10;
STORE top10 INTO 'hdfs://output';
```

**Operadores**:
- **LOAD/STORE**: I/O
- **FILTER**: Selección
- **FOREACH**: Proyección/transformación
- **GROUP/COGROUP**: Agrupación
- **JOIN**: Joins de bags
- **UNION**: Unión de bags
- **DISTINCT**: Duplicados

### 2.3 Apache Tez

**DAG execution engine** para Hive y Pig, reemplaza MapReduce.

**Ventajas**:
- DAGs arbitrarios (no solo map-reduce)
- Reuso de contenedores (menos overhead)
- Pipeline de tasks (reduce latencia)

**Mejoras de Performance**:
- Hive sobre Tez: 3-10x más rápido que MapReduce
- Menos materializaciones intermedias

## 3. Apache Kafka: Plataforma de Streaming Distribuida

### 3.1 Arquitectura

**Modelo**: Distributed commit log.

**Componentes**:
- **Topics**: Categorías de mensajes
- **Partitions**: Logs ordenados e inmutables, unidad de paralelismo
- **Brokers**: Servidores Kafka
- **Producers**: Publican mensajes
- **Consumers**: Suscriben y consumen mensajes
- **Consumer Groups**: Múltiples consumidores colaboran, cada partición asignada a un consumidor

**Garantías**:
- **Orden**: Dentro de partición
- **Durabilidad**: Replicación configurable
- **Escalabilidad**: Horizontal vía particiones

### 3.2 Producers

```python
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

producer.send('my-topic', {'key': 'value'})
producer.flush()
```

**Partitioning**:
- Por key (hash-based)
- Round-robin si sin key
- Custom partitioner

**Acks**:
- `acks=0`: No esperar confirmación (rápido, posible pérdida)
- `acks=1`: Esperar líder (balance)
- `acks=all`: Esperar todas replicas in-sync (durable, lento)

### 3.3 Consumers

```python
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'my-topic',
    bootstrap_servers=['localhost:9092'],
    group_id='my-group',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

for message in consumer:
    process(message.value)
```

**Consumer Groups**:
- Escalabilidad: Añadir consumers
- Fault tolerance: Rebalanceo automático
- Cada partición a máximo un consumer del grupo

**Offsets**:
- Posición de lectura en partición
- Commit manual o automático
- Storage en `__consumer_offsets` topic

### 3.4 Kafka Streams

**Librería para procesamiento de streams**:

```java
StreamsBuilder builder = new StreamsBuilder();

KStream<String, String> source = builder.stream("input-topic");

KTable<String, Long> counts = source
    .flatMapValues(value -> Arrays.asList(value.split("\\s+")))
    .groupBy((key, word) -> word)
    .count();

counts.toStream().to("output-topic");

KafkaStreams streams = new KafkaStreams(builder.build(), config);
streams.start();
```

**Conceptos**:
- **KStream**: Inmutable, append-only
- **KTable**: Mutable, changelog stream
- **GlobalKTable**: Replicada en todas instancias

**Operaciones**:
- **Stateless**: map, filter, flatMap
- **Stateful**: aggregations, joins, windowing

## 4. Apache Flink: Procesamiento Unificado True Streaming

### 4.1 Arquitectura

**Modelo**: Streaming nativo, batch como caso especial.

**Componentes**:
- **JobManager**: Coordina ejecución
- **TaskManagers**: Workers que ejecutan tareas
- **DataStream API**: Streaming
- **DataSet API**: Batch (deprecada, usar DataStream)

**Características**:
- **True streaming**: Record-at-a-time processing
- **Event time**: Soporte nativo de timestamps
- **Exactly-once state**: Via checkpointing distribuido
- **Low latency**: Milisegundos

### 4.2 DataStream API

```java
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

DataStream<String> text = env.socketTextStream("localhost", 9999);

DataStream<Tuple2<String, Integer>> counts = text
    .flatMap(new Tokenizer())
    .keyBy(0)
    .sum(1);

counts.print();
env.execute("Word Count");
```

**Event Time Processing**:
```java
env.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);

DataStream<Event> events = ...;

events
    .assignTimestampsAndWatermarks(
        WatermarkStrategy
            .<Event>forBoundedOutOfOrderness(Duration.ofSeconds(10))
            .withTimestampAssigner((event, timestamp) -> event.getTimestamp())
    )
    .keyBy(Event::getKey)
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .reduce(new SumReducer());
```

### 4.3 State Management

**Keyed State**:
- **ValueState**: Un valor por key
- **ListState**: Lista de valores
- **MapState**: Map de key-value
- **ReducingState**: Valor agregado con ReduceFunction

```java
ValueStateDescriptor<Long> descriptor = 
    new ValueStateDescriptor<>("count", Long.class, 0L);

ValueState<Long> count = getRuntimeContext().getState(descriptor);
Long currentCount = count.value();
count.update(currentCount + 1);
```

**Checkpointing**:
- Snapshots periódicos de estado distribuido
- Algoritmo Chandy-Lamport
- Backends: Memory, RocksDB, HDFS

### 4.4 Exactly-Once Semantics

**Two-Phase Commit con Sinks**:
- Pre-commit durante checkpointing
- Commit cuando checkpoint completo

**Garantías**:
- Exactly-once state updates
- End-to-end exactly-once con sinks compatibles (Kafka con transacciones)

## 5. Presto: Motor de Query Distribuido

**Consultas interactivas** sobre data lakes (S3, HDFS, Cassandra, etc.).

**Arquitectura**:
- **Coordinator**: Parsing, planning, scheduling
- **Workers**: Ejecutan tasks

**Características**:
- Baja latencia (segundos)
- SQL estándar (ANSI SQL)
- Federación: Query a múltiples fuentes

**Ejemplo**:
```sql
-- Query federado
SELECT u.name, o.total
FROM hive.users.users u
JOIN mysql.orders.orders o ON u.id = o.user_id
WHERE o.date >= CURRENT_DATE - INTERVAL '7' DAY;
```

**Optimizaciones**:
- Predicate pushdown a fuentes
- Columnar execution
- Vectorized processing
- Cost-based optimizer

## 6. Selección de Herramientas

| Criterio | Spark | Flink | Kafka Streams | Hive | Presto |
|----------|-------|-------|---------------|------|--------|
| **Latencia** | Segundos | Milisegundos | Milisegundos | Minutos | Segundos |
| **Throughput** | Alto | Alto | Medio | Muy alto | Alto |
| **Streaming** | Micro-batch | True stream | True stream | No | No |
| **SQL** | Excelente | Bueno | No | Excelente | Excelente |
| **ML** | MLlib | FlinkML | No | Básico | No |
| **Madurez** | Muy alta | Alta | Media | Muy alta | Alta |
| **Ecosistema** | Amplio | Creciendo | Kafka | Hadoop | Standalone |

**Recomendaciones**:
- **Batch ETL complejo**: Spark
- **Streaming baja latencia**: Flink
- **Event streaming simple**: Kafka Streams
- **Data warehouse SQL**: Hive
- **Analíticas interactivas**: Presto, Spark SQL
- **ML distribuido**: Spark MLlib

## 7. Optimizaciones de Performance

### 7.1 Spark

**Partitioning**: 
- Reparticionar para balance: `repartition(n)`
- Coalesce para reducir particiones: `coalesce(n)`
- Particiones típicas: 2-4x número de cores

**Broadcasting**:
```python
broadcast_var = sc.broadcast(small_data)
large_rdd.map(lambda x: lookup(broadcast_var.value, x))
```

**Persist Strategy**:
- Cachear RDDs reutilizados
- Nivel apropiado según disponibilidad de memoria

**Avoid Shuffles**:
- `reduceByKey` mejor que `groupByKey` + `reduce`
- Broadcast joins para tablas pequeñas

**Catalyst**:
- Usar DataFrames sobre RDDs cuando posible
- Filtros temprano en query

### 7.2 Kafka

**Partitions**: Más particiones → más paralelismo (límite: overhead)

**Batch Size**: Producer batching reduce latency y aumenta throughput

**Compression**: Snappy (balance), Gzip (ratio), LZ4 (rápido)

**Replication Factor**: Balance durabilidad vs performance (típico: 3)

## Referencias

- Zaharia, M., et al. (2012). Resilient distributed datasets: A fault-tolerant abstraction for in-memory cluster computing. *NSDI*.
- Armbrust, M., et al. (2015). Spark SQL: Relational data processing in Spark. *SIGMOD*.
- Carbone, P., et al. (2015). Apache Flink: Stream and batch processing in a single engine. *IEEE Data Engineering Bulletin*.
- Kreps, J., et al. (2011). Kafka: A distributed messaging system for log processing. *NetDB*.
- Thusoo, A., et al. (2009). Hive: A warehousing solution over a map-reduce framework. *VLDB*.
- Melnik, S., et al. (2010). Dremel: Interactive analysis of web-scale datasets. *VLDB*. [Inspiration for Presto/Drill]
- Akidau, T., et al. (2015). The Dataflow Model: A practical approach to balancing correctness, latency, and cost. *VLDB*.
