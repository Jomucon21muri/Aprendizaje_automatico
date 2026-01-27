# Análisis de Datos Masivos: Pipelines ETL, Analíticas Distribuidas y Visualización
## Big Data - Bloque 3

## Resumen

El análisis de datos masivos requiere pipelines sofisticados que orquestan extracción, transformación, carga, analíticas distribuidas y visualización a escalas de petabytes. Este documento examina arquitecturas de pipelines ETL/ELT, técnicas de procesamiento analítico (OLAP) sobre infraestructuras distribuidas, integración de machine learning en workflows de datos masivos, streaming analytics para insights en tiempo real, y estrategias de visualización que sintetizan billones de puntos de datos en dashboards accionables. Se proporciona análisis riguroso de trade-offs entre batch y streaming, optimizaciones de queries distribuidas, y herramientas del ecosistema moderno de datos.

## 1. Pipelines ETL/ELT para Big Data

### 1.1 ETL vs ELT

**ETL (Extract, Transform, Load)** - Tradicional:
1. **Extract**: Obtener datos de fuentes heterogéneas
2. **Transform**: Limpieza, normalización, enriquecimiento en sistema intermedio
3. **Load**: Cargar datos transformados a data warehouse

**ELT (Extract, Load, Transform)** - Moderno para Big Data:
1. **Extract**: Obtener datos raw
2. **Load**: Cargar datos sin transformar a data lake
3. **Transform**: Procesamiento in-situ con motores como Spark

**Motivaciones para ELT**:
- **Escalabilidad**: Data lakes (HDFS, S3) escalan horizontalmente
- **Flexibilidad**: Datos raw preservados, múltiples transformaciones posibles
- **Cost-effectiveness**: Almacenamiento barato, compute efímero
- **Schema-on-read**: Aplicar esquema al leer, no al escribir

**Trade-offs**:
- ETL: Datos limpios desde inicio, menor flexibilidad
- ELT: Mayor almacenamiento raw, transformaciones pueden ser costosas

### 1.2 Componentes de Pipelines Modernos

**Ingesta**:
- **Batch**: Sqoop (RDBMS → HDFS), Flume (logs), distcp (transferencia HDFS)
- **Streaming**: Kafka, Kinesis, Pub/Sub
- **Change Data Capture (CDC)**: Debezium captura cambios de bases de datos

**Orquestación**:
- **Apache Airflow**: DAGs Python para workflows complejos
- **Luigi**: Spotify, pipelines batch
- **Oozie**: Nativo Hadoop (coordinador, bundles)
- **Prefect/Dagster**: Modernos, Python-first

**Transformación**:
- **Batch**: Spark, Hive, Presto
- **Streaming**: Flink, Spark Streaming, Kafka Streams
- **SQL-based**: dbt (data build tool) sobre warehouses

**Storage**:
- **Raw Zone**: Data lake (S3, ADLS, GCS) - Datos sin procesar
- **Curated Zone**: Datos limpios y transformados
- **Serving Zone**: Agregaciones para consumo (BI, ML)

### 1.3 Ejemplo Pipeline con Airflow

```python
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'user_analytics_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
)

extract_task = PythonOperator(
    task_id='extract_from_mysql',
    python_callable=extract_mysql_data,
    dag=dag
)

transform_task = SparkSubmitOperator(
    task_id='transform_with_spark',
    application='/jobs/transform_users.py',
    conf={'spark.executor.memory': '4g'},
    dag=dag
)

load_task = PythonOperator(
    task_id='load_to_redshift',
    python_callable=load_to_warehouse,
    dag=dag
)

extract_task >> transform_task >> load_task
```

**Características Airflow**:
- **DAGs**: Directed Acyclic Graphs definen dependencias
- **Schedulers**: Ejecución basada en cron o triggers
- **Backfilling**: Ejecutar tareas históricas
- **Monitoring**: UI web para seguimiento

### 1.4 Data Quality

**Validaciones**:
```python
# Great Expectations ejemplo
expectation_suite = {
    "expect_column_values_to_not_be_null": {"column": "user_id"},
    "expect_column_values_to_be_unique": {"column": "user_id"},
    "expect_column_values_to_be_between": {
        "column": "age",
        "min_value": 0,
        "max_value": 120
    },
    "expect_column_values_to_match_regex": {
        "column": "email",
        "regex": r"^[\w\.-]+@[\w\.-]+\.\w+$"
    }
}
```

**Data Observability**:
- **Lineage**: Tracking de origen y transformaciones de datos
- **Freshness**: Detectar pipelines estancados
- **Volume**: Alertas en cambios anómalos de volumen
- **Schema**: Monitoreo de cambios de esquema

## 2. Analíticas OLAP Distribuidas

### 2.1 OLAP vs OLTP

**OLTP (Online Transaction Processing)**:
- Operaciones transaccionales (CRUD)
- Queries simples, baja latencia
- Alta concurrencia de escrituras
- Datos normalizados
- Ejemplo: Base de datos de comercio electrónico

**OLAP (Online Analytical Processing)**:
- Consultas analíticas complejas
- Aggregations, joins masivos
- Principalmente lecturas
- Datos desnormalizados (esquemas estrella/copo de nieve)
- Ejemplo: Data warehouse para BI

### 2.2 Esquemas Dimensionales

**Esquema Estrella**:
- **Fact Table**: Métricas cuantitativas (ventas, clics, transacciones)
- **Dimension Tables**: Contexto descriptivo (usuario, producto, tiempo, ubicación)

```sql
-- Fact Table
CREATE TABLE fact_sales (
    sale_id BIGINT,
    date_id INT,
    product_id INT,
    user_id INT,
    store_id INT,
    quantity INT,
    revenue DECIMAL(10,2),
    cost DECIMAL(10,2)
);

-- Dimension Tables
CREATE TABLE dim_date (date_id INT, date DATE, year INT, quarter INT, month INT);
CREATE TABLE dim_product (product_id INT, name STRING, category STRING);
CREATE TABLE dim_user (user_id INT, name STRING, segment STRING);
```

**Queries Analíticas**:
```sql
-- Top productos por categoría en Q1 2024
SELECT 
    p.category,
    p.name,
    SUM(f.revenue) AS total_revenue
FROM fact_sales f
JOIN dim_product p ON f.product_id = p.product_id
JOIN dim_date d ON f.date_id = d.date_id
WHERE d.year = 2024 AND d.quarter = 1
GROUP BY p.category, p.name
ORDER BY total_revenue DESC
LIMIT 10;
```

**Ventajas Esquema Estrella**:
- Queries más simples (menos joins)
- Mejor performance (denormalization)
- Fácil entendimiento para analistas

### 2.3 Cubos OLAP

**Operaciones**:
- **Roll-up**: Agregación a granularidad mayor (día → mes)
- **Drill-down**: Descomposición a granularidad menor (mes → día)
- **Slice**: Seleccionar dimensión específica (solo 2024)
- **Dice**: Subcubo con filtros múltiples
- **Pivot**: Rotación de dimensiones

**Ejemplo Cube**:
- Dimensiones: Tiempo, Producto, Ubicación
- Medidas: Ventas, Unidades
- Cube precomputa agregaciones para acceso rápido

### 2.4 Columnar Storage

**Row-oriented** (RDBMS tradicional):
```
[1, "Alice", 25, "US"] [2, "Bob", 30, "UK"] [3, "Charlie", 35, "US"]
```

**Column-oriented** (Parquet, ORC):
```
IDs: [1, 2, 3]
Names: ["Alice", "Bob", "Charlie"]
Ages: [25, 30, 35]
Countries: ["US", "UK", "US"]
```

**Ventajas para Analíticas**:
- **I/O Reducido**: Leer solo columnas necesarias
- **Compresión Superior**: Datos homogéneos comprimen mejor
- **Vectorización**: Operaciones SIMD eficientes
- **Predicate Pushdown**: Filtros aplicados al leer

**Performance**:
- Consultas analíticas (suma, promedio): 10-100x más rápidas
- Trade-off: Escrituras más lentas (pero analíticas son read-heavy)

### 2.5 Partitioning y Pruning

**Partitioning Estratégico**:
```sql
-- Particionado por fecha (común en analíticas)
CREATE TABLE events (
    user_id INT,
    event_type STRING,
    timestamp TIMESTAMP
)
PARTITIONED BY (year INT, month INT, day INT);

-- Query con partition pruning
SELECT COUNT(*)
FROM events
WHERE year = 2024 AND month = 1 AND event_type = 'purchase';
-- Solo lee particiones enero 2024
```

**Partition Pruning**: Query engine omite particiones irrelevantes basado en predicados.

## 3. Machine Learning a Escala

### 3.1 Feature Engineering Distribuido

**Agregaciones Temporales**:
```python
from pyspark.sql import functions as F, Window

# Feature: Compras últimos 30 días
window_30d = Window.partitionBy("user_id").orderBy("timestamp").rangeBetween(-30*86400, 0)

features = purchases.withColumn(
    "purchases_last_30d",
    F.count("*").over(window_30d)
).withColumn(
    "avg_amount_last_30d",
    F.avg("amount").over(window_30d)
)
```

**Feature Store**:
- **Feast**, **Tecton**, **Hopsworks**: Repositorios centralizados de features
- Reutilización entre modelos
- Consistencia training/serving
- Versionado y lineage

### 3.2 Entrenamiento Distribuido

**Data Parallelism** (Spark MLlib):
- Datos particionados, modelo replicado
- Gradientes agregados

**Model Parallelism**:
- Modelo muy grande, particionado en múltiples nodos
- Necesario para LLMs (GPT, LLaMA)

**Hyperparameter Tuning**:
```python
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=RegressionEvaluator(),
    numFolds=5,
    parallelism=10  # Paralelización
)

cvModel = crossval.fit(train)
```

### 3.3 Batch Inference

**Scoring de Modelos**:
```python
# Cargar modelo entrenado
model = PipelineModel.load("s3://models/churn_model_v2")

# Inferencia sobre datos masivos
predictions = model.transform(users_features)

# Guardar scores
predictions.select("user_id", "prediction", "probability") \
    .write.parquet("s3://scores/churn_predictions_2024-01-27")
```

**Optimizaciones**:
- **Broadcast**: Para modelos pequeños (<100MB)
- **Caching**: Si múltiples pases sobre datos
- **Partitioning**: Balance de carga

## 4. Stream Analytics

### 4.1 Real-Time Aggregations

**Tumbling Windows** (sin superposición):
```python
# Spark Structured Streaming
aggregated = (streaming_df
    .withWatermark("timestamp", "10 minutes")
    .groupBy(
        window("timestamp", "5 minutes"),
        "user_id"
    )
    .agg(
        F.count("*").alias("event_count"),
        F.sum("amount").alias("total_amount")
    ))
```

**Sliding Windows** (con superposición):
```python
# Ventana de 10 minutos, deslizando cada 5 minutos
window("timestamp", "10 minutes", "5 minutes")
```

**Session Windows** (basadas en inactividad):
```python
# Flink
stream
    .keyBy(_.userId)
    .window(EventTimeSessionWindows.withGap(Time.minutes(30)))
    .reduce((a, b) => a + b)
```

### 4.2 Complex Event Processing (CEP)

**Detección de Patrones**:
```java
// Flink CEP: Detectar 3 fallos consecutivos en 5 minutos
Pattern<Event, ?> pattern = Pattern.<Event>begin("start")
    .where(new SimpleCondition<Event>() {
        public boolean filter(Event event) {
            return event.getType().equals("error");
        }
    })
    .times(3).consecutive()
    .within(Time.minutes(5));

PatternStream<Event> patternStream = CEP.pattern(input, pattern);
```

**Casos de Uso**:
- Detección de fraude en transacciones
- Monitoreo de infraestructura (3 errores consecutivos → alerta)
- Trading algorítmico (patrones de mercado)

### 4.3 State Management en Streaming

**Keyed State** (Flink):
```java
public class CountFunction extends RichFlatMapFunction<Event, Tuple2<String, Long>> {
    private transient ValueState<Long> count;
    
    @Override
    public void open(Configuration config) {
        ValueStateDescriptor<Long> descriptor = 
            new ValueStateDescriptor<>("count", Long.class, 0L);
        count = getRuntimeContext().getState(descriptor);
    }
    
    @Override
    public void flatMap(Event event, Collector<Tuple2<String, Long>> out) {
        Long currentCount = count.value();
        currentCount++;
        count.update(currentCount);
        out.collect(new Tuple2<>(event.getKey(), currentCount));
    }
}
```

**Checkpointing**: Snapshots periódicos de estado para fault tolerance.

## 5. Visualización de Big Data

### 5.1 Desafíos

**Escalabilidad**:
- Browsers limitan rendering a millones de puntos
- Imposible visualizar billones de registros directamente

**Latencia**:
- Queries interactivos deben retornar en segundos
- Dashboards en tiempo real

**Interactividad**:
- Drill-down, filtrado, zoom sin recargar
- Exploración ad-hoc

### 5.2 Estrategias

**Agregación Previa**:
- Cubos OLAP precomputados
- Aggregation tables (Druid, Pinot)
- Reduce datos de TB a MB para visualización

**Sampling**:
- Muestras representativas para exploración inicial
- Reservoir sampling para streams

**Aproximaciones**:
- HyperLogLog para conteos distintos
- T-Digest para cuantiles
- Sketches probabilísticos

**Progressive Rendering**:
- Muestra aproximación rápida, refina progresivamente
- Ejemplo: Superset con queries asíncronas

### 5.3 Herramientas Modernas

**Business Intelligence**:
- **Tableau**: Conectores nativos a Hadoop, Spark
- **Power BI**: Azure Synapse, Databricks
- **Looker**: Modelling layer sobre warehouses
- **Superset**: Open-source, SQL-based

**Programáticas**:
- **Plotly Dash**: Python dashboards interactivos
- **Observable**: JavaScript notebooks
- **Streamlit**: Python apps rápidas

**Embedded Analytics**:
- **Apache Zeppelin**: Notebooks sobre Spark
- **Jupyter + BigQuery Magic**: Notebooks escalables

### 5.4 Ejemplo Dashboard con Superset

**Configuración**:
```python
# Conectar a Presto
DATABASE_URI = 'presto://coordinator:8080/hive/default'

# Dataset sobre tabla particionada
dataset = {
    'table_name': 'user_events',
    'sql': '''
        SELECT 
            date_trunc('hour', timestamp) AS hour,
            event_type,
            COUNT(*) AS event_count
        FROM events
        WHERE timestamp >= CURRENT_DATE - INTERVAL '7' DAY
        GROUP BY 1, 2
    ''',
    'cache_timeout': 300  # 5 minutos
}
```

**Dashboard**:
- Time-series chart: Eventos por hora
- Pie chart: Distribución de event_type
- Table: Top users por actividad
- Filtros interactivos: Fecha, tipo evento

## 6. Data Warehouses Modernos

### 6.1 Cloud Data Warehouses

**Características**:
- Separación compute-storage
- Auto-scaling
- Pago por uso
- Performance optimizado para analíticas

**Amazon Redshift**:
- Arquitectura MPP (Massively Parallel Processing)
- Columnar storage
- Spectrum: Queries sobre S3 directo
- Concurrency Scaling

**Google BigQuery**:
- Serverless
- Columnar (Capacitor format)
- ML integrado (BigQuery ML)
- Streaming inserts

**Snowflake**:
- Multi-cloud (AWS, Azure, GCP)
- Time Travel: Queries históricas
- Zero-copy cloning
- Secure data sharing

### 6.2 Ejemplo Query Optimization

**Mal**:
```sql
-- Cross join accidental, billones de filas
SELECT *
FROM users, events
WHERE users.country = 'US';
```

**Bien**:
```sql
-- Join explícito, predicate pushdown
SELECT e.*
FROM users u
INNER JOIN events e ON u.user_id = e.user_id
WHERE u.country = 'US' AND e.date >= '2024-01-01';
```

**Best Practices**:
- **Filtros tempranos**: Predicate pushdown reduce datos leídos
- **Projection pruning**: SELECT solo columnas necesarias
- **Partitions**: Alinear filtros con partitioning
- **Joins**: Broadcast pequeño, sort-merge grande
- **Aggregations**: Pre-aggregate cuando posible

## 7. Lakehouse Architecture

### 7.1 Concepto

**Data Lake** (tradicional):
- Storage barato (S3, ADLS)
- Schema-on-read
- Flexibilidad
- **Problemas**: Sin transacciones, calidad inconsistente, performance subóptimo para analíticas

**Data Warehouse** (tradicional):
- Performance alto
- ACID transactions
- Governance fuerte
- **Problemas**: Costoso, menos flexible, ETL complejo

**Lakehouse**: Mejor de ambos mundos.
- Storage en data lake
- Metadata layer provee ACID, schema evolution, time travel
- Query engines optimizados

### 7.2 Formatos Lakehouse

**Delta Lake** (Databricks):
```python
# Escritura con transacciones ACID
df.write.format("delta").mode("overwrite").save("/delta/users")

# Time travel
df = spark.read.format("delta").option("versionAsOf", 3).load("/delta/users")

# Schema evolution
df_new_schema.write.format("delta").mode("append").option("mergeSchema", "true").save("/delta/users")

# MERGE (upserts)
deltaTable.alias("target").merge(
    updates.alias("source"),
    "target.user_id = source.user_id"
).whenMatchedUpdate(set = {"status": "source.status"}) \
 .whenNotMatchedInsert(values = {...}) \
 .execute()
```

**Apache Iceberg** (Netflix):
- Snapshot isolation
- Hidden partitioning (transparente para queries)
- Partition evolution sin reescribir datos

**Apache Hudi** (Uber):
- **Copy-on-Write**: Reescribir archivos en updates (analíticas rápidas)
- **Merge-on-Read**: Append updates, merge al leer (escrituras rápidas)

### 7.3 Ventajas Lakehouse

- **Costo**: Storage barato de data lakes
- **Flexibilidad**: Múltiples formatos y herramientas
- **Performance**: Optimizaciones de warehouses
- **Governance**: ACID, versioning, audit logs
- **ML directo**: Entrenar sobre data lake sin ETL a warehouse

## 8. Optimización de Queries Distribuidas

### 8.1 Catalyst Optimizer (Spark)

**Fases**:
1. **Analysis**: Resolver nombres, tipos
2. **Logical Optimization**:
   - **Predicate Pushdown**: Filtros lo más abajo posible
   - **Constant Folding**: `1 + 1` → `2` en compile-time
   - **Projection Pruning**: Solo columnas usadas
   - **Join Reordering**: Orden óptimo de joins
3. **Physical Planning**:
   - **Broadcast Hash Join**: Tabla pequeña broadcast a todos
   - **Sort-Merge Join**: Para tablas grandes
4. **Code Generation**: Bytecode Java optimizado

### 8.2 Cost-Based Optimization

**Statistics**:
- Row counts, distinct values, histogramas
- Spark: `ANALYZE TABLE users COMPUTE STATISTICS`

**Join Selection**:
- Broadcast si tabla < `spark.sql.autoBroadcastJoinThreshold` (default 10MB)
- Sino, sort-merge join

### 8.3 Adaptive Query Execution (Spark 3.0+)

**Optimizaciones Runtime**:
- **Coalescing Shuffle Partitions**: Reduce particiones si datos pequeños
- **Converting Sort-Merge to Broadcast**: Si tabla resulta pequeña post-filtro
- **Skew Join Optimization**: Detecta y corrige skew en tiempo real

## 9. Tendencias y Futuro

### 9.1 Real-Time OLAP

- **Apache Druid**, **Apache Pinot**: Sub-second queries sobre eventos streaming
- **ClickHouse**: OLAP altísima performance

### 9.2 Federated Queries

- **Presto/Trino**: Query a través de múltiples fuentes (Hive, MySQL, Postgres, MongoDB) en una query
- Data virtualization: Sin mover datos

### 9.3 AutoML en Big Data

- **H2O.ai Driverless AI**: AutoML sobre Spark
- **Google Cloud AutoML Tables**: BigQuery integrado
- Democratización de ML para analistas

### 9.4 Data Mesh

- **Descentralización**: Domain-oriented data ownership
- **Data as a Product**: Equipos publican datos como productos con SLAs
- **Self-serve platform**: Infraestructura centralizada, uso descentralizado

## Referencias

- Kimball, R., & Ross, M. (2013). *The Data Warehouse Toolkit* (3rd ed.). Wiley.
- Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly.
- Armbrust, M., et al. (2015). Spark SQL: Relational data processing in Spark. *SIGMOD*.
- Zaharia, M., et al. (2018). Accelerating the Machine Learning Lifecycle with MLflow. *IEEE Data Engineering Bulletin*.
- Dehghani, Z. (2022). *Data Mesh*. O'Reilly.
- Akidau, T., et al. (2015). The Dataflow Model. *VLDB*.
- Abadi, D., et al. (2013). The Design and Implementation of Modern Column-Oriented Database Systems. *Foundations and Trends in Databases*.
