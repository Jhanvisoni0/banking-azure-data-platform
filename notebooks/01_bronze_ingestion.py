# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze Ingestion
# MAGIC Raw CSVs → Delta tables. No transformations, data stays as-is.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *

STORAGE_ACCOUNT = "bankinganalyticsdls"
storage_key     = "<YOUR_KEY1_HERE>"

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    storage_key
)

BRONZE_BASE = f"abfss://bronze@{STORAGE_ACCOUNT}.dfs.core.windows.net"
SILVER_BASE = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net"
GOLD_BASE   = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net"

print("config loaded")

# COMMAND ----------

df_customers = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(f"{BRONZE_BASE}/customers/customers.csv")
    .withColumn("_ingestion_timestamp", F.current_timestamp())
    .withColumn("_source_file", F.lit("customers.csv"))
    .withColumn("_bronze_layer", F.lit("bronze"))
)

df_customers.write.format("delta").mode("overwrite").save(f"{BRONZE_BASE}/delta/customers")
print(f"customers: {spark.read.format('delta').load(f'{BRONZE_BASE}/delta/customers').count():,} rows")

# COMMAND ----------

df_loans = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(f"{BRONZE_BASE}/loans/loans.csv")
    .withColumn("_ingestion_timestamp", F.current_timestamp())
    .withColumn("_source_file", F.lit("loans.csv"))
    .withColumn("_bronze_layer", F.lit("bronze"))
)

df_loans.write.format("delta").mode("overwrite").save(f"{BRONZE_BASE}/delta/loans")
print(f"loans: {spark.read.format('delta').load(f'{BRONZE_BASE}/delta/loans').count():,} rows")

# COMMAND ----------

df_transactions = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(f"{BRONZE_BASE}/transactions/transactions.csv")
    .withColumn("_ingestion_timestamp", F.current_timestamp())
    .withColumn("_source_file", F.lit("transactions.csv"))
    .withColumn("_bronze_layer", F.lit("bronze"))
)

df_transactions.write.format("delta").mode("overwrite").save(f"{BRONZE_BASE}/delta/transactions")
print(f"transactions: {spark.read.format('delta').load(f'{BRONZE_BASE}/delta/transactions').count():,} rows")

# COMMAND ----------

for table in ["customers", "loans", "transactions"]:
    c = spark.read.format("delta").load(f"{BRONZE_BASE}/delta/{table}").count()
    print(f"  {table:<15}: {c:,}")
