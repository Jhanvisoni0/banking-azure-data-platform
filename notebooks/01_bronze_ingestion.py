# Databricks notebook source
# MAGIC %md
# MAGIC ## 01 - Bronze Ingestion
# MAGIC Reads the raw CSVs (landed by the ADF pipeline) from the bronze
# MAGIC container and writes them as Delta tables. No cleaning here -
# MAGIC bronze stays as close to source as possible so we always have a
# MAGIC raw copy to fall back on if something downstream goes wrong.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *

STORAGE_ACCOUNT = "bankinganalyticsdls"

# Storage key - in the target architecture this comes from
# dbutils.secrets.get(scope="banking-kv", key="adls-storage-key")
# via a Key Vault-backed secret scope. Using a direct key for now
# because of the cross-tenant secret scope issue (personal Databricks
# workspace vs university Key Vault). Swap this back once resolved.
storage_key = "<YOUR_KEY1_HERE>"

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    storage_key
)

BRONZE_BASE = f"abfss://bronze@{STORAGE_ACCOUNT}.dfs.core.windows.net"
SILVER_BASE = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net"
GOLD_BASE   = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net"

print("Config loaded")

# COMMAND ----------

# Customers - core banking export
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

count = spark.read.format("delta").load(f"{BRONZE_BASE}/delta/customers").count()
print(f"customers: {count:,} rows written to bronze")

# COMMAND ----------

# Loans - loan origination system export
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

count = spark.read.format("delta").load(f"{BRONZE_BASE}/delta/loans").count()
print(f"loans: {count:,} rows written to bronze")

# COMMAND ----------

# Transactions - ledger export, largest of the 3 files
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

count = spark.read.format("delta").load(f"{BRONZE_BASE}/delta/transactions").count()
print(f"transactions: {count:,} rows written to bronze")

# COMMAND ----------

# Quick sanity check before moving on to silver
print("Bronze layer summary")
for table in ["customers", "loans", "transactions"]:
    c = spark.read.format("delta").load(f"{BRONZE_BASE}/delta/{table}").count()
    print(f"  {table:<15}: {c:,}")
