# Databricks notebook source
# MAGIC %md
# MAGIC ## 02 - Silver Transformation
# MAGIC Cleans up the bronze tables - casts types, standardises a few fields,
# MAGIC drops duplicates, and adds a couple of derived flags that the gold
# MAGIC layer will need (delinquency flag, partitioning by year for txns).

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *

STORAGE_ACCOUNT = "bankinganalyticsdls"

# Same note as notebook 1 - using direct key for now, secret scope
# is the production-intended setup once the Key Vault access issue
# across tenants is sorted.
storage_key = "<YOUR_KEY1_HERE>"

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    storage_key
)

BRONZE_BASE = f"abfss://bronze@{STORAGE_ACCOUNT}.dfs.core.windows.net"
SILVER_BASE = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net"

print("Config loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Customers
# MAGIC Mostly type casting here. Credit score had a few nulls in the
# MAGIC source data so we're backfilling those with 690 (roughly the
# MAGIC median) rather than dropping the row.

# COMMAND ----------

df_silver_cust = (
    spark.read.format("delta").load(f"{BRONZE_BASE}/delta/customers")
    .withColumn("customer_id",     F.col("customer_id").cast(StringType()))
    .withColumn("age",             F.col("age").cast(IntegerType()))
    .withColumn("annual_income",   F.col("annual_income").cast(LongType()))
    .withColumn("credit_score",    F.col("credit_score").cast(IntegerType()))
    .withColumn("account_balance", F.col("account_balance").cast(DoubleType()))
    .withColumn("is_active",       F.col("is_active").cast(IntegerType()))
    .withColumn("onboard_date",    F.to_date(F.col("onboard_date"), "yyyy-MM-dd"))
    .withColumn("credit_score",
        F.when(F.col("credit_score").isNull(), 690).otherwise(F.col("credit_score"))
    )
    .dropDuplicates(["customer_id"])
    .drop("_bronze_layer")
    .withColumn("_silver_processed_at", F.current_timestamp())
    .withColumn("_silver_layer", F.lit("silver"))
)

df_silver_cust.write.format("delta").mode("overwrite").save(f"{SILVER_BASE}/delta/customers")

count = spark.read.format("delta").load(f"{SILVER_BASE}/delta/customers").count()
print(f"silver customers: {count:,} rows | {len(df_silver_cust.columns)} cols")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loans
# MAGIC Standardising delinquency_status to a fixed set of values and
# MAGIC adding is_delinquent as a simple 0/1 flag - makes the gold layer
# MAGIC KPI logic much easier to read later.

# COMMAND ----------

df_silver_loan = (
    spark.read.format("delta").load(f"{BRONZE_BASE}/delta/loans")
    .withColumn("loan_id",             F.col("loan_id").cast(StringType()))
    .withColumn("customer_id",         F.col("customer_id").cast(StringType()))
    .withColumn("loan_amount",         F.col("loan_amount").cast(DoubleType()))
    .withColumn("outstanding_balance", F.col("outstanding_balance").cast(DoubleType()))
    .withColumn("interest_rate",       F.col("interest_rate").cast(DoubleType()))
    .withColumn("term_months",         F.col("term_months").cast(IntegerType()))
    .withColumn("monthly_payment",     F.col("monthly_payment").cast(DoubleType()))
    .withColumn("origination_date",    F.to_date(F.col("origination_date"), "yyyy-MM-dd"))
    .withColumn("delinquency_status",
        F.when(F.col("delinquency_status").isin(
            ["CURRENT", "30_DPD", "60_DPD", "90_DPD", "CLOSED", "CHARGED_OFF"]
        ), F.col("delinquency_status")).otherwise("UNKNOWN")
    )
    .withColumn("is_delinquent",
        F.when(F.col("delinquency_status").isin(
            ["30_DPD", "60_DPD", "90_DPD", "CHARGED_OFF"]
        ), 1).otherwise(0)
    )
    .dropDuplicates(["loan_id"])
    .drop("_bronze_layer")
    .withColumn("_silver_processed_at", F.current_timestamp())
    .withColumn("_silver_layer", F.lit("silver"))
)

df_silver_loan.write.format("delta").mode("overwrite").save(f"{SILVER_BASE}/delta/loans")

count = spark.read.format("delta").load(f"{SILVER_BASE}/delta/loans").count()
print(f"silver loans: {count:,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transactions
# MAGIC Largest table by far (50k rows), so this is partitioned by
# MAGIC transaction_year. Also pulling out hour/day-of-week here since
# MAGIC the risk scoring notebook needs them as features later.

# COMMAND ----------

df_silver_txn = (
    spark.read.format("delta").load(f"{BRONZE_BASE}/delta/transactions")
    .withColumn("transaction_id", F.col("transaction_id").cast(StringType()))
    .withColumn("customer_id", F.col("customer_id").cast(StringType()))
    .withColumn("transaction_date", F.to_date(F.col("transaction_date"), "yyyy-MM-dd"))
    .withColumn("transaction_datetime", F.to_timestamp(F.col("transaction_datetime"), "yyyy-MM-dd HH:mm:ss"))
    .withColumn("transaction_amount", F.col("transaction_amount").cast(DoubleType()))
    .withColumn("is_flagged", F.col("is_flagged").cast(IntegerType()))
    .withColumn("transaction_year", F.year(F.col("transaction_date")))
    .withColumn("transaction_month", F.month(F.col("transaction_date")))
    .withColumn("transaction_hour", F.hour(F.col("transaction_datetime")))
    .withColumn("day_of_week", F.dayofweek(F.col("transaction_date")))
    .dropDuplicates(["transaction_id"])
    .filter(F.col("transaction_amount") > 0)
    .drop("_bronze_layer")
    .withColumn("_silver_processed_at", F.current_timestamp())
    .withColumn("_silver_layer", F.lit("silver"))
)

(
    df_silver_txn.write.format("delta").mode("overwrite")
    .partitionBy("transaction_year")
    .save(f"{SILVER_BASE}/delta/transactions")
)

count = spark.read.format("delta").load(f"{SILVER_BASE}/delta/transactions").count()
print(f"silver transactions: {count:,} rows")

# COMMAND ----------

# Quick check - row counts should match bronze (we're not dropping
# rows here other than the rare $0 transaction)
print("Silver layer summary")
for table in ["customers", "loans", "transactions"]:
    c = spark.read.format("delta").load(f"{SILVER_BASE}/delta/{table}").count()
    print(f"  {table:<15}: {c:,}")
