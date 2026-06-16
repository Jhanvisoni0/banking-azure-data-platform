# Databricks notebook source
# MAGIC %md
# MAGIC ## 03 - Gold: Customer 360 + Banking KPIs
# MAGIC Joins the 3 silver tables into a single customer-level table and
# MAGIC computes the KPIs the risk team actually cares about - loan to
# MAGIC deposit ratio, delinquency tier, transaction velocity, dormancy.
# MAGIC Also builds a transaction_summary table for the Power BI trends page.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

STORAGE_ACCOUNT = "bankinganalyticsdls"

# Same direct-key setup as the other notebooks for now.
storage_key = "<YOUR_KEY1_HERE>"

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    storage_key
)

SILVER_BASE = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net"
GOLD_BASE   = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net"

df_loans_silver = spark.read.format("delta").load(f"{SILVER_BASE}/delta/loans")
df_cust_silver  = spark.read.format("delta").load(f"{SILVER_BASE}/delta/customers")
df_txn_silver   = spark.read.format("delta").load(f"{SILVER_BASE}/delta/transactions")

print("Config loaded, silver tables read")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loan KPIs per customer
# MAGIC Rolling each customer's loans up into one row - total loans,
# MAGIC outstanding balance, and counts at each DPD bucket. This is what
# MAGIC feeds the delinquency_risk_tier further down.

# COMMAND ----------

df_loan_kpis = (
    df_loans_silver
    .groupBy("customer_id")
    .agg(
        F.count("loan_id").alias("total_loans"),
        F.sum("outstanding_balance").alias("total_outstanding_balance"),
        F.sum("loan_amount").alias("total_loan_amount"),
        F.avg("interest_rate").alias("avg_interest_rate"),
        F.sum(F.when(F.col("delinquency_status") == "30_DPD", 1).otherwise(0)).alias("loans_30_dpd"),
        F.sum(F.when(F.col("delinquency_status") == "60_DPD", 1).otherwise(0)).alias("loans_60_dpd"),
        F.sum(F.when(F.col("delinquency_status") == "90_DPD", 1).otherwise(0)).alias("loans_90_dpd"),
        F.max("is_delinquent").alias("is_delinquent"),
    )
)

print(f"loan kpis: {df_loan_kpis.count():,} customers with at least one loan")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transaction KPIs + velocity
# MAGIC Aggregates per customer, plus a rolling 30-day transaction count
# MAGIC using a window function - this becomes max/avg_30d_txn_velocity
# MAGIC and is one of the inputs to the risk score below.

# COMMAND ----------

df_txn_kpis = (
    df_txn_silver
    .groupBy("customer_id")
    .agg(
        F.count("transaction_id").alias("total_transactions"),
        F.sum("transaction_amount").alias("total_transaction_amount"),
        F.avg("transaction_amount").alias("avg_transaction_amount"),
        F.max("transaction_amount").alias("max_transaction_amount"),
        F.max("transaction_date").alias("last_transaction_date"),
        F.sum("is_flagged").alias("flagged_transaction_count"),
    )
)

# 30-day rolling window per customer, ordered by date
windowSpec = (
    Window.partitionBy("customer_id")
    .orderBy(F.col("transaction_date").cast("long"))
    .rangeBetween(-30 * 86400, 0)
)

df_txn_velocity = (
    df_txn_silver
    .withColumn("rolling_30d_txn_count", F.count("transaction_id").over(windowSpec))
    .groupBy("customer_id")
    .agg(
        F.max("rolling_30d_txn_count").alias("max_30d_txn_velocity"),
        F.avg("rolling_30d_txn_count").alias("avg_30d_txn_velocity"),
    )
)

df_txn_combined = df_txn_kpis.join(df_txn_velocity, on="customer_id", how="left")

print(f"transaction kpis: {df_txn_combined.count():,} customers")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build customer_360
# MAGIC Left join everything onto the customer table (not every customer
# MAGIC has a loan, so left join keeps them in with nulls/zeros).
# MAGIC
# MAGIC A few derived fields here:
# MAGIC - **loan_to_deposit_ratio** - outstanding loans vs account balance
# MAGIC - **delinquency_risk_tier** - HIGH/MEDIUM/LOW/CURRENT/NO_LOAN based on worst DPD bucket
# MAGIC - **is_dormant** - no transaction in the last 90 days
# MAGIC - **customer_risk_score** - simple weighted score combining the above, just
# MAGIC   to have something for the risk dashboard before the ML model runs in notebook 4

# COMMAND ----------

df_customer_360 = (
    df_cust_silver
    .join(df_loan_kpis, on="customer_id", how="left")
    .join(df_txn_combined, on="customer_id", how="left")
    .withColumn("loan_to_deposit_ratio",
        F.when(F.col("account_balance") > 0,
            F.round(F.col("total_outstanding_balance") / F.col("account_balance"), 4)
        ).otherwise(F.lit(None))
    )
    .withColumn("delinquency_risk_tier",
        F.when(F.col("loans_90_dpd") > 0, "HIGH")
         .when(F.col("loans_60_dpd") > 0, "MEDIUM")
         .when(F.col("loans_30_dpd") > 0, "LOW")
         .when(F.col("total_loans") > 0, "CURRENT")
         .otherwise("NO_LOAN")
    )
    .withColumn("days_since_last_txn", F.datediff(F.current_date(), F.col("last_transaction_date")))
    .withColumn("is_dormant", F.when(F.col("days_since_last_txn") > 90, 1).otherwise(0))
    .withColumn("customer_risk_score",
        (
            F.when(F.col("delinquency_risk_tier") == "HIGH",   40).otherwise(0) +
            F.when(F.col("delinquency_risk_tier") == "MEDIUM", 25).otherwise(0) +
            F.when(F.col("delinquency_risk_tier") == "LOW",    10).otherwise(0) +
            F.when(F.col("loan_to_deposit_ratio") > 2,         20).otherwise(0) +
            F.when(F.col("flagged_transaction_count") > 0,     20).otherwise(0) +
            F.when(F.col("is_dormant") == 1,                   10).otherwise(0)
        )
    )
    .withColumn("_gold_processed_at", F.current_timestamp())
    .withColumn("_gold_layer", F.lit("gold"))
    # customers with no loans/transactions get nulls from the left join -
    # fill those with 0 so downstream aggregations don't break
    .fillna(0, subset=[
        "total_loans", "total_outstanding_balance", "total_loan_amount",
        "loans_30_dpd", "loans_60_dpd", "loans_90_dpd", "is_delinquent",
        "total_transactions", "flagged_transaction_count", "customer_risk_score"
    ])
)

print(f"customer_360: {df_customer_360.count():,} rows | {len(df_customer_360.columns)} cols")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write gold tables
# MAGIC customer_360 partitioned by state (matches how the Power BI report
# MAGIC filters by region). transaction_summary is a separate rollup table
# MAGIC used for the trends page - partitioned by year/month.

# COMMAND ----------

(
    df_customer_360.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("state")
    .save(f"{GOLD_BASE}/delta/customer_360")
)

df_txn_summary = (
    df_txn_silver
    .groupBy(
        "transaction_date", "transaction_type", "channel", "transaction_category",
        F.year("transaction_date").alias("year"),
        F.month("transaction_date").alias("month"),
    )
    .agg(
        F.count("transaction_id").alias("transaction_count"),
        F.sum("transaction_amount").alias("total_amount"),
        F.avg("transaction_amount").alias("avg_amount"),
        F.sum("is_flagged").alias("flagged_count"),
    )
)

(
    df_txn_summary.write.format("delta").mode("overwrite")
    .partitionBy("year", "month")
    .save(f"{GOLD_BASE}/delta/transaction_summary")
)

c1 = spark.read.format("delta").load(f"{GOLD_BASE}/delta/customer_360").count()
c2 = spark.read.format("delta").load(f"{GOLD_BASE}/delta/transaction_summary").count()
print(f"gold customer_360       : {c1:,} rows")
print(f"gold transaction_summary: {c2:,} rows")
