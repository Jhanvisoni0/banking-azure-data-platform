# Databricks notebook source
# MAGIC %md
# MAGIC ## 04 - Risk Scoring (Isolation Forest + MLflow)
# MAGIC Engineers a few features per transaction (z-score vs customer's
# MAGIC own average, night/weekend flags, 7-day velocity), runs Isolation
# MAGIC Forest to flag anomalous transactions, logs the run to MLflow, and
# MAGIC writes the scored output back to gold.
# MAGIC
# MAGIC Note: this runs as a batch job, not a live scoring endpoint - scores
# MAGIC get refreshed each time the pipeline runs.

# COMMAND ----------

# MAGIC %pip install mlflow scikit-learn

# COMMAND ----------

# mlflow/sklearn aren't on the base runtime image, so the pip install
# above needs a Python restart before the imports below will resolve.
dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

STORAGE_ACCOUNT = "bankinganalyticsdls"

# restartPython() wipes all variables, so config needs to be redone here
storage_key = "<YOUR_KEY1_HERE>"

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    storage_key
)

SILVER_BASE = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net"
GOLD_BASE   = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net"

print("Config reloaded after restart")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature engineering
# MAGIC - **amount_zscore** - how far this transaction is from the customer's
# MAGIC   own average, in std devs. A $5k transaction is normal for some
# MAGIC   customers and very unusual for others, so this is relative not absolute.
# MAGIC - **is_night_txn** - midnight to 4am, a common fraud window
# MAGIC - **velocity_7d** - rolling 7-day transaction count, same idea as the
# MAGIC   30-day version in notebook 3 but tighter window for the ML features

# COMMAND ----------

df_txn = spark.read.format("delta").load(f"{SILVER_BASE}/delta/transactions")

windowCust = Window.partitionBy("customer_id")

df_features = (
    df_txn
    .withColumn("cust_avg_amount", F.avg("transaction_amount").over(windowCust))
    .withColumn("cust_std_amount", F.stddev("transaction_amount").over(windowCust))
    .withColumn("amount_zscore",
        F.when(F.col("cust_std_amount") > 0,
            F.abs((F.col("transaction_amount") - F.col("cust_avg_amount")) / F.col("cust_std_amount"))
        ).otherwise(0.0)
    )
    .withColumn("is_night_txn", F.when(F.col("transaction_hour").between(0, 4), 1).otherwise(0))
    .withColumn("is_weekend", F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0))
    .withColumn("velocity_7d",
        F.count("transaction_id").over(
            Window.partitionBy("customer_id")
            .orderBy(F.col("transaction_date").cast("long"))
            .rangeBetween(-7 * 86400, 0)
        )
    )
    .withColumn("log_amount", F.log1p("transaction_amount"))
    .select(
        "transaction_id", "customer_id", "transaction_amount",
        "transaction_hour", "is_flagged", "amount_zscore",
        "is_night_txn", "is_weekend", "velocity_7d", "log_amount"
    )
    .fillna(0)
)

print(f"features built for {df_features.count():,} transactions")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Isolation Forest
# MAGIC contamination=0.03 because the synthetic data was generated with
# MAGIC ~3% anomalous transactions - in a real setup this would be tuned
# MAGIC against historical fraud rates rather than hardcoded.
# MAGIC
# MAGIC is_flagged from the source data is used here purely to compute
# MAGIC precision/recall for the MLflow run - the model itself is unsupervised
# MAGIC and doesn't see this column during training.

# COMMAND ----------

feature_cols = ["amount_zscore", "is_night_txn", "is_weekend", "velocity_7d", "log_amount"]

df_pd = df_features.select(["transaction_id"] + feature_cols + ["is_flagged"]).toPandas()

X = df_pd[feature_cols].values
y_true = df_pd["is_flagged"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

mlflow.set_experiment("/banking-risk-scoring")

with mlflow.start_run(run_name="isolation_forest_v1"):
    params = {"n_estimators": 100, "contamination": 0.03, "random_state": 42}
    mlflow.log_params(params)

    model = IsolationForest(**params)
    model.fit(X_scaled)

    predictions = model.predict(X_scaled)          # -1 = anomaly, 1 = normal
    anomaly_scores = model.score_samples(X_scaled) # lower = more anomalous
    predicted_flags = (predictions == -1).astype(int)

    precision = precision_score(y_true, predicted_flags, zero_division=0)
    recall    = recall_score(y_true, predicted_flags, zero_division=0)
    f1        = f1_score(y_true, predicted_flags, zero_division=0)
    flagged_count = int(predicted_flags.sum())

    mlflow.log_metrics({
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "flagged_count": flagged_count,
        "total_records": len(predictions),
        "flag_rate": round(flagged_count / len(predictions), 4),
    })
    mlflow.sklearn.log_model(model, "isolation_forest_model")

    print(f"precision: {precision:.4f}  recall: {recall:.4f}  f1: {f1:.4f}")
    print(f"flagged: {flagged_count:,} / {len(predictions):,} ({flagged_count/len(predictions)*100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write scores back to gold
# MAGIC Joins the predictions back onto the feature set and writes the
# MAGIC result as transaction_risk_scores. Power BI's risk page reads
# MAGIC from this table.

# COMMAND ----------

df_pd["anomaly_prediction"] = predicted_flags
df_pd["anomaly_score"] = anomaly_scores
df_pd["risk_label"] = df_pd["anomaly_prediction"].map({1: "HIGH_RISK", 0: "NORMAL"})

df_scores_spark = spark.createDataFrame(
    df_pd[["transaction_id", "anomaly_prediction", "anomaly_score", "risk_label"]]
)

df_risk_output = (
    df_features
    .join(df_scores_spark, on="transaction_id", how="left")
    .withColumn("_scored_at", F.current_timestamp())
    .withColumn("_model_version", F.lit("isolation_forest_v1"))
)

(
    df_risk_output.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{GOLD_BASE}/delta/transaction_risk_scores")
)

count = spark.read.format("delta").load(f"{GOLD_BASE}/delta/transaction_risk_scores").count()
print(f"transaction_risk_scores: {count:,} rows written")

df_risk_output.groupBy("risk_label").count().show()
