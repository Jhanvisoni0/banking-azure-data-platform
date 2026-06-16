# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Risk Scoring
# MAGIC Feature engineering + Isolation Forest anomaly detection. Scores written to gold.

# COMMAND ----------

# MAGIC %pip install mlflow scikit-learn

# COMMAND ----------

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
storage_key     = "<YOUR_KEY1_HERE>"

spark.conf.set(
    f"fs.azure.account.key.{STORAGE_ACCOUNT}.dfs.core.windows.net",
    storage_key
)

SILVER_BASE = f"abfss://silver@{STORAGE_ACCOUNT}.dfs.core.windows.net"
GOLD_BASE   = f"abfss://gold@{STORAGE_ACCOUNT}.dfs.core.windows.net"

print("config reloaded")

# COMMAND ----------

df_txn     = spark.read.format("delta").load(f"{SILVER_BASE}/delta/transactions")
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
    .withColumn("is_weekend",   F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0))
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

print(f"features: {df_features.count():,} rows")

# COMMAND ----------

feature_cols = ["amount_zscore", "is_night_txn", "is_weekend", "velocity_7d", "log_amount"]
df_pd        = df_features.select(["transaction_id"] + feature_cols + ["is_flagged"]).toPandas()

X      = df_pd[feature_cols].values
y_true = df_pd["is_flagged"].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

mlflow.set_experiment("/banking-risk-scoring")

with mlflow.start_run(run_name="isolation_forest_v1"):
    params = {"n_estimators": 100, "contamination": 0.03, "random_state": 42}
    mlflow.log_params(params)

    model           = IsolationForest(**params)
    model.fit(X_scaled)
    predictions     = model.predict(X_scaled)
    anomaly_scores  = model.score_samples(X_scaled)
    predicted_flags = (predictions == -1).astype(int)

    precision     = precision_score(y_true, predicted_flags, zero_division=0)
    recall        = recall_score(y_true, predicted_flags, zero_division=0)
    f1            = f1_score(y_true, predicted_flags, zero_division=0)
    flagged_count = int(predicted_flags.sum())

    mlflow.log_metrics({
        "precision"     : round(precision, 4),
        "recall"        : round(recall, 4),
        "f1_score"      : round(f1, 4),
        "flagged_count" : flagged_count,
        "total_records" : len(predictions),
        "flag_rate"     : round(flagged_count / len(predictions), 4),
    })
    mlflow.sklearn.log_model(model, "isolation_forest_model")

    print(f"precision: {precision:.4f}  recall: {recall:.4f}  f1: {f1:.4f}")
    print(f"flagged: {flagged_count:,} / {len(predictions):,} ({flagged_count/len(predictions)*100:.1f}%)")

# COMMAND ----------

df_pd["anomaly_prediction"] = predicted_flags
df_pd["anomaly_score"]      = anomaly_scores
df_pd["risk_label"]         = df_pd["anomaly_prediction"].map({1: "HIGH_RISK", 0: "NORMAL"})

df_scores = spark.createDataFrame(
    df_pd[["transaction_id", "anomaly_prediction", "anomaly_score", "risk_label"]]
)

df_risk_output = (
    df_features
    .join(df_scores, on="transaction_id", how="left")
    .withColumn("_scored_at",      F.current_timestamp())
    .withColumn("_model_version",  F.lit("isolation_forest_v1"))
)

(
    df_risk_output.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true")
    .save(f"{GOLD_BASE}/delta/transaction_risk_scores")
)

count = spark.read.format("delta").load(f"{GOLD_BASE}/delta/transaction_risk_scores").count()
print(f"risk scores written: {count:,} rows")
df_risk_output.groupBy("risk_label").count().show()
