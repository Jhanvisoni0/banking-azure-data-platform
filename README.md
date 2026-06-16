
# Retail Banking Customer Analytics Platform
### Azure · Databricks · PySpark · Delta Lake · ADF · MLflow · Power BI

---

## Business Problem

FirsTrust Bank had customer, loan, and transaction data spread across 3 separate source systems with no unified view. The risk team was spending 4 days every week manually reconciling Excel reports across systems before any analysis could happen.

This platform ingests, transforms, and serves all 3 data sources through a Medallion Architecture pipeline on Azure — reducing the reporting cycle from **4 days to 15 minutes** via automated Databricks Workflows.

---

## Architecture

<img width="1184" height="789" alt="architecture_diagram (1)" src="https://github.com/user-attachments/assets/bd3c4e16-9f61-437a-a11b-61c54cb83784" />


| Component | Purpose |
|---|---|
| **GitHub** | Hosts synthetic source CSVs simulating core banking, loan origination, and transaction ledger exports |
| **Azure Data Factory** | Ingests raw CSVs from GitHub into ADLS Gen2 Bronze container via scheduled Copy pipeline |
| **Azure ADLS Gen2** | Cloud data lake — Bronze (raw), Silver (clean), Gold (business-ready) layers |
| **Azure Key Vault** | Credential management — stores ADLS storage key, referenced by Databricks at runtime |
| **Azure Databricks** | PySpark transformation engine — runs Bronze → Silver → Gold → Risk Scoring pipeline |
| **Delta Lake** | Storage format — ACID transactions, schema enforcement, time travel, Delta MERGE |
| **MLflow** | Tracks Isolation Forest anomaly detection experiment — logs params, metrics, model artifact |
| **Databricks Workflows** | Orchestrates full pipeline on daily 6AM schedule with retry logic and failure alerting |
| **Power BI** | 3-page executive dashboard connected via DirectQuery to Databricks SQL warehouse |

---

## Data

Synthetic banking data generated using `faker` and `numpy` (seed=42) simulating 3 source systems:

| File | Rows | Description |
|---|---|---|
| `customers.csv` | 5,000 | Customer demographics, account balance, credit score, product type |
| `loans.csv` | 3,200 | Loan portfolio — type, amount, outstanding balance, delinquency status |
| `transactions.csv` | 50,000 | 3 years of transactions (2021-2023) with ~3% anomalous activity |

---

## Pipeline — Medallion Architecture

### Bronze Layer (`01_bronze_ingestion.py`)
- Reads raw CSVs from ADLS Gen2 bronze container (deposited by ADF)
- Adds metadata columns: `_ingestion_timestamp`, `_source_file`, `_bronze_layer`
- Writes as Delta tables — no cleaning, raw data preserved for full audit trail

### Silver Layer (`02_silver_transformation.py`)
- Schema enforcement — casts all columns to correct data types
- Deduplication on primary keys (`customer_id`, `loan_id`, `transaction_id`)
- Standardises `delinquency_status` to fixed value set, adds `is_delinquent` flag
- Extracts time features from transactions (hour, day of week, year, month)
- Transactions partitioned by `transaction_year` for query performance

### Gold Layer (`03_gold_customer360.py`)
- Joins all 3 Silver tables on `customer_id` → unified Customer 360 view (43 columns)
- Computes banking KPIs:
  - **Loan-to-deposit ratio** — outstanding loans / account balance
  - **30/60/90-day delinquency flags** — per customer across all loans
  - **Transaction velocity** — rolling 30-day transaction count via PySpark window functions
  - **Account dormancy** — no transaction in 90+ days
  - **Customer risk score** — weighted rule-based score combining delinquency tier, LDR, and velocity
- Transaction summary table partitioned by year/month for Power BI trends page
- Gold tables registered as managed Unity Catalog tables for SQL Warehouse access

### Risk Scoring (`04_risk_scoring.py`)
- Feature engineering: amount z-score (vs customer's own average), night transaction flag, 7-day velocity, log-transformed amount
- Unsupervised anomaly detection using **Isolation Forest** (`contamination=0.03`)
- MLflow experiment `/banking-risk-scoring` tracks: params, precision, recall, F1, flagged count
- **Result: 1,500 HIGH_RISK transactions flagged** out of 50,000 (3.0%)
- Scored results written back to Gold Delta table

---

## Orchestration

Databricks Workflow `banking_daily_pipeline` automates the full pipeline:
- **Schedule:** Daily at 6:00 AM (America/Chicago)
- **Retry:** 1 automatic retry on failure
- **Alerting:** Email notification on failure
- **Compute:** Auto-starts `banking-cluster` (Standard_D4s_v3, single node, 14.3 LTS) on run, terminates after completion

---

## Power BI Dashboard

3-page executive report connected via **DirectQuery** to Databricks SQL warehouse (`banking-sql`, 2X-Small Serverless):

| Page | Visuals |
|---|---|
| **Customer Overview** | Total customers, avg balance, avg credit score, total outstanding debt, product mix donut, segment bar chart, customers by state map |
| **Transaction Trends** | Daily volume line chart (2021-2023), category breakdown, channel split, delinquency rate |
| **Risk Flags** | High-risk count, delinquency tier donut, anomaly score distribution, flagged accounts table |

---

## Key Results

| Metric | Value |
|---|---|
| Source systems unified | 3 |
| Total customers | 5,000 |
| Total transactions | 50,000 |
| Reporting cycle | 4 days → 15 minutes |
| HIGH_RISK transactions flagged | 1,500 (3.0%) |
| MLflow experiment runs | Tracked with precision/recall/F1 |
| Pipeline automation | Daily scheduled Workflow |

---

## Tech Stack

```
Languages      : Python, SQL, PySpark
Cloud          : Microsoft Azure (ADLS Gen2, ADF, Key Vault, Databricks)
Data Format    : Delta Lake (ACID, time travel, schema evolution)
Orchestration  : Databricks Workflows
ML Tracking    : MLflow (Isolation Forest, experiment tracking)
BI             : Power BI (DirectQuery, DAX measures)
Version Control: Git / GitHub
```

---

## Project Structure

```
banking-azure-data-platform/
├── notebooks/
│   ├── 01_bronze_ingestion.py       # Raw CSV → Bronze Delta tables
│   ├── 02_silver_transformation.py  # Schema enforcement, deduplication, DQ
│   ├── 03_gold_customer360.py       # Customer 360, banking KPIs, SQL views
│   └── 04_risk_scoring.py           # Feature engineering, Isolation Forest, MLflow
├── data/
│   └── raw/
│       ├── customers.csv
│       ├── loans.csv
│       └── transactions.csv
├── dashboards/
│   ├── banking_dashboard.pbix
│   ├── Page1_CustomerOverview.png
│   ├── Page2_Transactions.png
│   └── Page3_RiskFlags.png
├── architecture/
│   └── architecture_diagram.png
└── README.md
```

---

## Author

**Jhanvi Soni** — Data Engineer | Financial Services
- LinkedIn: [linkedin.com/in/jhanvisoni](https://www.linkedin.com/in/jhanvisonii/)
- GitHub: [github.com/Jhanvisoni0](https://github.com/Jhanvisoni0)
