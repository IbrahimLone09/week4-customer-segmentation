"""
Week 4 Capstone: Customer Segmentation using K-Means
Dataset columns expected:
Gender, Age, Annual Income (k$), Spending Score (1-100)
"""
from pathlib import Path
import argparse
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
DATA_FILE = DATA_DIR / "Mall_Customers.csv"
DEMO_FILE = DATA_DIR / "Mall_Customers_demo.csv"

FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

def load_data(use_demo=False):
    path = DEMO_FILE if use_demo else DATA_FILE
    if not path.exists():
        if DEMO_FILE.exists():
            print("Mall_Customers.csv was not found.")
            print("Running the included DEMO dataset instead.")
            print("For the assignment dataset, download Kaggle's Mall Customers CSV")
            print("and save it as: data/Mall_Customers.csv")
            path = DEMO_FILE
        else:
            raise FileNotFoundError(f"Dataset not found: {path}")
    data = pd.read_csv(path)
    # Standardize common alternative column names from the Kaggle file.
    rename = {
        "Annual Income": "Annual Income (k$)",
        "Spending Score": "Spending Score (1-100)",
        "Spending Score (1-100)": "Spending Score (1-100)",
    }
    data = data.rename(columns=rename)
    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Check the CSV headers.")
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=FEATURES).copy()
    if "Gender" not in data.columns:
        data["Gender"] = "Unknown"
    return data

def label_segments(data):
    # Give human-readable labels based on average spending score.
    means = data.groupby("Cluster")["Spending Score (1-100)"].mean().sort_values()
    ordered = list(means.index)
    labels = {ordered[0]: "Low Spenders"}
    if len(ordered) == 3:
        labels[ordered[1]] = "Medium Spenders"
        labels[ordered[2]] = "High Spenders"
    else:
        for rank, cluster in enumerate(ordered):
            labels[cluster] = ["Low Spenders", "Medium Spenders", "High Spenders"][min(rank, 2)]
    return data["Cluster"].map(labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Use included demo data")
    args = parser.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    data = load_data(use_demo=args.demo)
    X = data[FEATURES].copy()

    # Scale features so Age, Income and Spending Score contribute comparably.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method: compare within-cluster sum of squares for K=1..10.
    inertias = []
    for k in range(1, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), inertias, marker="o")
    plt.title("Elbow Method for Choosing K")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "elbow_method.png", dpi=160)
    plt.close()

    # The assignment asks for three customer groups.
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    data["Cluster"] = kmeans.fit_predict(X_scaled)
    data["Segment"] = label_segments(data)

    score = silhouette_score(X_scaled, data["Cluster"]) if len(data) > 3 else float("nan")
    summary = data.groupby("Segment").agg(
        Customers=("Segment", "size"),
        Average_Age=("Age", "mean"),
        Average_Annual_Income=("Annual Income (k$)", "mean"),
        Average_Spending_Score=("Spending Score (1-100)", "mean")
    ).round(2)
    summary.to_csv(OUT_DIR / "segment_summary.csv")
    data.to_csv(OUT_DIR / "segmented_customers.csv", index=False)

    # Income vs spending score visualization.
    plt.figure(figsize=(9, 6))
    for segment, group in data.groupby("Segment"):
        plt.scatter(group["Annual Income (k$)"], group["Spending Score (1-100)"],
                    label=segment, alpha=0.75, s=45)
    plt.title("Customer Segments: Annual Income vs Spending Score")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "income_vs_spending.png", dpi=160)
    plt.close()

    # Age vs spending score visualization.
    plt.figure(figsize=(9, 6))
    for segment, group in data.groupby("Segment"):
        plt.scatter(group["Age"], group["Spending Score (1-100)"],
                    label=segment, alpha=0.75, s=45)
    plt.title("Customer Segments: Age vs Spending Score")
    plt.xlabel("Age")
    plt.ylabel("Spending Score (1-100)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "age_vs_spending.png", dpi=160)
    plt.close()

    joblib.dump({"model": kmeans, "scaler": scaler, "features": FEATURES},
                MODEL_DIR / "customer_segmentation_model.joblib")

    report = [
        "WEEK 4 - CUSTOMER SEGMENTATION RESULTS",
        f"Rows used: {len(data)}",
        "Algorithm: K-Means",
        "Number of clusters: 3",
        f"Silhouette score: {score:.4f}",
        "",
        "SEGMENT SUMMARY",
        summary.to_string(),
        "",
        "Note: cluster numbers are arbitrary; the segment names are assigned",
        "by sorting clusters according to their average Spending Score."
    ]
    (OUT_DIR / "results.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print("\nCharts, CSV results and trained model saved successfully.")

if __name__ == "__main__":
    main()
