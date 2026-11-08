# Week 4 Capstone: Customer Segmentation using K-Means

## Assignment objective
Use K-Means clustering to divide mall customers into groups and visualize them using Age, Annual Income and Spending Score. The three groups are named **Low Spenders**, **Medium Spenders** and **High Spenders** based on their average Spending Score.

## Project structure
```text
week4_customer_segmentation/
├── data/
│   └── Mall_Customers_demo.csv
├── src/
│   └── customer_segmentation.py
├── outputs/
│   ├── elbow_method.png
│   ├── income_vs_spending.png
│   ├── age_vs_spending.png
│   ├── segment_summary.csv
│   ├── segmented_customers.csv
│   └── results.txt
├── models/
│   └── customer_segmentation_model.joblib
├── requirements.txt
└── README.md
```

## Dataset note (important)
The assignment specifies Kaggle's **Mall Customers Dataset**. The ZIP includes a clearly labelled demonstration CSV so the project can run immediately without internet. To use the exact assignment dataset, download `Mall_Customers.csv` from Kaggle's Mall Customers Dataset and place it in the `data` folder with the exact filename `Mall_Customers.csv`. The script will automatically use that file instead of the demo file.

The included demo data is synthetic and must not be presented as the original Kaggle dataset.

## Setup in VS Code (Windows PowerShell)
Open this folder in VS Code, then run each command:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py src/customer_segmentation.py
```

To explicitly run the included demo dataset:
```powershell
py src/customer_segmentation.py --demo
```

## Methodology
1. Load the customer dataset.
2. Check required columns and convert the three numerical features to numeric values.
3. Remove rows with missing values in the selected features.
4. Standardize Age, Annual Income and Spending Score with `StandardScaler`.
5. Apply the Elbow Method for K values 1 through 10.
6. Train K-Means with K=3, as specified by the assignment.
7. Assign segment names by sorting clusters using their mean Spending Score.
8. Evaluate cluster separation with the Silhouette Score.
9. Save charts, CSV outputs and the trained model.

## Outputs
- `elbow_method.png`: helps inspect a suitable cluster count.
- `income_vs_spending.png`: shows customer groups by income and spending.
- `age_vs_spending.png`: shows customer groups by age and spending.
- `segment_summary.csv`: count and average profile of each segment.
- `segmented_customers.csv`: each customer with cluster number and readable segment.
- `results.txt`: run summary and silhouette score.
- `customer_segmentation_model.joblib`: saved K-Means model and scaler.

## Conclusion
K-Means groups customers with similar Age, Annual Income and Spending Score. The segment labels help a business understand different spending behaviours and plan targeted marketing strategies. The cluster numbers themselves have no natural order; readable labels are assigned using the average Spending Score.

## Presentation points
- **Why clustering?** It finds groups in data when the group labels are not already known.
- **Why scaling?** Age, income and spending score use different numeric ranges; scaling prevents a large-range feature from dominating distance calculations.
- **Why K=3?** The assignment specifically asks for three customer groups.
- **What does the Elbow Method do?** It plots inertia for different K values to help identify where adding more clusters gives diminishing improvement.
- **What is the Silhouette Score?** It measures how well-separated the clusters are; higher values generally indicate better separation.
