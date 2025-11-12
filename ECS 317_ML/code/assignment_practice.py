import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import normalized_mutual_info_score

# --- Load Data from CSV ---
iris_df = pd.read_csv('./iris.csv') # Give your datapath

# --- Drop ID and extract features + labels ---
X = iris_df.drop(['ID', 'Actual Cluster Label'], axis = 1).values
y,_ = pd.factorize(iris_df['Actual Cluster Label'])
feature_names = iris_df.drop(['ID', 'Actual Cluster Label'], axis = 1).columns

# --- Try different cluster numbers with AgglomerativeClustering ---
max_k = 10
nmi_scores = []

for k in range(3, max_k):
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = agg.fit_predict(X)
    nmi = normalized_mutual_info_score(y, labels)
    nmi_scores.append(nmi)

print('The NMI of AgglomerativeClustering for different k values from 3 to 10 \n')
print(nmi_scores)

# --- Minimal DBSCAN example (single set of params) ---
db = DBSCAN(eps=0.5, min_samples=5)
db_labels = db.fit_predict(X)
nmi_db = normalized_mutual_info_score(y, db_labels)
print(f'DBSCAN (eps=0.5, min_samples=5) NMI: {nmi_db:.4f}')
