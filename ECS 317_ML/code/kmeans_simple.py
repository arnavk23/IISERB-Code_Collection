import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

# --- Load Data from CSV ---
iris_df = pd.read_csv('./iris.csv') # Give your datapath

# --- Drop ID and extract features + labels ---
X = iris_df.drop(['ID', 'Actual Cluster Label'], axis = 1).values
y,_ = pd.factorize(iris_df['Actual Cluster Label'])
feature_names = iris_df.drop(['Id', 'Actual Cluster Label'], axis = 1).columns

# --- Try different cluster numbers ---
max = 10
nmi_scores = []

for k in range(3,max):
  kmeans = KMeans(n_clusters=k, random_states=42, algorithm='lloyd') # Default
  labels = kmeans.fit_predict(X)
  nmi = normalized_mutual_info_score(y,labels)
  nmi_scores.append(nmi)

print('The NMI of K-means for different k values from 3 to 10 \n')

print(nmi_scores)
