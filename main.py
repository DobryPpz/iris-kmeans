import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

datafile = "iris.csv"

data = np.genfromtxt(
    datafile,
    delimiter = ",",
    usecols = range(0,4),
    skip_header = 0,
    dtype = "float"
)

true_labels = np.genfromtxt(
    datafile,
    delimiter = ",",
    usecols = range(4,5),
    skip_header = 0,
    dtype = "int"
)

print(data)
print(true_labels)

preprocessor = Pipeline(
    [
        ("scaler",MinMaxScaler()),
        ("pca",PCA(n_components = 2,random_state=42))
    ]
)

clusterer = Pipeline(
    [
        (
            "kmeans",
            KMeans(
                n_clusters = 3,
                init = "k-means++",
                n_init = 50,
                max_iter = 500,
                random_state = 42
            )
        )
    ]
)

pipe = Pipeline(
    [
        ("preprocessor",preprocessor),
        ("clusterer",clusterer)
    ]
)

pipe.fit(data)

pcadf = pd.DataFrame(
    pipe["preprocessor"].transform(data),
    columns=["component_1","component_2"]
)

pcadf["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_
plt.style.use("fivethirtyeight")
plt.figure(figsize=(8, 8))
scat = sns.scatterplot(
    "component_1",
    "component_2",
    s = 50,
    data=pcadf,
    hue="predicted_cluster",
    palette="Set2"
)


plt.show()
