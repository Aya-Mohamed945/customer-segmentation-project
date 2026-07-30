````markdown

\# 📊 Methodology



\## 1. Introduction



This document outlines the technical methodology used to segment customers of an FMCG store using unsupervised machine learning. The goal is to identify distinct customer groups based on demographic and purchasing behavior data.



\---



\## 2. Dataset Description



\- \*\*Source:\*\* Loyalty card data from an FMCG store.

\- \*\*Samples:\*\* 2,000 customers.

\- \*\*Features:\*\* 7 attributes (Sex, Marital Status, Age, Education, Income, Occupation, Settlement Size).

\- \*\*Quality:\*\* No missing values and no duplicate records.



\---



\## 3. Data Preprocessing



\### 3.1 Data Cleaning



\- Dropped the \*\*ID\*\* column because it is not relevant for clustering analysis.

\- Checked for duplicate records and confirmed that no duplicate rows exist.



\---



\### 3.2 Feature Encoding



All categorical features were already numerically encoded:



\- \*\*Sex\*\*

&#x20; - 0 = Male

&#x20; - 1 = Female



\- \*\*Marital Status\*\*

&#x20; - 0 = Single

&#x20; - 1 = Non-single



\- \*\*Education\*\*

&#x20; - 0 = Other

&#x20; - 1 = High School

&#x20; - 2 = University

&#x20; - 3 = Graduate School



\- \*\*Occupation\*\*

&#x20; - 0 = Unemployed

&#x20; - 1 = Skilled Employee

&#x20; - 2 = Management



\- \*\*Settlement Size\*\*

&#x20; - 0 = Small City

&#x20; - 1 = Mid-sized City

&#x20; - 2 = Big City



\---



\### 3.3 Feature Scaling (Standardization)



The dataset was standardized using \*\*StandardScaler\*\* from Scikit-learn.



```python

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X\_scaled = scaler.fit\_transform(df\_clean)

````



\*\*Why Standardization?\*\*



The features have different numerical scales.



For example:



\* \*\*Income:\*\* $35,832 – $309,364

\* \*\*Age:\*\* 18 – 76



Standardization ensures that all features contribute equally to the distance calculations used by clustering algorithms.



\---



\### 3.4 Dimensionality Reduction (PCA)



Principal Component Analysis (PCA) was applied for visualization and dimensionality reduction.



\*\*Results\*\*



\* First \*\*2 principal components\*\* explain \*\*61.95%\*\* of the total variance.

\* First \*\*5 principal components\*\* explain \*\*94.04%\*\* of the total variance.



\*\*Recommendation\*\*



\* Use \*\*5 principal components\*\* for clustering.

\* Use \*\*2 principal components\*\* for visualization.



\---



\# 4. Clustering Algorithms



\## 4.1 K-Means Clustering



\*\*Why K-Means?\*\*



\* Simple and computationally efficient.

\* Easy to interpret.

\* Performs well on standardized numerical data.



\*\*Optimal Number of Clusters\*\*



The optimal number of clusters was determined using:



\* Elbow Method

\* Silhouette Score

\* Calinski-Harabasz Index

\* Davies-Bouldin Index



\*\*Result\*\*



\* Optimal number of clusters: \*\*9\*\*

\* Best Silhouette Score: \*\*0.3042\*\*



\---



\## 4.2 Hierarchical Clustering



\*\*Why Hierarchical Clustering?\*\*



\* Does not require selecting the number of clusters in advance.

\* Produces a dendrogram for cluster visualization.



\*\*Configuration\*\*



\* Linkage Method: \*\*Ward\*\*

\* Distance Metric: \*\*Euclidean\*\*



\*\*Result\*\*



\* Silhouette Score: \*\*0.3016\*\*



The performance was very close to that of K-Means.



\---



\## 4.3 DBSCAN Clustering



\*\*Why DBSCAN?\*\*



\* Detects clusters of arbitrary shapes.

\* Robust to outliers and noisy observations.



\### Parameter Tuning



\*\*EPS values tested\*\*



\* 1.5

\* 1.8

\* 2.0

\* 2.2

\* 2.5



\*\*Min Samples tested\*\*



\* 3

\* 5

\* 7

\* 10



\*\*Best Parameters\*\*



\* \*\*eps = 1.8\*\*

\* \*\*min\_samples = 10\*\*



\*\*Result\*\*



\* Silhouette Score: \*\*0.1792\*\*

\* Number of clusters: \*\*4\*\*

\* Noise points detected: \*\*21\*\*



\---



\# 5. Evaluation Metrics



The clustering algorithms were evaluated using the following metrics:



\* \*\*Silhouette Score\*\*



&#x20; \* Measures cluster cohesion and separation.

&#x20; \* Higher values indicate better clustering performance.



\* \*\*Calinski-Harabasz Index\*\*



&#x20; \* Measures the ratio between inter-cluster and intra-cluster variance.

&#x20; \* Higher values indicate better-defined clusters.



\* \*\*Davies-Bouldin Index\*\*



&#x20; \* Measures the average similarity between clusters.

&#x20; \* Lower values indicate better clustering quality.



\---



\# 6. Tools \& Libraries



| Library      | Purpose                         |

| ------------ | ------------------------------- |

| Pandas       | Data manipulation               |

| NumPy        | Numerical operations            |

| Matplotlib   | Data visualization              |

| Seaborn      | Statistical visualization       |

| Scikit-learn | Machine learning algorithms     |

| SciPy        | Hierarchical clustering support |



```

```



