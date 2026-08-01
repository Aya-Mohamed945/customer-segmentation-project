# 🎯 Customer Segmentation Using Unsupervised Learning



> **A comprehensive machine learning pipeline for customer segmentation using unsupervised learning techniques.**



[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)

[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Clustering-orange?logo=scikitlearn)](https://scikit-learn.org/)

[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)](https://pandas.pydata.org/)

[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)](https://numpy.org/)

[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue)](https://matplotlib.org/)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)



---



## 📌 Project Overview



Customer segmentation is one of the most valuable applications of unsupervised learning. Instead of predicting predefined labels, clustering algorithms automatically discover hidden customer groups based on similarities in demographic characteristics.



This project analyzes customer data collected from an FMCG (Fast-Moving Consumer Goods) retailer to identify meaningful customer segments using multiple clustering techniques. The resulting segments can support targeted marketing campaigns, customer relationship management, and business decision-making.



---



## 🎯 Objectives



- Perform exploratory data analysis (EDA).

- Preprocess customer data for clustering.

- Apply multiple clustering algorithms.

- Determine the optimal number of clusters.

- Compare clustering performance using evaluation metrics.

- Visualize customer segments.

- Generate business-oriented insights.



---



## 📊 Dataset



The dataset contains information about \*\*2,000 customers\*\* collected through loyalty cards from an FMCG retail store.



To protect customer privacy:



- Personal information has been anonymized.

- Missing values have already been handled.

- Basic preprocessing was performed before publishing the dataset.



### Dataset Features



| Feature | Description |

|----------|-------------|

| ID | Unique customer identifier (excluded from clustering) |

| Sex | Male / Female |

| Marital Status | Single / Non-single |

| Age | Customer age |

| Education | Education level |

| Income | Annual income (USD) |

| Occupation | Occupation category |

| Settlement Size | City size |



---



## 📈 Exploratory Data Analysis



The exploratory analysis focused on understanding customer demographics before applying clustering.



Main analyses included:



- Distribution of categorical variables

- Distribution of numerical variables

- Correlation analysis

- Outlier inspection

- Feature scaling preparation



---



# 🛠️ Technologies Used



| Category | Tools |

|-----------|-------|

| Programming Language | Python |

| Data Analysis | Pandas, NumPy |

| Visualization | Matplotlib, Seaborn |

| Machine Learning | Scikit-Learn |

| Notebook | Jupyter Notebook |



---



## 🤖 Machine Learning Techniques



### Data Preprocessing



- StandardScaler

- Principal Component Analysis (PCA)



### Clustering Algorithms



- K-Means

- Agglomerative Hierarchical Clustering

- DBSCAN



### Evaluation Metrics



- Silhouette Score

- Calinski-Harabasz Index

- Davies-Bouldin Index



---



# 📂 Project Structure



```text

customer-segmentation-project/
│
├── data/
│   └── segmentation_data.csv          # Dataset (2,000 customers)
│
├── scripts/
│   ├── 01_data_exploration.py         # EDA and visualizations
│   ├── 02_data_preprocessing.py       # Standardization & PCA
│   ├── 03_clustering_kmeans.py        # K-Means implementation
│   ├── 04_clustering_hierarchical.py  # Hierarchical clustering
│   ├── 05_clustering_dbscan.py        # DBSCAN implementation
│   ├── 06_evaluation.py               # Model comparison module
│   ├── 07_visualization.py            # Results visualization
│   └── 08_save_reports.py             # Save reports to CSV
│
├── results/
│   ├── plots/                         # Generated visualizations
│   │   ├── age_income_distribution.png
│   │   ├── categorical_distribution.png
│   │   ├── correlation_matrix.png
│   │   ├── dbscan_k_distance.png
│   │   ├── dendrogram.png
│   │   ├── kmeans_clusters.png
│   │   └── pca_variance.png
│   └── reports/                       # Generated reports
│       ├── clustered_data.csv
│       ├── cluster_profiles.csv
│       ├── cluster_counts.csv
│       └── cluster_statistics.csv
│
├── docs/
│   ├── project_report.pdf             # Final project report
│   ├── methodology.md                 # Technical methodology
│   ├── results_summary.md             # Results summary
│   └── business_insights.md           # Business recommendations
│
├── notebooks/
│   └── clustering_analysis.ipynb      # Jupyter Notebook (optional)
│
├── README.md                          # Project overview
├── requirements.txt                   # Dependencies
├── .gitignore                         # Git ignore file
├── run.py                       	# Run all scripts
└── LICENSE                            # MIT License
```

# ⚙️ Installation



## 1. Clone the Repository



```bash

git clone https://github.com/your-username/customer-segmentation.git

cd customer-segmentation

```



---



## 2. Create a Virtual Environment (Optional)



### Windows



```bash

python -m venv venv

venv\\Scripts\\activate

```



### Linux / macOS



```bash

python3 -m venv venv

source venv/bin/activate

```



---



## 3. Install Dependencies



```bash

pip install -r requirements.txt

```



---



## 4. Launch Jupyter Notebook



```bash

jupyter notebook

```



Open the notebook:



```

Customer_Segmentation.ipynb

```



---



# 🚀 Project Workflow



The project follows a structured machine learning workflow:



### 1. Data Exploration



- Dataset overview

- Feature analysis

- Statistical summary

- Missing value verification

- Correlation analysis



---



### 2. Data Preprocessing



- Remove unnecessary columns

- Standardize numerical features

- Prepare data for clustering



---



### 3. Clustering



Three clustering algorithms were implemented and compared:



### K-Means



- Elbow Method

- Silhouette Score

- Cluster interpretation



### Agglomerative Hierarchical Clustering



- Dendrogram analysis

- Cluster comparison



### DBSCAN



- Density-based clustering

- Noise point detection



---



### 4. Model Evaluation



Different clustering metrics were used to compare model performance.



| Metric | Purpose |

|---------|----------|

| Silhouette Score | Measures cluster cohesion and separation |

| Calinski-Harabasz Index | Evaluates cluster compactness |

| Davies-Bouldin Index | Measures cluster similarity |



---



# 📊 Results



After comparing multiple clustering algorithms, the following observations were obtained:



- K-Means achieved the strongest overall clustering performance.

- Hierarchical Clustering produced comparable segmentation results.

- DBSCAN successfully detected noise points but generated fewer meaningful customer groups.



The optimal number of clusters was selected using:



- Elbow Method

- Silhouette Score



---



# 📈 Visualizations



The notebook includes several visualizations that support the analysis.



### Exploratory Analysis



- Distribution of categorical features

- Distribution of numerical features

- Correlation heatmap



### Clustering Analysis



- Elbow Method

- Silhouette Score comparison

- PCA cluster visualization

- Cluster comparison



Example figures:



```

results/
├── plots/                         # Generated visualizations
│   ├── age_income_distribution.png
│   ├── categorical_distribution.png
│   ├── correlation_matrix.png
│   ├── dbscan_k_distance.png
│   ├── dendrogram.png
│   ├── kmeans_clusters.png
│   └── pca_variance.png

```

---

# 📸 Project Gallery

The following visualizations summarize the main analysis and clustering results generated throughout the project.

## Exploratory Data Analysis

| Categorical Distribution | Age & Income Distribution |
|--------------------------|---------------------------|
| ![](results/plots/categorical_distribution.png) | ![](results/plots/age_income_distribution.png) |

| Correlation Matrix | PCA Explained Variance |
|--------------------|------------------------|
| ![](results/plots/correlation_matrix.png) | ![](results/plots/pca_variance.png) |

## Clustering Results

| K-Means Clusters | Hierarchical Clustering |
|------------------|-------------------------|
| ![](results/plots/kmeans_clusters.png) | ![](results/plots/dendrogram.png) |

| DBSCAN k-Distance Graph |
|-------------------------|
| ![](results/plots/dbscan_k_distance.png) |

---

# 💼 Business Insights



The discovered customer segments can support several business decisions, including:



- Personalized marketing campaigns

- Customer targeting

- Loyalty program optimization

- Product recommendation strategies

- Customer retention initiatives

- Data-driven decision making



These insights enable businesses to better understand customer behavior and allocate marketing resources more effectively.



---



# 🔮 Future Improvements



Possible extensions of this project include:



- Applying Gaussian Mixture Models (GMM)

- Feature engineering

- Hyperparameter optimization

- Interactive dashboards using Streamlit

- Deployment as a web application

- Real-time customer segmentation

# ⚙️ Installation



## 1. Clone the Repository



```bash

git clone https://github.com/your-username/customer-segmentation.git

cd customer-segmentation

```



---



## 2. Create a Virtual Environment (Optional)



### Windows



```bash

python -m venv venv

venv\\Scripts\\activate

```



### Linux / macOS



```bash

python3 -m venv venv

source venv/bin/activate

```



---



## 3. Install Dependencies



```bash

pip install -r requirements.txt

```



---



## 4. Launch Jupyter Notebook



```bash

jupyter notebook

```



Open the notebook:



```

Customer_Segmentation.ipynb

```



---



# 🚀 Project Workflow



The project follows a structured machine learning workflow:



### 1. Data Exploration



- Dataset overview

- Feature analysis

- Statistical summary

- Missing value verification

- Correlation analysis



---



### 2. Data Preprocessing



- Remove unnecessary columns

- Standardize numerical features

- Prepare data for clustering



---



### 3. Clustering



Three clustering algorithms were implemented and compared:



### K-Means



- Elbow Method

- Silhouette Score

- Cluster interpretation



### Agglomerative Hierarchical Clustering



- Dendrogram analysis

- Cluster comparison



### DBSCAN



- Density-based clustering

- Noise point detection



---



### 4. Model Evaluation



Different clustering metrics were used to compare model performance.



| Metric | Purpose |

|---------|----------|

| Silhouette Score | Measures cluster cohesion and separation |

| Calinski-Harabasz Index | Evaluates cluster compactness |

| Davies-Bouldin Index | Measures cluster similarity |



---



# 📊 Results



After comparing multiple clustering algorithms, the following observations were obtained:



- K-Means achieved the strongest overall clustering performance.

- Hierarchical Clustering produced comparable segmentation results.

- DBSCAN successfully detected noise points but generated fewer meaningful customer groups.



The optimal number of clusters was selected using:



- Elbow Method

- Silhouette Score



---



# 📈 Visualizations



The notebook includes several visualizations that support the analysis.



### Exploratory Analysis



- Distribution of categorical features

- Distribution of numerical features

- Correlation heatmap



### Clustering Analysis



- Elbow Method

- Silhouette Score comparison

- PCA cluster visualization

- Cluster comparison



Example figures:



```

images/

│

├── correlation_heatmap.png

├── elbow_method.png

├── silhouette_scores.png

└── cluster_visualization.png

```



---



# 💼 Business Insights



The discovered customer segments can support several business decisions, including:



- Personalized marketing campaigns

- Customer targeting

- Loyalty program optimization

- Product recommendation strategies

- Customer retention initiatives

- Data-driven decision making



These insights enable businesses to better understand customer behavior and allocate marketing resources more effectively.



---



# 🔮 Future Improvements



Possible extensions of this project include:



- Applying Gaussian Mixture Models (GMM)

- Feature engineering

- Hyperparameter optimization

- Interactive dashboards using Streamlit

- Deployment as a web application

- Real-time customer segmentation

---



# 📦 Requirements



The project was developed using Python and the following core libraries:



- Python 3.10+

- Pandas

- NumPy

- Matplotlib

- Seaborn

- Scikit-learn

- Jupyter Notebook



Install all dependencies using:



```bash

pip install -r requirements.txt

```



---



# 🤝 Contributing



Contributions are always welcome.



If you would like to improve this project:



1. Fork the repository.

2. Create a new branch.



```bash

git checkout -b feature/your-feature

```



3. Commit your changes.



```bash

git commit -m "Add your feature"

```



4. Push to your branch.



```bash

git push origin feature/your-feature

```



5. Open a Pull Request.



Please ensure your code is well documented and follows Python best practices.



---



# 📄 License



This project is distributed under the \*\*MIT License\*\*.



See the **LICENSE** file for more information.



---



# 👩‍💻 Author



**Aya Mohamed**



Computer Science Student | AI & Data Science



Interested in:



- Machine Learning

- Data Science

- Data Engineering

- Artificial Intelligence



---



# 📬 Contact



- **LinkedIn:** *https://www.linkedin.com/in/aya-abd-elazim-94a256347/*

- **GitHub:** *https://github.com/Aya-Mohamed945*

- **Email:** *aya.320240137@ejust.edu.eg*



---



# 🙏 Acknowledgments



Special thanks to the open-source community and the developers behind:



- Scikit-learn

- Pandas

- NumPy

- Matplotlib

- Seaborn



for providing the tools that made this project possible.



---



# ⭐ Support



If you found this repository helpful:



⭐ Star the repository



🍴 Fork the project



📢 Share it with others



💡 Feel free to contribute improvements



---



## 📌 Project Highlights



✔ Customer Segmentation using Unsupervised Learning



✔ Data Cleaning \& Preprocessing



✔ Feature Standardization



✔ K-Means Clustering



✔ Hierarchical Clustering



✔ DBSCAN



✔ PCA Visualization



✔ Elbow Method



✔ Silhouette Score Evaluation



✔ Business Insight Generation



---



> **This project demonstrates a complete customer segmentation workflow, from data preparation and exploratory analysis to clustering, evaluation, visualization, and business insight generation using modern unsupervised machine learning techniques.**

