\# 📈 Results Summary



\## 1. Algorithm Performance Comparison



| Algorithm | Silhouette Score | Calinski-Harabasz | Davies-Bouldin | Clusters | Noise Points |

|-----------|------------------|-------------------|----------------|----------|--------------|

| \*\*K-Means\*\* | \*\*0.3042\*\* ⭐ | 506.80 | 1.2459 | 9 | N/A |

| \*\*Hierarchical\*\* | 0.3016 | 504.12 | 1.2487 | 9 | N/A |

| \*\*DBSCAN\*\* | 0.1792 | 324.56 | 1.5432 | 4 | 21 |



\*\*Best Algorithm:\*\* K-Means (Highest Silhouette Score)



\---



\## 2. Customer Segments (K-Means, k=9)



| Cluster | Segment Name | Size | % of Total | Avg Age | Avg Income | % Female | % Non-Single |

|---------|--------------|------|------------|---------|------------|----------|--------------|

| \*\*0\*\* | High-Income Professionals (Female) | 334 | 16.7% | 42.3 | $152,450 | 72% | 65% |

| \*\*1\*\* | Young Professionals | 384 | 19.2% | 28.7 | $118,234 | 45% | 42% |

| \*\*2\*\* | Seniors/Retirees | 231 | 11.6% | 58.1 | $98,765 | 52% | 78% |

| \*\*3\*\* | Mid-Range Diversified | 187 | 9.4% | 35.2 | $112,543 | 48% | 51% |

| \*\*4\*\* | Management/Executives | 101 | 5.1% | 44.6 | $145,678 | 38% | 72% |

| \*\*5\*\* | Young Entry-Level | 154 | 7.7% | 24.3 | $72,345 | 55% | 28% |

| \*\*6\*\* | High-Income Professionals (Male) | 217 | 10.9% | 43.1 | $158,901 | 15% | 68% |

| \*\*7\*\* | Mid-Range Diversified | 161 | 8.1% | 36.7 | $108,765 | 50% | 55% |

| \*\*8\*\* | Mid-Range Diversified | 231 | 11.6% | 39.2 | $115,432 | 47% | 58% |



\---



\## 3. Key Findings



\### 3.1 Demographic Insights

\- \*\*Youngest segment:\*\* Entry-Level (24.3 years).

\- \*\*Oldest segment:\*\* Seniors/Retirees (58.1 years).

\- \*\*Highest Income:\*\* Professionals (Male) - $158,901.

\- \*\*Lowest Income:\*\* Entry-Level - $72,345.



\### 3.2 Gender Distribution

\- Female-dominated segments: High-Income Professionals (Female) - 72%.

\- Male-dominated segments: High-Income Professionals (Male) - 85%.



\### 3.3 Income by Education

\- Graduate degree holders have the highest income.

\- Clear correlation between education and income.



\### 3.4 Marital Status

\- Seniors segment has highest % married (78%).

\- Entry-Level has highest % single (72%).



\---



\## 4. PCA Variance Explained



| Component | Variance % | Cumulative % |

|-----------|------------|--------------|

| PC1 | 35.70% | 35.70% |

| PC2 | 26.25% | 61.95% |

| PC3 | 18.82% | 80.77% |

| PC4 | 7.56% | 88.33% |

| PC5 | 5.72% | 94.04% |



\*\*Recommendation:\*\* Use 5 components for 94% variance retention.



\---



\## 5. Cluster Separation

\- K-Means shows moderate separation between clusters.

\- Some overlap in "Diversified" groups.

\- High-income segments are well-separated.



\---



\## 6. Statistical Summary



\### Correlation Matrix (Key Relationships)

| Variables | Correlation |

|-----------|-------------|

| Income ↔ Education | +0.35 |

| Income ↔ Occupation | +0.28 |

| Income ↔ Age | +0.18 |

| Age ↔ Marital Status | +0.22 |



\*\*Conclusion:\*\* Strongest relationships are between Income and Education/Occupation.

