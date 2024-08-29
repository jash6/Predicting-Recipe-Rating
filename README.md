# Recipe for Success: Predictive Analytics for Recipe Recommendations
## Table of Contents
1. [Introduction](#1-introduction)
2. [Data Preprocessing](#2-data-preprocessing)
   1. [Dataset Overview](#21-dataset-overview)
   2. [Missing Values Handling](#22-missing-values-handling)
   3. [Data Splitting](#23-data-splitting)
   4. [Feature Engineering](#24-feature-engineering)
3. [Modeling Approaches](#3-modeling-approaches)
   1. [Similarity-Based Models](#31-similarity-based-models)
      1. [Jaccard Similarity](#311-jaccard-similarity)
      2. [Cosine Similarity](#312-cosine-similarity)
      3. [Pearson Similarity](#313-pearson-similarity)
      4. [Ratings Prediction Function](#314-ratings-prediction-function)
      5. [Temporal Factors](#315-temporal-factors)
   2. [Latent Factor Models](#32-latent-factor-models)
      1. [Bias-Only Model](#321-bias-only-model)
      2. [Latent Factor Model](#322-latent-factor-model)
      3. [SVD++](#323-svd)
   3. [Text-Based Models](#33-text-based-models)
      1. [Bag of Words (BoW)](#331-bag-of-words-bow)
      2. [TF-IDF Vectorization](#332-tf-idf-vectorization)
      3. [Using Text Embeddings](#333-using-text-embeddings)
      4. [Textual Compatibility-Based Recommendation](#334-textual-compatibility-based-recommendation)
4. [Experimental Results](#4-experimental-results)
   1. [Similarity-Based Models](#41-similarity-based-models)
   2. [Latent Factor Models](#42-latent-factor-models)
   3. [Text-Based Models](#43-text-based-models)
5. [Conclusion and Future Work](#5-conclusion-and-future-work)
6. [Repository Structure](#6-repository-structure)
7. [Running the Project](#7-running-the-project)
   1. [Prerequisites](#prerequisites)
   2. [Setting up the Environment](#setting-up-the-environment)


## 1. Introduction
This project aims to predict recipe preferences by analyzing user behavior and recipe features. We use a dataset derived from Food.com, containing reviews, ratings, and recipe metadata. Our primary goal is to build models that can predict user ratings for recipes based on historical data.

## 2. Data Preprocessing

### 2.1 Dataset Overview
The dataset comprises recipes and corresponding reviews. Each review consists of user information, a rating (1 to 5), and text feedback.

### 2.2 Missing Values Handling
We removed entries with missing ratings or essential metadata, such as recipe names or ingredients.

### 2.3 Data Splitting
We split the dataset into training, validation, and test sets, maintaining consistent temporal distribution of reviews across all sets.

### 2.4 Feature Engineering
We engineered several features from the dataset, including:
- **User Features:** Number of reviews submitted by the user.
- **Recipe Features:** Average rating, total number of reviews, and ingredient list.
- **Temporal Features:** Time-based features such as days since the last review.

## 3. Modeling Approaches

### 3.1 Similarity-Based Models

#### 3.1.1 Jaccard Similarity
Jaccard similarity measures the intersection over the union of sets:

$$
\text{Jaccard Similarity} = \frac{|U_i \cap U_j|}{|U_i \cup U_j|}
$$

where $U_i$ and $U_j$ represent the set of users who rated items $i$ and $j$ respectively.

#### 3.1.2 Cosine Similarity
Cosine similarity evaluates the cosine of the angle between two vectors:

$$
\text{Cosine Similarity} = \frac{|R_u \cap R_v|}{\sqrt{|R_u| \cdot |R_v|}}
$$

where $R_u$ and $R_v$ represent the set of users who rated items $u$ and $v$ respectively.

#### 3.1.3 Pearson Similarity
Pearson similarity measures the linear correlation between two variables:

$$
\text{Pearson Similarity} = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}
$$

#### 3.1.4 Ratings Prediction Function
We use similarities between users and items for rating prediction:

$$
\hat{r}_{ui} = \mu + b_u + b_i + \text{Similarity}(u, i)
$$

#### 3.1.5 Temporal Factors
We incorporated temporal factors by calculating the number of days between two reviews:

$$
f(\text{days}) = \frac{1}{\text{floor}(\frac{\text{days}}{\lambda}) + 1}
$$

### 3.2 Latent Factor Models

#### 3.2.1 Bias-Only Model
This model considers only the overall average rating, user bias, and item bias:

$$
\text{Loss} = \sum_{(u, i) \in \text{data}} (r_{ui} - (\mu + b_u + b_i))^2 + \lambda (\|b_u\|^2 + \|b_i\|^2)
$$

where $\lambda$ is the regularization constant.

#### 3.2.2 Latent Factor Model
We used the SVD algorithm popularized by Simon Funk:

$$
R_{ui} \approx \mu + b_u + b_i + p_u^T q_i
$$

where $\mu$ is the global bias, $b_u$ and $b_i$ are user and item biases, and $p_u$ and $q_i$ are latent factor vectors for users and items, respectively.

#### 3.2.3 SVD++
SVD++ builds on SVD by incorporating both explicit (ratings) and implicit (interactions) feedback into the model.

### 3.3 Text-Based Models

#### 3.3.1 Bag of Words (BoW)
BoW converts a document into a set of words, disregarding grammar and word order.

#### 3.3.2 TF-IDF Vectorization
TF-IDF evaluates the importance of a word to a document in a collection:

$$
\text{TF-IDF} = \text{TF} \cdot \text{IDF}
$$

#### 3.3.3 Using Text Embeddings
Text embeddings were generated using BoW and TF-IDF vectors. The regression model followed the form:

$$
y = \Theta x + b
$$

where $x$ represents vectorized text features and $y$ is the rating.

#### 3.3.4 Textual Compatibility-Based Recommendation
This approach computes similarity based on recipe metadata (e.g., name, description, ingredients) rather than interactions.

## 4. Experimental Results

### 4.1 Similarity-Based Models
Jaccard, Cosine, and Pearson similarity measures were evaluated, with temporal factors slightly improving predictions.

### 4.2 Latent Factor Models
Latent factor models performed well, with SVD++ providing the best accuracy. Hyperparameter tuning was crucial.

### 4.3 Text-Based Models
Text-based models effectively utilized textual features, with TF-IDF outperforming BoW. The textual compatibility model also showed promise.

## 5. Conclusion and Future Work

The project demonstrated the effectiveness of combining collaborative filtering methods with latent factor and text-based models. Future work will focus on deep learning approaches and integrating additional features.

## 6. Repository Structure

```
Predicting-Recipe-Rating/
├── main.ipynb # Contains all the code from data cleaning to all models and results
├── Recipe_For_Success # Paper containing all details in more depth
```

## 7. Running the Project

### Prerequisites
- Ensure you have python 3.9 or above installed.
- Download the dataset from the link mentioned in the main.ipynb file

### Setting up the Environment
1. Clone the repository:
   ```
   git clone https://github.com/jash6/Predicting-Recipe-Rating.git
   ```

2. Create a new Conda environment and install dependencies:
   ```
   conda env create -f environment.yml
   conda activate recipe-env
   ```
