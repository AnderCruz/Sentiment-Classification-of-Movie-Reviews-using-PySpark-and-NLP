# Sentiment Classification of Movie Reviews using PySpark and NLP

*A Natural Language Processing project using Spark MLlib*

This project applies **Natural Language Processing (NLP)** techniques with **PySpark** to classify **movie reviews** from IMDb as either **positive** or **negative**.

Developed by **Nowa Analytics**, a consulting firm focused on data science, AI, and Big Data engineering.


## Project Objective

The goal of this project is to build a machine learning model capable of classifying **text reviews** based on their sentiment. We apply various **NLP preprocessing steps** and use **Spark MLlib** to build and evaluate a classification model.

We work with a dataset of user-generated movie reviews labeled as **positive** or **negative**, and use **text processing techniques** to make this unstructured data understandable for a machine learning algorithm.


## Key Learning Objectives

* ✅ Learn the main steps in **Natural Language Processing (NLP)**
* ✅ Clean and preprocess text data using **regular expressions**
* ✅ Apply **tokenization** and **stopword removal**
* ✅ Construct a **Bag of Words** representation
* ✅ Use **TF-IDF** to measure word importance
* ✅ Train a **Decision Tree** classifier using PySpark MLlib
* ✅ Evaluate model performance on unseen data


## Dataset Description

* **Source**: IMDb (Internet Movie Database)
* **Type**: Labeled text reviews
* **Classes**:

  * `positive` – favorable movie reviews
  * `negative` – unfavorable movie reviews
* **Format**: `.CSV` or `.TSV` containing text and label columns


## Technologies and Libraries

* Python 3.9+
* Apache Spark (PySpark)
* Spark MLlib (for classification)
* Spark NLP features (tokenizer, stopword remover, TF-IDF)
* Regex (for text cleaning)
* Jupyter Notebook


## Project Structure

```
📦 movie-review-sentiment-spark
│
├── data/                  # Raw and cleaned review datasets
├── notebooks/             # Jupyter Notebooks for EDA and modeling
├── src/                   # Python scripts for preprocessing and modeling
│   ├── text_cleaning.py
│   ├── sentiment_pipeline.py
│   └── evaluation.py
├── results/               # Output metrics, visualizations, and logs
├── README.md              # Project documentation
└── requirements.txt       # Required Python libraries
```


## Main Steps in the Pipeline

1. **Text Preprocessing**

   * Removing special characters and punctuation using **regular expressions**
   * Lowercasing all text
   * Tokenizing text (splitting sentences into words)
   * Removing stopwords (common, meaningless words like "the", "and", etc.)

2. **Feature Extraction**

   * **Bag of Words (BoW)** representation
   * **TF-IDF (Term Frequency - Inverse Document Frequency)** to assess word importance

3. **Model Training**

   * Training a **Decision Tree Classifier** using PySpark MLlib
   * Splitting the data into training and test sets

4. **Model Evaluation**

   * Accuracy, Precision, Recall, F1-Score

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/movie-review-sentiment-spark.git
cd movie-review-sentiment-spark
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start a PySpark session and open the notebooks:

```bash
jupyter notebook
```

4. Or run the classification pipeline:

```bash
spark-submit src/sentiment_pipeline.py
```


## Sample Results

| Metric   | Value |
| -------- | ----- |
| Accuracy | 0.86  |
| F1-Score | 0.84  |

> Results may vary depending on dataset size and preprocessing techniques.


## Key Insights

* Words with high TF-IDF scores strongly influenced classification decisions.
* Cleaning and removing irrelevant words greatly improved model performance.
* PySpark’s scalable NLP tools make it easy to process large text datasets efficiently.


## About Nowa Analytics

**Nowa Analytics** is a boutique consulting firm that specializes in data science, AI, and scalable data engineering. We help companies turn complex data into actionable insights using modern tools and cloud-based architectures.

📍 Offices in São Paulo, Madrid, and London
🌐 [nowaanalytics.com](http://nowaanalytics.com) *(replace with your official link)*

---

## 📬 Contact

* 📧 [contact@nowaanalytics.com](mailto:contact@nowaanalytics.com)
* 💼 [LinkedIn – Nowa Analytics](https://linkedin.com/company/nowaanalytics)
