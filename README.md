# IMDB Movie Reviews Sentiment Analysis

This repository contains a Jupyter Notebook that demonstrates sentiment analysis using a machine learning model. The dataset used consists of 50,000 IMDB movie reviews, which are labeled as either positive or negative. This project showcases data preprocessing, exploratory data analysis, and model training for sentiment classification.

## Dataset
The dataset used in this project is the IMDB Dataset of 50K Movie Reviews, which is available on Kaggle:
[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download)

### Dataset Details
- **Columns:**
  - `review`: The text of the movie review.
  - `sentiment`: The sentiment label (`positive` or `negative`).
- **Size:** 50,000 rows.
- **Purpose:** Sentiment analysis for text classification tasks.

## Requirements
To run the Jupyter Notebook, you need the following Python libraries installed:

- pandas
- numpy
- nltk
- sklearn
- matplotlib
- seaborn
- wordcloud

Install the required libraries using pip:
```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn wordcloud
```

## Project Steps

### 1. Data Loading
- The dataset is loaded into a pandas DataFrame.
- Initial exploration is performed to understand the structure and distribution of the data.

### 2. Data Preprocessing
- **Text Cleaning:**
  - Lowercasing text.
  - Removing HTML tags.
  - Removing URLs, special characters, and stopwords.
- **Tokenization and Stemming:**
  - Tokenize the text into words.
  - Apply stemming using NLTK's `PorterStemmer`.

### 3. Exploratory Data Analysis
- Visualizations using matplotlib and seaborn to explore word distributions and sentiment proportions.
- A WordCloud is generated to highlight frequently occurring words.

### 4. Feature Engineering
- Text features are extracted using TF-IDF vectorization.

### 5. Model Training and Evaluation
- The data is split into training and testing sets.
- A machine learning model (e.g., Logistic Regression, Naive Bayes, or Random Forest) is trained.
- Metrics like accuracy, precision, recall, and F1-score are used for evaluation.

### 6. Results
- The model's performance on the test set is presented.
- Misclassified examples are analyzed to understand areas of improvement.

## Running the Notebook
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download).
2. Save the dataset as `IMDB Dataset.csv` in the same directory as the notebook.
3. Open the Jupyter Notebook and run the cells sequentially.

## Output
The notebook outputs:
- Data visualizations.
- Trained machine learning model.
- Sentiment predictions for sample reviews.

## References
- [Kaggle: IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download)
- NLTK Documentation: [https://www.nltk.org/](https://www.nltk.org/)
