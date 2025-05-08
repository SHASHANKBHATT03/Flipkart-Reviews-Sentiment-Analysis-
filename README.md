# Flipkart-Reviews-Sentiment-Analysis-

# Flipkart Product Reviews – Sentiment Analysis

This project performs sentiment analysis on Flipkart product reviews using basic Natural Language Processing (NLP) techniques and a Decision Tree Classifier. The goal is to classify customer reviews as either **positive** or **negative**.

---

## 📄 Dataset

- A CSV file containing Flipkart product reviews
- Columns:
  - `Review` (text)
  - `Sentiment` (1 for positive, 0 for negative)

---

## 🛠 Libraries & Tools Used

- `pandas` – for loading and manipulating data  
- `nltk` – for stopword removal  
- `scikit-learn`:
  - `TfidfVectorizer` – to convert text into numeric vectors  
  - `train_test_split` – for splitting the dataset  
  - `DecisionTreeClassifier` – ML model  
  - `accuracy_score`, `confusion_matrix` – evaluation metrics  
- `wordcloud`, `matplotlib`, `seaborn` – for visualizations

---

## ⚙️ Project Workflow

1. **Data Loading**
   - Read CSV file with review text and sentiment labels

2. **Text Preprocessing**
   - Converted reviews to lowercase  
   - Removed stopwords using NLTK

3. **Feature Extraction**
   - Used `TfidfVectorizer` to convert reviews into numerical format

4. **Model Training**
   - Split the dataset (80% training, 20% testing)  
   - Trained a `DecisionTreeClassifier` on the vectorized data

5. **Model Evaluation**
   - Calculated accuracy score on test data  
   - Plotted the confusion matrix using `seaborn`

6. **Visualization**
   - Generated a WordCloud from the reviews

---

## 📊 Results

- **Model**: Decision Tree Classifier  
- **Vectorizer**: TF-IDF  
- **Evaluation Metric**: Accuracy + Confusion Matrix  
- **Visualization**: WordCloud of review text 
