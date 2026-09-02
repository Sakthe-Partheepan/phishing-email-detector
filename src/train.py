import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# Load data
df = pd.read_csv('data/phishing_email.csv')

# Split BEFORE vectorizing, to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    df['text_combined'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Vectorize: fit on train only, transform both
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- Model 1: Logistic Regression ---
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_tfidf, y_train)
logreg_preds = logreg.predict(X_test_tfidf)

print("=== Logistic Regression ===")
print(f"Accuracy:  {accuracy_score(y_test, logreg_preds):.4f}")
print(f"Precision: {precision_score(y_test, logreg_preds):.4f}")
print(f"Recall:    {recall_score(y_test, logreg_preds):.4f}")
print(f"F1 Score:  {f1_score(y_test, logreg_preds):.4f}")
print()

# --- Model 2: Multinomial Naive Bayes ---
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
nb_preds = nb.predict(X_test_tfidf)

print("=== Multinomial Naive Bayes ===")
print(f"Accuracy:  {accuracy_score(y_test, nb_preds):.4f}")
print(f"Precision: {precision_score(y_test, nb_preds):.4f}")
print(f"Recall:    {recall_score(y_test, nb_preds):.4f}")
print(f"F1 Score:  {f1_score(y_test, nb_preds):.4f}")



# --- Model 3: Linear SVM ---
svm = LinearSVC(max_iter=2000)
svm.fit(X_train_tfidf, y_train)
svm_preds = svm.predict(X_test_tfidf)

print("=== Linear SVM ===")
print(f"Accuracy:  {accuracy_score(y_test, svm_preds):.4f}")
print(f"Precision: {precision_score(y_test, svm_preds):.4f}")
print(f"Recall:    {recall_score(y_test, svm_preds):.4f}")
print(f"F1 Score:  {f1_score(y_test, svm_preds):.4f}")

# Save the better-performing model + vectorizer for later use in the FastAPI app

# Linear SVM won on all metrics — this is the model going into deployment
joblib.dump(svm, 'models/svm_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')