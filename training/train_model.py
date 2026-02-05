import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load data
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true])
df = df[["text", "label"]]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluation
preds = model.predict(X_test_vec)
print(classification_report(y_test, preds))

# Save model
joblib.dump(model, "app/ml/fake_news_model.pkl")
joblib.dump(vectorizer, "app/ml/vectorizer.pkl")

print("Model saved!")
