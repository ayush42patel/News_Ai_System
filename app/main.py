from fastapi import FastAPI, Depends, Body
from sqlalchemy.orm import Session
import joblib

from app.models import ReadingHistory
from app.auth import get_current_user
from app.recommender import recommend_for_user
from app.database import SessionLocal, engine
from app.models import Base, News
from app.news_fetcher import fetch_and_store_news
from app.recommender import recommend_articles
from app.models import User
from app.auth import hash_password, verify_password, create_access_token
from fastapi import HTTPException


#Base.metadata.create_all(bind=engine)

app = FastAPI()

# 🔹 LOAD ML MODEL (ADD THIS BLOCK)
model = joblib.load("app/ml/fake_news_model.pkl")
vectorizer = joblib.load("app/ml/vectorizer.pkl")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {"message": "AI News Backend Running"}

@app.get("/fetch-news")
def fetch_news(db: Session = Depends(get_db)):
    fetch_and_store_news(db)
    return {"status": "News fetched"}

@app.get("/news")
def get_news(db: Session = Depends(get_db)):
    return db.query(News).all()

# 🔹 FAKE NEWS PREDICTION API (ADD THIS)
@app.post("/predict")
def predict_news(text: str = Body(..., embed=True)):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec).max()

    return {
        "prediction": "REAL" if prediction == 1 else "FAKE",
        "confidence": round(float(probability), 3)
    }

@app.get("/recommend/{article_id}")
def get_recommendations(article_id: int, db: Session = Depends(get_db)):
    recs = recommend_articles(db, article_id)
    return recs

@app.post("/register")
def register(username: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if user:
        raise HTTPException(status_code=400, detail="Username exists")

    new_user = User(username=username, hashed_password=hash_password(password))
    db.add(new_user)
    db.commit()
    return {"message": "User created"}

@app.post("/login")
def login(username: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/read/{article_id}")
def read_article(article_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    entry = ReadingHistory(user_id=user.id, article_id=article_id)
    db.add(entry)
    db.commit()
    return {"message": "Reading history updated"}

@app.get("/recommend-user")
def user_recommendations(db: Session = Depends(get_db), user=Depends(get_current_user)):
    return recommend_for_user(db, user.id)