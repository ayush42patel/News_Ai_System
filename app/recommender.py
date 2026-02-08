from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.models import News

# Shared vectorizer
vectorizer = TfidfVectorizer(stop_words="english")


def _get_texts(articles):
    return [a.title + " " + (a.content or "") for a in articles]


# 🔹 ARTICLE-TO-ARTICLE RECOMMENDATION
def recommend_articles(db, article_id, top_k=5):
    articles = db.query(News).all()

    if not articles:
        return []

    texts = _get_texts(articles)
    tfidf_matrix = vectorizer.fit_transform(texts)

    index_map = {article.id: idx for idx, article in enumerate(articles)}
    if article_id not in index_map:
        return []

    target_idx = index_map[article_id]
    sims = cosine_similarity(tfidf_matrix[target_idx], tfidf_matrix).flatten()

    similar_indices = sims.argsort()[::-1][1:top_k+1]
    return [articles[i] for i in similar_indices]


# 🔹 USER PERSONALIZED RECOMMENDATION
def recommend_for_user(db, user_id, top_k=5):
    from app.models import ReadingHistory

    history = db.query(ReadingHistory).filter(ReadingHistory.user_id == user_id).all()
    if not history:
        return []

    read_articles = [h.article for h in history]
    all_articles = db.query(News).all()

    if not all_articles:
        return []

    texts = _get_texts(all_articles)
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Build user profile vector (average of read article vectors)
    read_texts = _get_texts(read_articles)
    user_vec = vectorizer.transform(read_texts).mean(axis=0)

    sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = sims.argsort()[::-1][:top_k]

    return [all_articles[i] for i in top_indices]
