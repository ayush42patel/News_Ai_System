from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.models import News

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_article_embeddings(articles):
    texts = [a.title + " " + (a.content or "") for a in articles]
    return model.encode(texts)

def recommend_articles(db, article_id, top_k=5):
    articles = db.query(News).all()

    if not articles:
        return []

    embeddings = get_article_embeddings(articles)

    index_map = {article.id: idx for idx, article in enumerate(articles)}
    if article_id not in index_map:
        return []

    target_index = index_map[article_id]
    sims = cosine_similarity([embeddings[target_index]], embeddings)[0]

    # sort by similarity
    similar_indices = sims.argsort()[::-1][1:top_k+1]

    return [articles[i] for i in similar_indices]

def recommend_for_user(db, user_id, top_k=5):
    from app.models import ReadingHistory

    history = db.query(ReadingHistory).filter(ReadingHistory.user_id == user_id).all()
    if not history:
        return []

    read_articles = [h.article for h in history]
    all_articles = db.query(News).all()

    user_vec = model.encode([a.title + " " + (a.content or "") for a in read_articles]).mean(axis=0)
    article_vecs = model.encode([a.title + " " + (a.content or "") for a in all_articles])

    sims = cosine_similarity([user_vec], article_vecs)[0]
    top_indices = sims.argsort()[::-1][:top_k]

    return [all_articles[i] for i in top_indices]
