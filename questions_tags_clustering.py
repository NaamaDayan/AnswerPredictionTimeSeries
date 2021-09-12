from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def _fit_kmeans(encodings, k):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(encodings)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(scaled_features)
    return (k, kmeans.labels_), kmeans.inertia_, silhouette_score(encodings, kmeans.labels_, metric='euclidean')


def _calc_tfidf_tag_encodings(questions):
    vectorizer = TfidfVectorizer()
    questions['tags'].fillna("", inplace=True)
    return vectorizer.fit_transform(questions['tags'].values).toarray()


def get_questions_tag_groups(questions, k_options=range(2, 15)):
    encodings = _calc_tfidf_tag_encodings(questions)
    labels, distortions, silhouette_values = zip(*list(map(lambda k: _fit_kmeans(encodings, k), k_options)))
    scores_by_k = dict(zip(k_options, silhouette_values))
    best_k = max(scores_by_k, key=scores_by_k.get)
    return dict(labels)[best_k]
