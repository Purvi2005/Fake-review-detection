import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st

# ---------------------------
# Step 1: Data Collection Function
# ---------------------------
def get_amazon_reviews(review_url, max_reviews=10):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/114.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(review_url, headers=headers, timeout=10)
        if response.status_code != 200:
            st.error(f"Failed to fetch page, status code: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching reviews: {e}")
        return []
    
    soup = BeautifulSoup(response.content, "html.parser")
    reviews = []

    review_spans = soup.find_all("span", {"data-hook": "review-body"})
    for span in review_spans[:max_reviews]:
        review_text = span.get_text(strip=True)
        if review_text:
            reviews.append(review_text)
    return reviews

# ---------------------------
# Step 2: Model Training and Saving
# ---------------------------
def train_and_save_model():
    # Sample labeled dataset (for demo purposes)
    reviews = [
        "Great product! Must buy now!",  # fake
        "This is the best purchase I've ever made.",  # fake
        "Five stars, highly recommend this.",  # fake
        "Awesome product, worth every penny!",  # fake
        "I love this item, works perfectly.",  # real
        "The product matches the description and arrived on time.",  # real
        "Good quality for the price.",  # real
        "Not satisfied with the purchase. It broke after a week.",  # real
        "The item is defective and customer service was unhelpful.",  # real
        "Do not buy this product, very disappointing.",  # real
    ]

    labels = [1,1,1,1,0,0,0,0,0,0]  # 1=fake, 0=real

    X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.3, random_state=42, stratify=labels)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(random_state=42, max_iter=200)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Real', 'Fake'])

    # Save model and vectorizer
    joblib.dump(clf, 'fake_review_model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')

    return acc, report

# ---------------------------
# Step 3: Load model and analyze reviews
# ---------------------------
def load_model():
    model = joblib.load('fake_review_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    return model, vectorizer

def predict_fake(model, vectorizer, review):
    review_vec = vectorizer.transform([review])
    pred = model.predict(review_vec)
    return pred[0]  # 1 for fake, 0 for real

def analyze_reviews(model, vectorizer, reviews):
    sentiment_counts = {'Positive':0, 'Negative':0, 'Neutral':0}
    fake_counts = {'Fake': 0, 'Real': 0}

    detailed_results = []

    for review in reviews:
        polarity = TextBlob(review).sentiment.polarity
        if polarity > 0:
            sentiment = 'Positive'
        elif polarity < 0:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        sentiment_counts[sentiment] += 1

        fake_flag = predict_fake(model, vectorizer, review)
        label = 'Fake' if fake_flag == 1 else 'Real'
        fake_counts[label] += 1

        detailed_results.append({
            'review': review,
            'sentiment': sentiment,
            'polarity': polarity,
            'fake_or_real': label
        })

    return sentiment_counts, fake_counts, detailed_results

# ---------------------------
# Step 4: Visualization of Sentiments
# ---------------------------
def plot_sentiment_graph(sentiment_counts):
    labels = list(sentiment_counts.keys())
    counts = [sentiment_counts[label] for label in labels]
    colors = ['green', 'red', 'gray']

    plt.figure(figsize=(8,5))
    bars = plt.bar(labels, counts, color=colors)
    plt.title("Sentiment Distribution of Amazon Reviews")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.ylim(0, max(counts)+1)

    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height}',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0,3),
                     textcoords='offset points',
                     ha='center', va='bottom')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_fake_real_graph(fake_counts):
    labels = list(fake_counts.keys())
    counts = [fake_counts[label] for label in labels]
    colors = ['red', 'blue']

    plt.figure(figsize=(6,4))
    bars = plt.bar(labels, counts, color=colors)
    plt.title("Fake vs Real Reviews")
    plt.xlabel("Label")
    plt.ylabel("Number of Reviews")
    plt.ylim(0, max(counts)+1)

    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height}',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0,3),
                     textcoords='offset points',
                     ha='center', va='bottom')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

# ---------------------------
# Streamlit Web App Interface
# ---------------------------
def main():
    st.title("Amazon Review Analysis & Fake Review Detection")

    st.header("Step 1: Train Model")
    if st.button("Train Model"):
        acc, report = train_and_save_model()
        st.success(f"Model trained with accuracy: {acc:.2f}")
        st.text("Classification Report:")
        st.text(report)

    st.header("Step 2: Enter Amazon Review Page URL")
    url = st.text_input("Enter Amazon All Reviews URL:")

    max_reviews = st.slider("Number of Reviews to Analyze", 1, 50, 10)

    if url and st.button("Fetch and Analyze Reviews"):
        with st.spinner("Fetching reviews..."):
            reviews = get_amazon_reviews(url, max_reviews)
        if not reviews:
            st.error("No reviews found or unable to fetch reviews.")
            return

        st.write(f"Fetched {len(reviews)} reviews:")
        for i, rev in enumerate(reviews, 1):
            st.markdown(f"**Review {i}:** {rev}")

        # Load the saved model and vectorizer
        model, vectorizer = load_model()

        sentiment_counts, fake_counts, detailed_results = analyze_reviews(model, vectorizer, reviews)

        st.header("Sentiment Summary")
        st.write(sentiment_counts)
        plot_sentiment_graph(sentiment_counts)

        st.header("Fake vs Real Review Count")
        st.write(fake_counts)
        plot_fake_real_graph(fake_counts)

        st.header("Detailed Review Analysis")
        for res in detailed_results:
            st.markdown(f"- Review: {res['review']}")
            st.markdown(f"  - Sentiment: {res['sentiment']} (Polarity: {res['polarity']:.2f})")
            fake_color = 'red' if res['fake_or_real'] == 'Fake' else 'green'
            st.markdown(f"  - Authenticity: <span style='color:{fake_color}'>{res['fake_or_real']}</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
