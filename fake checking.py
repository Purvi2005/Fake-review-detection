import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

# Load your ML model (must be pre-trained)
# For demo: We will simulate with a rule-based fake review detector
def is_fake_review(review):
    suspicious_phrases = ['great product', 'must buy', 'five stars', 'awesome', 'worth every penny']
    return any(phrase in review.lower() for phrase in suspicious_phrases)

# Function to get reviews from the "all reviews" page url
def get_amazon_reviews(review_url, max_reviews=10):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/114.0.0.0 Safari/537.36"
    }
    response = requests.get(review_url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch page, status code: {response.status_code}")
        return ["No reviews found or invalid Amazon review link."]
    
    soup = BeautifulSoup(response.content, "html.parser")
    reviews = []
    
    # Amazon reviews are in span tags with data-hook='review-body'
    review_spans = soup.find_all("span", {"data-hook": "review-body"})
    for span in review_spans[:max_reviews]:
        review_text = span.get_text(strip=True)
        if review_text:
            reviews.append(review_text)
    return reviews if reviews else ["No reviews found or invalid Amazon review link."]

# Color formatting for terminal output
def color_text(text, color):
    colors = {'green': '32', 'red': '31', 'gray': '37'}
    return f"\033[1;{colors.get(color, '37')}m{text}\033[0m"

# Main function to analyze reviews
def analyze_reviews(reviews):
    for i, review in enumerate(reviews, 1):
        print(f"\nReview {i}:")
        print(color_text(review, 'gray'))

        polarity = TextBlob(review).sentiment.polarity
        if polarity > 0:
            sentiment = 'Positive'
            color = 'green'
        elif polarity < 0:
            sentiment = 'Negative'
            color = 'red'
        else:
            sentiment = 'Neutral'
            color = 'gray'

        print(f"Sentiment: {color_text(sentiment, color)}")
        print(f"Sentiment Polarity: {polarity:.2f}")

        fake = is_fake_review(review)
        status = "Fake" if fake else "Real"
        print(f"Fake Review Detection: {color_text(status, 'red' if fake else 'green')}")

if __name__ == "__main__":
    print("IMPORTANT: You need to provide the URL of the Amazon ALL REVIEWS page for the product.")
    print("Example: https://www.amazon.com/product-reviews/PRODUCT_ID")
    review_url = input("Enter the 'All Reviews' page URL from Amazon:\n").strip()
    reviews = get_amazon_reviews(review_url)
    analyze_reviews(reviews)
