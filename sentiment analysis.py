from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from textblob import TextBlob
import time
import colorama
from colorama import Fore, Back, Style

colorama.init(autoreset=True)

def scrape_amazon_reviews(url, max_reviews=10):
    # Setup headless Chrome
    options = Options()
    options.add_argument("--headless")
    service = Service()  # You can pass path to chromedriver if needed

    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    time.sleep(3)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()

    review_elements = soup.find_all("span", {"data-hook": "review-body"})
    review_texts = [review.get_text(strip=True) for review in review_elements[:max_reviews]]

    if not review_texts:
        return ["No reviews found or invalid Amazon link."]
    
    return review_texts

def analyze_sentiment(review):
    polarity = TextBlob(review).sentiment.polarity
    if polarity > 0:
        return "Positive", polarity, Fore.GREEN + Style.BRIGHT
    elif polarity < 0:
        return "Negative", polarity, Fore.RED + Style.BRIGHT
    else:
        return "Neutral", polarity, Fore.YELLOW + Style.BRIGHT

# ---------- Main ----------
user_url = input("Enter the Amazon product URL:\n").strip()
reviews = scrape_amazon_reviews(user_url)

if isinstance(reviews, list) and "No reviews found" not in reviews[0]:
    for i, review in enumerate(reviews, 1):
        sentiment, polarity, color = analyze_sentiment(review)
        print(f"{Style.BRIGHT}Review {i}: {Fore.CYAN}{review}")
        print(f"{color}Sentiment: {sentiment} (Polarity: {polarity:.2f}){Style.RESET_ALL}\n")
else:
    print(Fore.RED + reviews[0])
