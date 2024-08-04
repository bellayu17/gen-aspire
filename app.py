import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from googleapiclient.discovery import build
import requests
from datetime import datetime, timedelta
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes and allow all origins

# API configuration
YOUTUBE_API_KEY = 'AIzaSyCVoyUjeaYbVxJ0PJdiQx39xqi_rbMzF7U'
NEWS_API_KEY = '642b66beb998476c963949a13e59ff48'
EVENTBRITE_API_URL = 'https://www.eventbriteapi.com/v3/users/me/?token=625ORUYPF46KORZIRG3K'
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Initialize a pipeline for keyword extraction using a Hugging Face model
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
keyword_extractor = pipeline('ner', model=model, tokenizer=tokenizer)

def extract_keywords(text):
    keywords = set()
    entities = keyword_extractor(text)
    current_phrase = []
    for entity in entities:
        if entity['entity_group'] in ['ORG', 'MISC', 'LOC', 'PER']:
            if entity['word'].startswith('##'):
                current_phrase[-1] = current_phrase[-1] + entity['word'][2:]
            else:
                if current_phrase:
                    keywords.add(" ".join(current_phrase))
                current_phrase = [entity['word']]
        else:
            if current_phrase:
                keywords.add(" ".join(current_phrase))
                current_phrase = []
    if current_phrase:
        keywords.add(" ".join(current_phrase))
    return list(keywords)

@app.route('/')
def home():
    return "Welcome to the AI Personal Growth Planner API!"

@app.route('/generate-plan', methods=['POST'])
def generate_plan():
    data = request.get_json()
    milestones = data['milestones']
    timeframe = data['timeframe']
    learning_preference = int(data['learningPreference'])

    keywords = extract_keywords(milestones)
    plan = {
        'tasks': []
    }

    for keyword in keywords:
        if learning_preference < 10:
            articles = get_news(keyword)
            books = get_books(keyword)
            for article in filter_relevant_articles(articles, keyword, 15):  # More articles
                plan['tasks'].append({
                    'title': article['title'],
                    'description': article['url'],
                    'type': 'article'
                })
            for book in filter_relevant_books(books, keyword, 5):  # Fewer books
                plan['tasks'].append({
                    'title': book['title'],
                    'description': book['url'],
                    'type': 'book'
                })
        elif 10 <= learning_preference < 50:
            articles = get_news(keyword)
            books = get_books(keyword)
            videos = get_ted_talks(keyword)
            events = get_events(keyword)
            for article in filter_relevant_articles(articles, keyword, 10):  # More articles
                plan['tasks'].append({
                    'title': article['title'],
                    'description': article['url'],
                    'type': 'article'
                })
            for book in filter_relevant_books(books, keyword, 7):  # Fewer books
                plan['tasks'].append({
                    'title': book['title'],
                    'description': book['url'],
                    'type': 'book'
                })
            for video in videos[:5]:  # Even fewer videos
                plan['tasks'].append({
                    'title': video['title'],
                    'description': video['url'],
                    'type': 'video'
                })
            for event in events[:3]:  # Fewest events
                plan['tasks'].append({
                    'title': event['name'],
                    'description': event['url'],
                    'type': 'event'
                })
        elif 50 <= learning_preference <= 70:
            videos = get_ted_talks(keyword)
            events = get_events(keyword)
            for video in videos[:12]:  # More videos
                plan['tasks'].append({
                    'title': video['title'],
                    'description': video['url'],
                    'type': 'video'
                })
            for event in events[:8]:  # Fewer events
                plan['tasks'].append({
                    'title': event['name'],
                    'description': event['url'],
                    'type': 'event'
                })
        elif learning_preference > 70:
            videos = get_ted_talks(keyword)
            events = get_events(keyword)
            articles = get_news(keyword)
            books = get_books(keyword)
            for video in videos[:10]:  # More videos
                plan['tasks'].append({
                    'title': video['title'],
                    'description': video['url'],
                    'type': 'video'
                })
            for event in events[:7]:  # More events
                plan['tasks'].append({
                    'title': event['name'],
                    'description': event['url'],
                    'type': 'event'
                })
            for article in filter_relevant_articles(articles, keyword, 5):  # Fewer articles
                plan['tasks'].append({
                    'title': article['title'],
                    'description': article['url'],
                    'type': 'article'
                })
            for book in filter_relevant_books(books, keyword, 3):  # Fewest books
                plan['tasks'].append({
                    'title': book['title'],
                    'description': book['url'],
                    'type': 'book'
                })

    # Distribute events evenly over the selected timeframe
    distribute_tasks(plan['tasks'], timeframe)

    return jsonify(plan)

def filter_relevant_articles(articles, keyword, limit):
    relevant_articles = []
    for article in articles:
        if keyword.lower() in article['title'].lower() or keyword.lower() in article['description'].lower():
            relevant_articles.append(article)
        if len(relevant_articles) >= limit:
            break
    return relevant_articles

def filter_relevant_books(books, keyword, limit):
    relevant_books = []
    for book in books:
        if keyword.lower() in book['title'].lower() or keyword.lower() in book['description'].lower():
            relevant_books.append(book)
        if len(relevant_books) >= limit:
            break
    return relevant_books

def distribute_tasks(tasks, timeframe):
    current_date = datetime.now()
    end_date = current_date

    if timeframe == '1 day':
        end_date += timedelta(days=1)
    elif timeframe == '3 days':
        end_date += timedelta(days=3)
    elif timeframe == '1 week':
        end_date += timedelta(weeks=1)
    elif timeframe == '2 weeks':
        end_date += timedelta(weeks=2)
   
    total_days = (end_date - current_date).days
    max_events_per_day = 3
    total_slots = total_days * max_events_per_day

    if len(tasks) > total_slots:
        tasks = tasks[:total_slots]  # Limit to the maximum number of slots available

    slots = []
    for day in range(total_days):
        for slot in range(max_events_per_day):
            slots.append(current_date + timedelta(days=day, hours=slot * 2))

    for i, task in enumerate(tasks):
        start_time = slots[i]
        if task['type'] == 'article':
            end_time = start_time + timedelta(minutes=15)
        elif task['type'] == 'book':
            end_time = start_time + timedelta(hours=1)
        else:
            end_time = start_time + timedelta(hours=1)
        task['startDateTime'] = start_time.isoformat()
        task['endDateTime'] = end_time.isoformat()

def get_ted_talks(query):
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}+TED+talk&key={YOUTUBE_API_KEY}&maxResults=10"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching videos: {response.status_code}, {response.text}")
        return []
    videos = response.json().get('items', [])
    video_list = [{'title': video['snippet']['title'], 'description': video['snippet']['description'], 'url': f"https://www.youtube.com/watch?v={video['id']['videoId']}"} for video in videos if video['id']['kind'] == 'youtube#video']
    return video_list

def get_news(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching news: {response.status_code}, {response.text}")
        return []
    articles = response.json().get('articles', [])
    news_list = [{'title': article.get('title'), 'description': article.get('description'), 'url': article.get('url')} for article in articles]
    return news_list

def get_books(query):
    url = f"https://openlibrary.org/search.json?q={query}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching books: {response.status_code}, {response.text}")
        return []
    books = response.json().get('docs', [])
    book_list = [{'title': book.get('title'), 'description': ', '.join(book.get('author_name', ['Unknown Author'])), 'url': f"https://openlibrary.org{book.get('key')}"} for book in books]
    return book_list

def get_events(query):
    url = f"{EVENTBRITE_API_URL}&q={query}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching events: {response.status_code}, {response.text}")
        return []
    events = response.json().get('events', [])
    event_list = [{'name': event.get('name', {}).get('text'), 'description': event.get('description', {}).get('text'), 'url': event.get('url')} for event in events]
    return event_list

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)