# Hello World — AI Travel Itinerary Planner

Hello World is an AI-powered travel and event discovery app built with Python and Streamlit. It helps users find nearby events and restaurants, then builds a personalized day itinerary through a conversational chatbot interface.

This project was built as part of a semester-long engineering course at UT Dallas and used Claude (Anthropic) as an AI assistant during development.

---

## Features

- **AI Chatbot Itinerary Planner** — Chat with a travel assistant to discover events and restaurants and generate a full day schedule
- **Live Event Search** — Pulls real events near you using the Ticketmaster API
- **Restaurant Recommendations** — Uses TF-IDF vectorization and collaborative filtering to match your cuisine preferences with nearby restaurants
- **Geolocation** — Automatically detects your location via browser and reverse geocodes it to a city/state
- **Interactive Maps** — Visualizes your location and restaurant recommendations using PyDeck
- **Dynamic Itinerary Management** — Add, remove, and view itinerary items through chat or quick action buttons

---

## Tech Stack

| Layer | Tools |
|---|---|
| Frontend | Streamlit |
| Language | Python |
| ML / Recommendations | Scikit-learn (TF-IDF, cosine similarity), K-means clustering |
| Maps | PyDeck, Geopy |
| APIs | Ticketmaster Discovery API, Nominatim (reverse geocoding) |
| LLM | LLaMA 3 (via Ollama) |
| AI Development Assistant | Claude (Anthropic) |

---

## Pages

### Home / Itinerary Planner
The main page. Chat with the travel assistant to:
- Search for events by type (concerts, sports, comedy, etc.)
- Get restaurant recommendations by cuisine and budget
- Generate a full day itinerary
- Add or remove specific items from your schedule

### Events
Search for nearby events using a keyword and radius slider. Results are pulled from the Ticketmaster API and sorted by date.

### Restaurant
Enter your cuisine preferences and price range to get personalized restaurant recommendations powered by TF-IDF cosine similarity and location-based scoring.

---

## Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/sanspandey45/Hello-World-Travel-App.git
cd Hello-World-Travel-App
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

### API Keys
This project uses the Ticketmaster Discovery API. To use your own key, replace the `API_KEY` value in `app.py`:

```python
API_KEY = "your_ticketmaster_api_key_here"
```

You can get a free API key at [developer.ticketmaster.com](https://developer.ticketmaster.com).

---

## AI Development

This project was built with assistance from Claude by Anthropic, which helped with:
- Debugging API integration and data parsing
- Structuring the chatbot intent detection logic
- Improving the itinerary generation algorithm
- Code cleanup and documentation

---

## Team

Built by students at the University of Texas at Dallas.

---

## License

This project is for educational purposes.
