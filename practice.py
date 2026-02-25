import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import pydeck as pdk
import requests
from streamlit_javascript import st_javascript
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import numpy as np
from datetime import datetime, timedelta
import re
import pydeck as pdk
from geopy.distance import geodesic
from streamlit_javascript import st_javascript


from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

@st.cache_resource
def get_llm():
    return Ollama(model=DEFAULT_MODEL)

    # Enhanced SYSTEM_PROMPT with memory capabilities
SYSTEM_PROMPT = """
You are a helpful travel assistant helping users plan their itinerary. 
You have access to the following information:

Current Location: {location}
User Preferences:
- Budget: {budget}
- Cuisine Preference: {cuisine}
- Event Interests: {event_interests}
- Search Radius: {radius} miles

Current Itinerary:
{itinerary}

Previous Conversation:
{chat_history}

When responding:
1. For event requests, respond with: "event_request:<event_type>"
2. For restaurant requests, respond with: "food_request:<cuisine>"
3. For itinerary creation, respond with: "itinerary_request"
4. Keep responses friendly and concise
5. Maintain context from previous messages
6. For general chat, provide helpful responses based on the conversation history
"""


# --- Ticketmaster API Key ---
API_KEY = "sPTGoDBnMjr6gfs9TqQYd4FomA5oDBYC"
BASE_URL = "https://app.ticketmaster.com/discovery/v2/events.json"

# --- Ollama Model Configuration ---
DEFAULT_MODEL = "llama3"

# --- Reverse Geocoding Function ---
def reverse_geocode(lat, lon):
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
    headers = {"User-Agent": "NearbyEventsApp/1.0 (your_email@example.com)"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        address = data.get("address", {})
        city = address.get("city") or address.get("town") or address.get("village")
        state = address.get("state")
        country = address.get("country")
        full_location = f"{city}, {state}, {country}" if city and state and country else None
        return full_location
    else:
        return None

# --- Page Setup ---
if "page" not in st.session_state:
    st.session_state.page = "itinerary"

query_params = st.query_params
page = query_params.get("page", st.session_state.page)
st.session_state.page = page
# this is my comment
# --- Location Fetching Function ---
def fetch_location():
    if "location" not in st.session_state:
        coords = st_javascript("""await new Promise((resolve, reject) => {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    resolve({
                        coords: {
                            latitude: position.coords.latitude,
                            longitude: position.coords.longitude
                        }
                    });
                },
                (error) => {
                    resolve({error: error.message});
                }
            );
        });""")

        if coords and "coords" in coords:
            st.session_state.location = coords["coords"]

            lat = coords["coords"]["latitude"]
            lon = coords["coords"]["longitude"]

            # Reverse geocode city
            full_location = reverse_geocode(lat, lon)

            if full_location:
                st.session_state.location_details = full_location
                st.success(f"📍 Location saved: {full_location}")
            else:
                st.success(f"📍 Location saved: {lat:.4f}, {lon:.4f} (Location unknown)")
        else:
            st.warning("⚠️ Waiting for geolocation or permission denied.")

#------------------------------ AI ITINERARY PAGE WITH INTEGRATED CHATBOT------------------------------------------------------
if st.session_state.page == "itinerary":
    st.title("🌎 Hello World")
    st.title("AI Itinerary Planner")
    fetch_location()  # Ensure location is fetched on this page

    # Initialize session state variables for chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "user_preferences" not in st.session_state:
        st.session_state.user_preferences = {
            "interests": [],
            "budget": "medium",
            "start_time": "09:00",
            "event_type": "",
            "cuisine_preference": "",
            "radius": 25  # Default radius in miles
        }

    # Initialize current itinerary in session state if not exists
    if "current_itinerary" not in st.session_state:
        st.session_state.current_itinerary = []

    # Display location if available
    if "location" in st.session_state and "location_details" in st.session_state:
        st.markdown(f"📍 **Planning for:** {st.session_state.location_details}")

    # Function to extract and update the current itinerary from the bot's response
    def update_current_itinerary_from_text(response_text):
        # Check if this is an itinerary response
        if "Here's your personalized itinerary" not in response_text:
            return

        # Clear the current itinerary
        st.session_state.current_itinerary = []

        # Extract itinerary items using regex
        pattern = r"\*\*([\d:]+\s[AP]M)\*\*\s-\s(.+?)(?=\n\n|\Z)"
        matches = re.finditer(pattern, response_text)

        for match in matches:
            time = match.group(1)
            activity = match.group(2)
            st.session_state.current_itinerary.append({
                "time": time,
                "activity": activity
            })

        # Sort itinerary by time
        def time_to_minutes(time_str):
            # Convert time like "9:00 AM" to minutes since midnight
            try:
                return datetime.strptime(time_str, "%I:%M %p").hour * 60 + datetime.strptime(time_str, "%I:%M %p").minute
            except:
                return 0

        st.session_state.current_itinerary.sort(key=lambda x: time_to_minutes(x["time"]))

    # Function to get events from Ticketmaster API - reusing your existing code
    def get_events(keyword, lat, lon, radius):
        if not keyword:
            return []

        params = {
            "apikey": API_KEY,
            "keyword": keyword,
            "latlong": f"{lat},{lon}",
            "radius": radius,
            "unit": "miles"
        }

        try:
            response = requests.get(BASE_URL, params=params)
            data = response.json()
            events = data.get("_embedded", {}).get("events", [])
            return events
        except Exception as e:
            st.error(f"Error retrieving events: {e}")
            return []

    # Mock restaurant data since the restaurant function is not working
    mock_restaurants = {
        "italian": [
            {"name": "Bella Pasta", "rating": 4.5, "price": "medium", "distance": 1.2},
            {"name": "Romano's Trattoria", "rating": 4.3, "price": "high", "distance": 2.5},
            {"name": "Pizza Palace", "rating": 4.1, "price": "low", "distance": 0.8}
        ],
        "chinese": [
            {"name": "Golden Dragon", "rating": 4.4, "price": "medium", "distance": 1.7},
            {"name": "Szechuan Garden", "rating": 4.6, "price": "high", "distance": 3.2},
            {"name": "Quick Wok", "rating": 3.9, "price": "low", "distance": 0.5}
        ],
        "mexican": [
            {"name": "Taco Fiesta", "rating": 4.2, "price": "low", "distance": 1.3},
            {"name": "El Mariachi", "rating": 4.7, "price": "high", "distance": 4.1},
            {"name": "Burrito Express", "rating": 4.0, "price": "medium", "distance": 2.1}
        ],
        "vegan": [
            {"name": "Green Leaf Cafe", "rating": 4.8, "price": "medium", "distance": 2.3},
            {"name": "Plant Power", "rating": 4.5, "price": "high", "distance": 3.8},
            {"name": "Veggie Delight", "rating": 4.2, "price": "low", "distance": 1.9}
        ],
        "japanese": [
            {"name": "Sushi Samba", "rating": 4.6, "price": "high", "distance": 2.8},
            {"name": "Tokyo Express", "rating": 4.0, "price": "low", "distance": 1.5},
            {"name": "Sakura Japanese", "rating": 4.4, "price": "medium", "distance": 3.1}
        ],
        "indian": [
            {"name": "Taj Mahal", "rating": 4.5, "price": "medium", "distance": 2.4},
            {"name": "Spicy Indian Kitchen", "rating": 4.7, "price": "high", "distance": 3.7},
            {"name": "Curry Express", "rating": 4.1, "price": "low", "distance": 1.2}
        ]
    }

    # Default to a general list if specific cuisine not found
    general_restaurants = [
        {"name": "The Local Diner", "rating": 4.3, "price": "medium", "distance": 1.5},
        {"name": "City View Restaurant", "rating": 4.6, "price": "high", "distance": 2.8},
        {"name": "Quick Bites", "rating": 4.0, "price": "low", "distance": 0.7},
        {"name": "Riverside Cafe", "rating": 4.4, "price": "medium", "distance": 2.1},
        {"name": "Urban Kitchen", "rating": 4.2, "price": "medium", "distance": 1.8}
    ]

    # Function to generate response based on user message
    def get_chatbot_response(user_message):
        message_lower = user_message.lower()

        # Check for itinerary viewing intent
        itinerary_view_keywords = ["show itinerary", "view itinerary", "see itinerary", "my itinerary",
                                "current itinerary", "what's in my itinerary", "what is in my itinerary"]
        if any(keyword in message_lower for keyword in itinerary_view_keywords):
            return "itinerary_view"

        # Handle greetings
        greetings = ["hi", "hello", "hey", "howdy", "greetings"]
        if any(greeting in message_lower for greeting in greetings) and len(message_lower) < 10:
            return "Hello! I'm your itinerary assistant. I can help you find events and restaurants near you. What kind of activities are you interested in?"

        # Handle event interest
        event_types = {
            "concert": ["concert", "music", "band", "show", "performance", "live music", "gig"],
            "sports": ["sports", "game", "match", "basketball", "football", "baseball", "hockey", "soccer"],
            "arts": ["art", "exhibition", "gallery", "museum", "theater", "cultural"],
            "family": ["family", "children", "kids", "zoo", "park", "aquarium"],
            "comedy": ["comedy", "laugh", "funny", "comedian", "stand-up"],
            "festival": ["festival", "fair", "carnival"]
        }

        # Check if user is asking about events
        event_keywords = ["event", "happening", "going on", "show", "concert", "game"]
        is_asking_events = any(keyword in message_lower for keyword in event_keywords)

        # Check if user is asking about food
        food_keywords = ["food", "restaurant", "eat", "dining", "lunch", "dinner", "breakfast", "cuisine"]
        is_asking_food = any(keyword in message_lower for keyword in food_keywords)

        # Check if user is requesting an itinerary
        itinerary_keywords = ["itinerary", "plan", "schedule", "day", "both", "everything", "suggest"]
        is_asking_itinerary = any(keyword in message_lower for keyword in itinerary_keywords)

        # Detect cuisine preferences
        cuisine_types = ["chinese", "italian", "mexican", "indian", "japanese", "thai", "french", "american",
                        "mediterranean", "vegan", "vegetarian", "seafood", "steakhouse", "fast food", "pizza"]
        detected_cuisines = [cuisine for cuisine in cuisine_types if cuisine in message_lower]

        # Detect budget preferences
        budget_patterns = {
            "low": ["cheap", "inexpensive", "budget", "affordable", "low cost"],
            "high": ["expensive", "fancy", "high-end", "luxury", "fine dining", "upscale"]
        }

        detected_budget = None
        for budget, patterns in budget_patterns.items():
            if any(pattern in message_lower for pattern in patterns):
                detected_budget = budget
                st.session_state.user_preferences["budget"] = budget

        # Detect event type interests
        detected_event_type = None
        for event_type, patterns in event_types.items():
            if any(pattern in message_lower for pattern in patterns):
                detected_event_type = event_type
                st.session_state.user_preferences["event_type"] = event_type

        # Detect radius changes
        radius_patterns = [
            (r'(\d+)\s*miles', lambda m: int(m.group(1))),
            (r'within\s*(\d+)', lambda m: int(m.group(1))),
            (r'(\d+)\s*mile radius', lambda m: int(m.group(1)))
        ]

        for pattern, extractor in radius_patterns:
            match = re.search(pattern, message_lower)
            if match:
                try:
                    st.session_state.user_preferences["radius"] = extractor(match)
                except:
                    pass

        # Update cuisine preference if detected
        if detected_cuisines:
            st.session_state.user_preferences["cuisine_preference"] = detected_cuisines[0]

        # Generate appropriate response based on detected intent

        # Case 1: User wants an itinerary with both events and food
        if is_asking_itinerary:
            return "itinerary_request"

        # Case 2: User is interested in events
        elif is_asking_events or detected_event_type:
            event_keyword = detected_event_type if detected_event_type else "events"
            return f"event_request:{event_keyword}"

        # Case 3: User is interested in food
        elif is_asking_food or detected_cuisines:
            cuisine = st.session_state.user_preferences.get("cuisine_preference", "")
            return f"food_request:{cuisine}"

        # Default response if no specific intent is detected
        return "I can help you find events and restaurants in your area. What are you interested in exploring? You can ask about concerts, sports events, or various cuisines."

    def handle_event_request(event_type):
        # Ensure we have location
        if "location" not in st.session_state:
            return "I need your location to find events near you. Please make sure location access is enabled."

        lat = st.session_state.location['latitude']
        lon = st.session_state.location['longitude']
        radius = st.session_state.user_preferences.get("radius", 25)

        # Get events from Ticketmaster API
        events = get_events(event_type, lat, lon, radius)

        if not events:
            return f"I couldn't find any {event_type} events near you within {radius} miles. Would you like to try a broader search or a different type of event?"

        # Format response with events
        response = f"I found {len(events)} {event_type} events near you:\n\n"

        # Sort events by date (closest first)
        sorted_events = sorted(events, key=lambda x: x["dates"]["start"].get("localDate"))

        # Display up to 5 events
        for i, event in enumerate(sorted_events[:5]):
            name = event.get("name")
            venue = event["_embedded"]["venues"][0]["name"]
            date = event["dates"]["start"].get("localDate")
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            formatted_date = date_obj.strftime("%A, %B %d")

            response += f"**{i+1}. {name}**\n"
            response += f"📍 {venue}\n"
            response += f"📅 {formatted_date}\n"
            response += f"*Reply with \"Add event {i+1}\" to add this to your itinerary.*\n\n"

        response += "You can also ask for restaurants or say \"Create itinerary\" to build a complete schedule."

        # Store events for itinerary creation
        st.session_state["found_events"] = sorted_events[:5]

        return response

    # Function to handle food requests with selection options
    def handle_food_request(cuisine):
        # Ensure we have location
        if "location" not in st.session_state:
            return "I need your location to find restaurants near you. Please make sure location access is enabled."

        # Get user's budget preference
        user_budget = st.session_state.user_preferences.get("budget", "medium")

        # Select restaurants based on cuisine
        selected_restaurants = mock_restaurants.get(cuisine.lower(), None)

        # If cuisine not found or not specified, use general list
        if not selected_restaurants:
            selected_restaurants = general_restaurants
            cuisine = "popular"

        # Filter by budget if specified
        if user_budget != "medium":  # medium is default, so no filtering needed
            filtered = [r for r in selected_restaurants if r["price"] == user_budget]
            # Only use filtered list if it's not empty
            if filtered:
                selected_restaurants = filtered

        # Format response with restaurants
        response = f"Here are some {cuisine} restaurant recommendations near you:\n\n"

        for i, restaurant in enumerate(selected_restaurants[:5]):
            response += f"**{i+1}. {restaurant['name']}**\n"
            response += f"⭐ Rating: {restaurant['rating']}\n"
            response += f"💵 Price: {restaurant['price'].capitalize()}\n"
            response += f"📍 Distance: {restaurant['distance']:.1f} km\n"
            response += f"*Reply with \"Add restaurant {i+1}\" to add this to your itinerary.*\n\n"

        response += "You can also ask for events or say \"Create itinerary\" to build a complete schedule."

        # Store restaurants for itinerary creation
        st.session_state["found_restaurants"] = selected_restaurants[:5]

        return response

    # Function to create a complete itinerary
    def create_full_itinerary():
        # Check if we have preferences
        event_type = st.session_state.user_preferences.get("event_type", "")
        cuisine = st.session_state.user_preferences.get("cuisine_preference", "")

        # Ensure we have location
        if "location" not in st.session_state:
            return "I need your location to create your itinerary. Please make sure location access is enabled."

        # Initialize itinerary
        itinerary = []
        start_time = datetime.strptime(st.session_state.user_preferences.get("start_time", "09:00"), "%H:%M")
        current_time = start_time

        # Get events if not already searched
        events = []
        if "found_events" in st.session_state and st.session_state["found_events"]:
            events = st.session_state["found_events"]
        elif event_type:
            lat = st.session_state.location['latitude']
            lon = st.session_state.location['longitude']
            radius = st.session_state.user_preferences.get("radius", 25)
            events_results = get_events(event_type, lat, lon, radius)
            if events_results:
                # Sort events by date
                events = sorted(events_results, key=lambda x: x["dates"]["start"].get("localDate"))[:2]

        # Get restaurant recommendations if not already searched
        restaurants = []
        if "found_restaurants" in st.session_state and st.session_state["found_restaurants"]:
            restaurants = st.session_state["found_restaurants"]
        elif cuisine:
            # Get user's budget preference
            user_budget = st.session_state.user_preferences.get("budget", "medium")

            # Select restaurants based on cuisine
            selected_restaurants = mock_restaurants.get(cuisine.lower(), general_restaurants)

            # Filter by budget if specified
            if user_budget != "medium":  # medium is default, so no filtering needed
                filtered = [r for r in selected_restaurants if r["price"] == user_budget]
                # Only use filtered list if it's not empty
                if filtered:
                    selected_restaurants = filtered

            restaurants = selected_restaurants[:2]
        else:
            # If no cuisine specified, use general restaurants
            restaurants = general_restaurants[:2]

        # Add morning activity if we have an early start time
        if current_time.hour < 10:
            morning_activity = "Start your day with a relaxing walk at Prairie Creek Park"
            itinerary.append({
                "time": current_time.strftime("%I:%M %p"),
                "activity": morning_activity
            })
            current_time += timedelta(hours=1, minutes=30)

        # Add events to itinerary
        for event in events:
            event_time = event["dates"]["start"].get("localTime")
            if event_time:
                try:
                    event_datetime = datetime.strptime(event_time, "%H:%M:%S")
                    hour, minute = event_datetime.hour, event_datetime.minute
                    # Only replace time if event is later than current time
                    if hour > current_time.hour or (hour == current_time.hour and minute > current_time.minute):
                        current_time = current_time.replace(hour=hour, minute=minute)
                except:
                    # If we can't parse the event time, just use current time
                    pass

            itinerary.append({
                "time": current_time.strftime("%I:%M %p"),
                "activity": f"{event.get('name')} at {event['_embedded']['venues'][0]['name']}"
            })

            # Add 2 hours for event duration
            current_time += timedelta(hours=2)

        # Add lunch around noon if we have restaurants
        lunch_time = start_time.replace(hour=12, minute=0)
        if current_time < lunch_time and restaurants:
            current_time = lunch_time
            itinerary.append({
                "time": current_time.strftime("%I:%M %p"),
                "activity": f"Lunch at {restaurants[0]['name']}"
            })
            current_time += timedelta(hours=1, minutes=30)
            restaurants = restaurants[1:] if len(restaurants) > 1 else []

        # Add afternoon activity if we have a gap
        afternoon_time = start_time.replace(hour=14, minute=0)
        if current_time < afternoon_time and current_time.hour >= 12:
            if event_type:
                activity = f"Explore {event_type} related activities"
            else:
                activity = "Visit the City Museum or Historic Downtown"

            itinerary.append({
                "time": current_time.strftime("%I:%M %p"),
                "activity": activity
            })
            current_time += timedelta(hours=2)

        # Add dinner around 6pm if we have restaurants left
        dinner_time = start_time.replace(hour=18, minute=0)
        if restaurants:
            if current_time < dinner_time:
                current_time = dinner_time

            itinerary.append({
                "time": current_time.strftime("%I:%M %p"),
                "activity": f"Dinner at {restaurants[0]['name']}"
            })
            current_time += timedelta(hours=1, minutes=30)


        # Sort itinerary by time
        itinerary = sorted(itinerary, key=lambda x: datetime.strptime(x["time"], "%I:%M %p"))

        if not itinerary:
            return "I couldn't create an itinerary based on the available information. Please try specifying event types or cuisines you're interested in."

        # Format itinerary as text
        itinerary_text = "📋 **Here's your personalized itinerary:**\n\n"
        for item in itinerary:
            itinerary_text += f"**{item['time']}** - {item['activity']}\n\n"

        itinerary_text += "How does this look? I can help you refine it or find different options."

        # Update the session state for sidebar display
        st.session_state.current_itinerary = itinerary

        return itinerary_text

    def add_event_to_itinerary(event_index):
        if "found_events" not in st.session_state or not st.session_state["found_events"]:
            return "I don't have any events to add. Please search for events first."

        try:
            index = int(event_index) - 1
            if index < 0 or index >= len(st.session_state["found_events"]):
                return f"Please select a valid event number between 1 and {len(st.session_state['found_events'])}."

            selected_event = st.session_state["found_events"][index]

            # Check if we already have this event in the itinerary
            event_name = selected_event.get("name")
            for item in st.session_state.current_itinerary:
                if event_name in item["activity"]:
                    return f"'{event_name}' is already in your itinerary."

            # Determine an appropriate time for the event
            event_time = selected_event["dates"]["start"].get("localTime")
            if event_time:
                try:
                    event_datetime = datetime.strptime(event_time, "%H:%M:%S")
                    formatted_time = event_datetime.strftime("%I:%M %p")
                except:
                    # Default to evening if time parse fails
                    formatted_time = "07:00 PM"
            else:
                # Default to evening if no time provided
                formatted_time = "07:00 PM"

            # Add to itinerary
            venue = selected_event["_embedded"]["venues"][0]["name"]
            new_item = {
                "time": formatted_time,
                "activity": f"{event_name} at {venue}"
            }

            st.session_state.current_itinerary.append(new_item)

            # Sort itinerary by time
            def time_to_minutes(time_str):
                try:
                    return datetime.strptime(time_str, "%I:%M %p").hour * 60 + datetime.strptime(time_str, "%I:%M %p").minute
                except:
                    return 0

            st.session_state.current_itinerary.sort(key=lambda x: time_to_minutes(x["time"]))

            return f"Added '{event_name}' to your itinerary at {formatted_time}. Would you like to add anything else?"

        except Exception as e:
            return f"Sorry, I couldn't add that event. Please try again."

    # Function to add a specific restaurant to the itinerary
    def add_restaurant_to_itinerary(restaurant_index):
        if "found_restaurants" not in st.session_state or not st.session_state["found_restaurants"]:
            return "I don't have any restaurants to add. Please search for restaurants first."

        try:
            index = int(restaurant_index) - 1
            if index < 0 or index >= len(st.session_state["found_restaurants"]):
                return f"Please select a valid restaurant number between 1 and {len(st.session_state['found_restaurants'])}."

            selected_restaurant = st.session_state["found_restaurants"][index]

            # Check if we already have this restaurant in the itinerary
            restaurant_name = selected_restaurant["name"]
            for item in st.session_state.current_itinerary:
                if restaurant_name in item["activity"]:
                    return f"'{restaurant_name}' is already in your itinerary."

            # Determine if it should be lunch or dinner based on existing itinerary
            lunch_exists = any("Lunch" in item["activity"] for item in st.session_state.current_itinerary)
            dinner_exists = any("Dinner" in item["activity"] for item in st.session_state.current_itinerary)

            if not lunch_exists:
                meal_type = "Lunch"
                formatted_time = "12:30 PM"
            elif not dinner_exists:
                meal_type = "Dinner"
                formatted_time = "07:00 PM"
            else:
                # Both meals exist, make it a generic meal
                meal_type = "Visit"
                formatted_time = "03:00 PM"

            # Add to itinerary
            new_item = {
                "time": formatted_time,
                "activity": f"{meal_type} at {restaurant_name}"
            }

            st.session_state.current_itinerary.append(new_item)

            # Sort itinerary by time
            def time_to_minutes(time_str):
                try:
                    return datetime.strptime(time_str, "%I:%M %p").hour * 60 + datetime.strptime(time_str, "%I:%M %p").minute
                except:
                    return 0

            st.session_state.current_itinerary.sort(key=lambda x: time_to_minutes(x["time"]))

            return f"Added '{restaurant_name}' to your itinerary as {meal_type} at {formatted_time}. Would you like to add anything else?"

        except Exception as e:
            return f"Sorry, I couldn't add that restaurant. Please try again."


    # Function to remove a specific item from the itinerary
    def remove_from_itinerary(item_index):
        try:
            # Check if we have any items in the itinerary
            if not st.session_state.current_itinerary:
                return "Your itinerary is already empty. There's nothing to remove."

            # Convert the index to integer and adjust for 1-based indexing
            index = int(item_index) - 1

            # Check if the index is valid
            if index < 0 or index >= len(st.session_state.current_itinerary):
                return f"Please select a valid itinerary item number between 1 and {len(st.session_state.current_itinerary)}."

            # Get the item to be removed
            removed_item = st.session_state.current_itinerary[index]
            removed_activity = removed_item["activity"]

            # Remove the item
            st.session_state.current_itinerary.pop(index)

            return f"Removed '{removed_activity}' from your itinerary. Would you like to add something else instead?"

        except ValueError:
            return "Please provide a valid number to remove an item from your itinerary."
        except Exception as e:
            return f"Sorry, I couldn't remove that item. Please try again."

    # Function to display the current itinerary with removal options
    def display_current_itinerary():
        if not st.session_state.current_itinerary:
            return "Your itinerary is currently empty. Let's add some activities!"

        response = "📋 **Your Current Itinerary:**\n\n"

        # Display each item with an option to remove it
        for i, item in enumerate(st.session_state.current_itinerary):
            response += f"**{i+1}. {item['time']}** - {item['activity']}\n"
            response += f"*Reply with \"Remove {i+1}\" to remove this item.*\n\n"

        response += "You can continue adding events or restaurants, or say \"Create itinerary\" to fill in any gaps."
        return response

    # Update the handle_chat_input function to include removal commands
    def handle_chat_input():
        user_message = st.session_state.chat_input

        if not user_message:
            return

        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_message})

        # Check for specific commands
        message_lower = user_message.lower()

        # Check for "show itinerary" command
        if "show itinerary" in message_lower or "view itinerary" in message_lower or "my itinerary" in message_lower or "itinerary" in message_lower:
            bot_response = display_current_itinerary()
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            st.session_state.chat_input = ""
            return

        # Check for "add event X" command
        event_add_match = re.search(r"add event (\d+)", message_lower)
        if event_add_match:
            event_index = event_add_match.group(1)
            bot_response = add_event_to_itinerary(event_index)
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            st.session_state.chat_input = ""
            return

        # Check for "add restaurant X" command
        restaurant_add_match = re.search(r"add restaurant (\d+)", message_lower)
        if restaurant_add_match:
            restaurant_index = restaurant_add_match.group(1)
            bot_response = add_restaurant_to_itinerary(restaurant_index)
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            st.session_state.chat_input = ""
            return

        # Check for "remove X" command
        remove_match = re.search(r"remove (\d+)", message_lower)
        if remove_match:
            item_index = remove_match.group(1)
            bot_response = remove_from_itinerary(item_index)
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            st.session_state.chat_input = ""
            return

        # Process normal chatbot responses
        else:
            response = get_chatbot_response(user_message)

            # Process different response types
            if response.startswith("event_request:"):
                event_type = response.split(":")[1]
                bot_response = handle_event_request(event_type)
            elif response.startswith("food_request:"):
                cuisine = response.split(":")[1]
                bot_response = handle_food_request(cuisine)
            elif response == "itinerary_request":
                bot_response = create_full_itinerary()
                # Extract itinerary from text for sidebar display
            else:
                # Only use Llama 3 if no command is detected
                llm = get_llm()
                prompt_template = ChatPromptTemplate.from_template("""
                You are a travel assistant. Respond with one of these exact formats:
                - For events: "event_request:<event_type>"
                - For restaurants: "food_request:<cuisine>"
                - For itinerary: "itinerary_request"
                - For general help: "<your response>"
                
                NEVER make up events or details. Always use the command formats above.
                
                Location: {location}
                Budget: {budget}
                Cuisine: {cuisine}
                Interests: {event_interests}
                Radius: {radius} miles
                
                Conversation:
                {chat_history}
                
                User: {message}
                """)
                
                context = {
                    "location": st.session_state.get("location_details", "unknown location"),
                    "budget": st.session_state.user_preferences.get("budget", "not specified"),
                    "cuisine": st.session_state.user_preferences.get("cuisine_preference", "not specified"),
                    "event_interests": st.session_state.user_preferences.get("event_type", "not specified"),
                    "radius": st.session_state.user_preferences.get("radius", 25),
                    "chat_history": "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history[-6:]]),
                    "message": user_message
                }
                
                try:
                    chain = prompt_template | llm | StrOutputParser()
                    llm_response = chain.invoke(context)
                    
                    if llm_response.startswith("event_request:"):
                        event_type = llm_response.split(":")[1].strip()
                        bot_response = handle_event_request(event_type)
                    elif llm_response.startswith("food_request:"):
                        cuisine = llm_response.split(":")[1].strip()
                        bot_response = handle_food_request(cuisine)
                    elif llm_response == "itinerary_request":
                        bot_response = create_full_itinerary()
                        update_current_itinerary_from_text(bot_response)
                    else:
                        bot_response = llm_response
                        
                except Exception as e:
                    st.error(f"Error getting LLM response: {e}")
                    bot_response = "I encountered an error. Please try again."

        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

        # Clear the input box
        st.session_state.chat_input = ""


    # Layout the chat interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Chat with Your Travel Assistant")

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div style='background-color: #E8F0FE; color: #000; padding: 10px; border-radius: 5px; margin-bottom: 10px; display: flex; justify-content: flex-end;'>
                        <div style='max-width: 80%;'>
                            <p style='margin: 0;'>{message['content']}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background-color: #F0F2F6; color: #000; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                        <div style='max-width: 80%;'>
                            <p style='margin: 0;'>{message['content']}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Chat input using callback method
        st.text_input("Type your message here:",
                      placeholder="E.g., Find me concerts near here",
                      key="chat_input",
                      on_change=handle_chat_input)

    # Right sidebar with quick actions at the top and other info below
    with col2:
        # Quick action buttons at the top
        st.subheader("Quick Actions")

        def find_events():
            if "location" in st.session_state:
                lat = st.session_state.location['latitude']
                lon = st.session_state.location['longitude']
                event_type = st.session_state.user_preferences.get("event_type", "events")
                response = handle_event_request(event_type)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": "I need your location to find events. Please allow location access."})

        def find_restaurants():
            if "location" in st.session_state:
                cuisine = st.session_state.user_preferences.get("cuisine_preference", "")
                response = handle_food_request(cuisine)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": "I need your location to find restaurants. Please allow location access."})

        def generate_itinerary():
            response = create_full_itinerary()
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            # Update the current itinerary display
            update_current_itinerary_from_text(response)

        def display_itinerary():
            response = display_current_itinerary()
            st.session_state.chat_history.append({"role": "assistant", "content": response})

        def clear_itinerary():
            st.session_state.current_itinerary = []
            st.session_state.chat_history.append({"role": "assistant", "content": "I've cleared your itinerary. Let's start fresh!"})

        def reset_preferences():
            st.session_state.user_preferences = {
                "interests": [],
                "budget": "medium",
                "start_time": "09:00",
                "event_type": "",
                "cuisine_preference": "",
                "radius": 25
            }
            st.session_state.chat_history = []
            st.session_state.current_itinerary = []
            if "found_events" in st.session_state:
                del st.session_state["found_events"]
            if "found_restaurants" in st.session_state:
                del st.session_state["found_restaurants"]

        # First row of buttons
        col1, col2 = st.columns(2)
        with col1:
            st.button("Find Events", on_click=find_events, use_container_width=True)
            st.button("Create Itinerary", on_click=generate_itinerary, use_container_width=True)

        with col2:
            st.button("Find Restaurants", on_click=find_restaurants, use_container_width=True)
            st.button("Clear Itinerary", on_click=clear_itinerary, use_container_width=True)

        # Second row of buttons
        col1, col2 = st.columns(2)
        with col1:
            st.button("Reset All", on_click=reset_preferences, use_container_width=True,
                    help="Reset all preferences and clear the chat history")

        st.divider()

        # Current Itinerary Section
        st.subheader("📋 Current Itinerary")

        # Display current itinerary
        if st.session_state.current_itinerary:
            for item in st.session_state.current_itinerary:
                st.markdown(f"**{item['time']}** - {item['activity']}")
        else:
            st.info("No itinerary items yet. Use the buttons above or chat with the assistant to build your perfect day!")

        st.divider()

        # Preferences Section
        st.subheader("Your Preferences")

        # Display current preferences
        st.write(f"**Event type:** {st.session_state.user_preferences.get('event_type', 'Not specified')}")
        st.write(f"**Cuisine:** {st.session_state.user_preferences.get('cuisine_preference', 'Not specified')}")
        st.write(f"**Budget:** {st.session_state.user_preferences.get('budget', 'Medium').capitalize()}")
        st.write(f"**Search radius:** {st.session_state.user_preferences.get('radius', 25)} miles")
        st.write(f"**Start time:** {st.session_state.user_preferences.get('start_time', '09:00')}")

        st.divider()

        # Tips section
        st.subheader("Usage Tips")
        st.markdown("""
        - Ask about events like "concerts" or "sports"
        - Specify cuisines like "italian" or "vegan"
        - Mention your budget preference
        - Say "within 10 miles" to adjust search radius
        - Type "Add event 2" to add specific events
        - Type "Remove 3" to remove items from your itinerary
        """)

    # Initialize chat if it's empty
    if len(st.session_state.chat_history) == 0:
        welcome_message = "👋 Hello! I'm your travel assistant. I can help you find events and restaurants near you, and create a personalized itinerary. What are you interested in exploring today?"
        st.session_state.chat_history.append({"role": "assistant", "content": welcome_message})

#------------------------------------ EVENTS PAGE ------------------------------------------------------
elif st.session_state.page == "events":
    st.title("Events Near You:")

    fetch_location()  # Ensure location is fetched on this page

    # Ensure location is available
    lat = st.session_state.location['latitude']
    lon = st.session_state.location['longitude']
    city = st.session_state.get("city", None)

    # Map display: ----------------------------------------------
    user_location = pd.DataFrame({'lat': [lat], 'lon': [lon]})

    layer = pdk.Layer(
        'ScatterplotLayer',
        data=user_location,
        get_position='[lon, lat]',
        get_color='[255, 0, 0, 160]',
        get_radius=30
    )

    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=15, pitch=0)
    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/streets-v11",
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "You are here"}
        ),
        use_container_width=True
    )

    # Event Search
    st.subheader("🔍 Search Nearby Events")
    keyword = st.text_input("What are you looking for? (e.g. concerts, sports, comedy)")
    radius = st.slider("Radius (miles)", min_value=5, max_value=100, value=25)

    if st.button("Search Events"):
        if keyword:
            params = {
                "apikey": API_KEY,
                "keyword": keyword,
                "latlong": f"{lat},{lon}",
                "radius": radius,
                "unit": "miles"
            }

            response = requests.get(BASE_URL, params=params)
            data = response.json()
            events = data.get("_embedded", {}).get("events", [])

            if events:
                st.success(f"Found {len(events)} event(s) near you!")

                # Sort events by date (closest first)
                sorted_events = sorted(events, key=lambda x: x["dates"]["start"].get("localDate"))

                # Display sorted events
                for event in sorted_events:
                    name = event.get("name")
                    venue = event["_embedded"]["venues"][0]["name"]
                    date = event["dates"]["start"].get("localDate")
                    st.subheader(name)
                    st.write("📍", venue)
                    st.write("📅", date)
                    if event.get("url"):
                        st.markdown(f"[More Info]({event.get('url')})")
                    st.markdown("---")
            else:
                st.info("No events found nearby for that keyword.")
        else:
            st.warning("Please enter a keyword to search.")

# -------------------------- RESTAURANT PAGE --------------------
elif st.session_state.page == "restaurant":
    st.title("🍽️ Restaurant Recommender")

    fetch_location()  # Ensure location is fetched and stored in session state

    if "location" not in st.session_state:
        st.warning("📍 Location not available. Please allow location access from the Home page.")
        st.stop()

    lat = st.session_state.location['latitude']
    lon = st.session_state.location['longitude']
    full_location = st.session_state.get("location_details", "Unknown Location")

    st.markdown(f"📍 **You are in:** {full_location}")

    # --- Load and Prepare Data ---
    DATA_PATH = r"C:\flutter\grubhub.csv"
    data = pd.read_csv(DATA_PATH)

    data.drop(['delivery_fee_raw', 'delivery_fee', 'delivery_time_raw', 'delivery_time', 'service_fee'],
            axis=1, inplace=True, errors='ignore')
    price_categories = ['low', 'medium', 'high']
    if 'prices' not in data.columns:
        data['prices'] = np.random.choice(price_categories, size=len(data))
    data['cuisines'] = data['cuisines'].fillna('')  # Fill NaN cuisines

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['cuisines'])

    def location_similarity(user_location, restaurant_location):
        return 1 / (1 + geodesic(user_location, restaurant_location).km)

    def recommend_restaurants(user_cuisine, user_lat, user_lon, user_price=None, top_n=5):
        tfidf_user = tfidf.transform([user_cuisine])
        cuisine_sim_scores = cosine_similarity(tfidf_user, tfidf_matrix).flatten()

        user_location = (user_lat, user_lon)
        location_sim_scores = np.array([
            location_similarity(user_location, (lat, lon))
            for lat, lon in zip(data['latitude'], data['longitude'])
        ])

        combined_scores = 0.7 * cuisine_sim_scores + 0.3 * location_sim_scores

        filtered_data = data
        filtered_scores = combined_scores
        if user_price:
            price_mask = (data['prices'] == user_price)
            filtered_data = data[price_mask]
            filtered_scores = combined_scores[price_mask.values]

        top_indices = filtered_scores.argsort()[-top_n:][::-1]
        recommended = filtered_data.iloc[top_indices][['loc_name', 'latitude', 'longitude', 'cuisines', 'review_rating', 'prices']]
        recommended['distance_km'] = [
            geodesic(user_location, (lat, lon)).km for lat, lon in zip(recommended['latitude'], recommended['longitude'])
        ]
        return recommended

    user_location_df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=user_location_df,
        get_position='[lon, lat]',
        get_color='[255, 0, 0, 160]',
        get_radius=30
    )
    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=15, pitch=0)
    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/streets-v11",
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "You are here"}
        ),
        use_container_width=True
    )

    with st.form("recommendation_form"):
        user_cuisine = st.text_input("What cuisines are you craving?", "chinese italian")
        user_price = st.selectbox("Price range", ["Any", "low", "medium", "high"])
        top_n = st.slider("Number of recommendations", 1, 10, 5)
        submitted = st.form_submit_button("Find Restaurants")

    recommendations = None

    if submitted:
        recommendations = recommend_restaurants(
            user_cuisine, lat, lon,
            user_price if user_price != "Any" else None,
            top_n
        )

    if submitted and user_cuisine:
        st.subheader("You're craving:")
        cuisine_list = [c.strip().capitalize() for c in user_cuisine.split()]
        cols = st.columns(len(cuisine_list))
        for idx, cuisine in enumerate(cuisine_list):
            with cols[idx]:
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 8px 12px; border-radius: 20px; text-align: center; font-weight: 600; color: #000000">
                    {cuisine}
                </div>
                """, unsafe_allow_html=True)

    if recommendations is not None and not recommendations.empty:
        st.subheader("Top Recommendations")
        for idx, row in recommendations.iterrows():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.map(pd.DataFrame({'lat': [row['latitude']], 'lon': [row['longitude']]}), zoom=15, use_container_width=True)
            with col2:
                st.markdown(f"""
                <div style="background-color: #ffffff; border-radius: 12px; padding: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);  color: #000000">
                    <h3>🍴 {row['loc_name']}</h3>
                    <p><strong>Cuisines:</strong> {row['cuisines']}</p>
                    <p><strong>Rating:</strong> ⭐ {row['review_rating']}</p>
                    <p><strong>Price:</strong> 💵 {row['prices']}</p>
                </div>
                """, unsafe_allow_html=True)

        st.subheader("📋 Summary Table")
        st.dataframe(
            recommendations[['loc_name', 'cuisines', 'review_rating', 'prices', 'distance_km']].rename(
                columns={
                    "loc_name": "Restaurant",
                    "cuisines": "Cuisines",
                    "review_rating": "Rating",
                    "prices": "Price",
                }
            )
        )
    elif submitted:
        st.warning("No recommendations found for your criteria.")


# ---------------------------- Navigation ----------------------------
else:
    # All non-home pages just show the page name
    st.markdown(f"## 🧭 {st.session_state.page.capitalize()} Page")

# --- Custom Navigation Bar ---
st.markdown("""
    <style>
    .bottom-nav {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        max-width: 700px;
        width: 90%;
        background-color: #fff;
        border-top: 2px solid #ccc;
        display: flex;
        justify-content: space-around;
        align-items: center;
        padding: 6px 0;
        z-index: 9999;
        box-shadow: 0 -1px 5px rgba(0,0,0,0.1);
        border-radius: 12px 12px 0 0;
    }
    .bottom-nav a {
        text-decoration: none;
        color: black;
        font-weight: 500;
        text-align: center;
        flex-grow: 1;
        font-size: 14px;
        padding: 6px 0;
        border-right: 1px solid #eee;
    }
    .bottom-nav a:last-child {
        border-right: none;
    }
    .bottom-nav a:hover {
        color: #00aced;
    }
    </style>
    <div class="bottom-nav">
        <a href="?page=itinerary">Home</a>
        <a href="?page=events">Events</a>
        <a href="?page=restaurant">Restaurant</a>
    </div>
""", unsafe_allow_html=True)
