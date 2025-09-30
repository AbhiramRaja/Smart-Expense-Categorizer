import numpy as np
import pandas as pd
import random
import re
import joblib
import os
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- Configuration ---
MODEL_FILE = 'expense_classifier_realistic.pkl'
VECTORIZER_FILE = 'tfidf_vectorizer_realistic.pkl'
DATA_FILE = 'transactions.csv'
SAMPLES_PER_CATEGORY = 100

# --- Noise Generation and Formatting Functions for Realism ---

def introduce_noise(text):
    """
    Introduces common real-world noise like casing, typos, and abbreviations.
    """
    # Randomly change case
    if random.random() < 0.3:
        text = text.lower()
    elif random.random() < 0.6:
        text = text.upper()
    
    # Introduce random typos (simple character swap)
    if random.random() < 0.05 and len(text) > 5:
        i = random.randint(0, len(text) - 2)
        text_list = list(text)
        text_list[i], text_list[i+1] = text_list[i+1], text_list[i]
        text = "".join(text_list)

    # Abbreviate or misspell common words (e.g., 'delivered' -> 'dlvry')
    text = re.sub(r'delivered', random.choice(['dlvrd', 'delivered', 'delivrd']), text, flags=re.I)
    text = re.sub(r'payment', random.choice(['pmt', 'payment', 'paymnt']), text, flags=re.I)

    return text

def format_amount(amount):
    """
    Formats the amount realistically (e.g., with or without 'Rs', abbreviations, commas).
    """
    # Randomly use different "Rupees" formats
    rs_format = random.choice(["Rs ", "INR ", "Rupees ", "rs ", "rs. ", ""])

    # Randomly include a decimal or comma (if amount is large)
    if random.random() < 0.2 and amount > 1000:
        amount_str = f"{amount:,}"
    else:
        amount_str = str(amount)

    return f"{rs_format}{amount_str}"

# Defining templates for each category
templates = {
    "Food & Dining": [
        "Swiggy dlvry order for {}", "Zomato paid to {} Rs {}", "Lunch at {} restaurant {}",
        "Dominos pizza bill {}", "McD payment {} Rs {}", "Cafe Coffee Day bill {}",
        "Food delivery {} rupees", "Dinner for {} at {} Pmt", "Breakfast order {}",
        "Takeaway from {} {}",
    ],
    "Transportation": [
        "Uber trip completed Rs {}", "Ola ride payment {}", "Rapido bike ride Rs {}",
        "Metro card recharge {} rupees", "Petrol pump {} Rs {}", "Auto rickshaw {}",
        "Bus ticket Rs {}", "Fastag recharge Rs {}", "Parking fee Rs {}",
        "Cab service {}",
    ],
    "Shopping": [
        "Amazon purchase Rs {}", "Flipkart order delivered Rs {}", "Myntra shopping {}",
        "Paid to {} store Rs {}", "Online shopping {} rupees", "Ajio order Rs {}",
        "Meesho purchase {}", "Big Bazaar bill Rs {}", "DMart shopping Rs {}",
        "Grocery purchase {}",
    ],
    "Bills & Utilities": [
        "Electricity bill paid Rs {}", "Water bill payment Rs {}", "Internet bill {} rupees",
        "Mobile recharge Rs {}", "DTH recharge Rs {}", "Gas cylinder payment Rs {}",
        "Rent payment Rs {}", "Maintenance charges Rs {}", "Broadband bill {}",
        "Phone bill paid {}",
    ],
    "Entertainment": [
        "BookMyShow ticket Rs {}", "Netflix subscription Rs {}", "Amazon Prime payment Rs {}",
        "Movie ticket {}", "Hotstar subscription Rs {}", "Spotify premium {}",
        "Gaming purchase Rs {}", "Concert ticket {}", "Theme park entry Rs {}",
        "Youtube Premium {}",
    ],
    "Healthcare": [
        "Pharmacy bill Rs {}", "Doctor consultation Rs {}", "Lab test payment Rs {}",
        "Hospital bill Rs {}", "Medicine purchase {}", "Health checkup Rs {}",
        "Dental treatment Rs {}", "Clinic visit {}", "Medical store {} rupees",
        "Prescription {} Rs {}",
    ],
    "Education": [
        "Course fee payment Rs {}", "Book purchase Rs {}", "Udemy course {}",
        "Tuition fee Rs {}", "Coursera subscription Rs {}", "Stationery {} rupees",
        "College fee Rs {}", "Online class Rs {}", "Study material {}",
        "Exam fee Rs {}",
    ],
    "Others": [
        "ATM withdrawal Rs {}", "Bank charges Rs {}", "Donation Rs {}",
        "Gift purchase Rs {}", "Miscellaneous payment Rs {}", "Online payment Rs {}",
        "UPI transfer Rs {}", "Cash withdrawal Rs {}", "Service charge Rs {}",
        "Other expense Rs {}",
    ]
}

# Expanded brands for better realism
brands = {
    "Food & Dining": ["Burger King", "KFC", "Subway", "Pizza Hut", "Starbucks", "A2B", "Saravana Bhavan", "Local Cafe", "Taco Bell"],
    "Transportation": ["HP Petrol", "IOCL", "BPCL", "Local Pump"],
    "Shopping": ["Reliance Digital", "Croma", "Decathlon", "Max Fashion", "Zara", "Lifestyle", "Pantaloons", "Local Store"],
    "Bills & Utilities": ["Airtel", "Jio", "BSNL", "Vi", "ACT Fibernet", "BESCOM", "TNEB", "Reliance Energy"],
    "Healthcare": ["Apollo Pharmacy", "Medplus", "Dr. Lal Pathlabs", "Manipal Hospital", "Local Clinic"],
    "Education": ["BYJU'S", "Vedantu", "FIITJEE", "Aakash", "The Study Center"],
}


# --- Function to Run Data Generation and Training ---

def generate_and_train(templates, brands, data_file, model_file, vectorizer_file):
    
    st.info("Training required! Generating synthetic data and training the model. This will run only once.")
    
    # Generate dataset
    data = []
    for category, template_list in templates.items():
        for _ in range(SAMPLES_PER_CATEGORY):
            template = random.choice(template_list)
            
            # Generate appropriate amount based on category
            if category == "Bills & Utilities":
                amount = random.randint(200, 5000)
            elif category == "Healthcare":
                amount = random.randint(50, 7500)
            elif category == "Education":
                amount = random.randint(500, 15000)
            elif category == "Food & Dining":
                amount = random.randint(30, 2000)
            elif category == "Transportation":
                amount = random.randint(20, 1500)
            else:
                amount = random.randint(50, 6000)
                
            formatted_amt = format_amount(amount)

            # Fill template logic
            if template.count('{}') == 2:
                brand = random.choice(brands.get(category, ["Merchant ABC", "Shop X"]))
                text = template.format(brand, formatted_amt)
            else:
                if template.endswith("}"):
                    text = template.format(formatted_amt)
                else: 
                    brand = random.choice(brands.get(category, ["Pvt Ltd", "Generic Shop"]))
                    text = template.format(brand)

            # Apply realistic noise
            text = introduce_noise(text)

            data.append({'transaction_text': text, 'category': category})

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(data_file, index=False)

    # --- Training Pipeline ---
    X = df['transaction_text']
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['category'])

    vectorizer = TfidfVectorizer(
        max_features=2000, 
        ngram_range=(1, 3),
        lowercase=True,
        stop_words='english',
        token_pattern=r'(?u)\b\w[\w\-\.\#]*\b'
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    model.fit(X_train_tfidf, y_train)

    # Save the trained model and vectorizer
    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)

    st.success(f"Training complete! {len(df)} transactions generated. Model saved.")
    return model, vectorizer


# --- Streamlit UI and Prediction Logic ---

# Use st.cache_resource for objects that should be loaded only once across sessions
# This is crucial for performance in a Streamlit app!
@st.cache_resource
def load_assets(model_file, vectorizer_file):
    """Loads the trained model and vectorizer from disk, training if necessary."""
    
    if not os.path.exists(model_file) or not os.path.exists(vectorizer_file):
        # If files don't exist, run the full training pipeline
        model, vectorizer = generate_and_train(templates, brands, DATA_FILE, model_file, vectorizer_file)
    else:
        # If files exist, just load them
        model = joblib.load(model_file)
        vectorizer = joblib.load(vectorizer_file)
    
    return model, vectorizer


# Load the model and vectorizer (will train first time)
model, vectorizer = load_assets(MODEL_FILE, VECTORIZER_FILE)


st.title("💸 Transaction Category Classifier")
st.markdown("---")
st.header("Predict Your Expense Category")
st.markdown("Enter a transaction message (e.g., *Uber trip 350 rs*) and click Predict.")

# Text input for the user
user_input = st.text_input(
    "Enter Transaction Message:",
    placeholder="e.g., SWIGGY order dlrd 450",
    key="user_text_input"
)

# Button to trigger classification
if st.button("Predict Category", type="primary"):
    if user_input:
        
        # 1. Transform the input text
        input_tfidf = vectorizer.transform([user_input])

        # 2. Make the prediction
        prediction = model.predict(input_tfidf)[0]

        # 3. Get the prediction probability (confidence)
        confidence = model.predict_proba(input_tfidf).max() * 100

        # 4. Display the results
        st.success("✅ Classification Result:")
        st.metric(
            label="Predicted Category",
            value=prediction
        )
        st.info(f"Confidence: **{confidence:.1f}%**")
        
    else:
        st.warning("Please enter a transaction message to classify.")

st.markdown("---")
st.subheader("Example Inputs")
st.code("""
Uber ride to office 280 RS
AMZN shopping 1,500
electrictiy bill pmt 850
Movie tkt BookMyShow 300
""")