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

# --- Configuration (Simplified) ---
MODEL_FILE = 'expense_classifier_clean.pkl' # Renamed files for clarity
VECTORIZER_FILE = 'tfidf_vectorizer_clean.pkl'
DATA_FILE = 'transactions_clean.csv'
SAMPLES_PER_CATEGORY = 50 # Reduced sample count for speed on clean data

# --- Templates (Original Clean Structure) ---

templates = {
    "Food & Dining": [
        "Swiggy order delivered Rs {}",
        "Zomato - {} paid",
        "Paid to {} restaurant Rs {}",
        "Dominos pizza order Rs {}",
        "McDonald's payment Rs {}",
        "Cafe Coffee Day bill Rs {}",
        "Food delivery {} rupees",
        "Dinner at {} Rs {}",
        "Breakfast from {} Rs {}",
        "Lunch order {} Rs {}"
    ],
    "Transportation": [
        "Uber trip completed Rs {}",
        "Ola ride payment Rs {}",
        "Rapido bike ride Rs {}",
        "Metro card recharge Rs {}",
        "Petrol pump {} Rs {}",
        "Auto rickshaw Rs {}",
        "Bus ticket Rs {}",
        "Fastag recharge Rs {}",
        "Parking fee Rs {}",
        "Cab service Rs {}"
    ],
    "Shopping": [
        "Amazon purchase Rs {}",
        "Flipkart order delivered Rs {}",
        "Myntra shopping Rs {}",
        "Paid to {} store Rs {}",
        "Online shopping {} rupees",
        "Ajio order Rs {}",
        "Meesho purchase Rs {}",
        "Big Bazaar bill Rs {}",
        "DMart shopping Rs {}",
        "Grocery purchase Rs {}"
    ],
    "Bills & Utilities": [
        "Electricity bill paid Rs {}",
        "Water bill payment Rs {}",
        "Internet bill {} rupees",
        "Mobile recharge Rs {}",
        "DTH recharge Rs {}",
        "Gas cylinder payment Rs {}",
        "Rent payment Rs {}",
        "Maintenance charges Rs {}",
        "Broadband bill Rs {}",
        "Phone bill paid Rs {}"
    ],
    "Entertainment": [
        "BookMyShow ticket Rs {}",
        "Netflix subscription Rs {}",
        "Amazon Prime payment Rs {}",
        "Movie ticket Rs {}",
        "Hotstar subscription Rs {}",
        "Spotify premium Rs {}",
        "Gaming purchase Rs {}",
        "Concert ticket Rs {}",
        "Theme park entry Rs {}",
        "Youtube Premium Rs {}"
    ],
    "Healthcare": [
        "Pharmacy bill Rs {}",
        "Doctor consultation Rs {}",
        "Lab test payment Rs {}",
        "Hospital bill Rs {}",
        "Medicine purchase Rs {}",
        "Health checkup Rs {}",
        "Dental treatment Rs {}",
        "Clinic visit Rs {}",
        "Medical store {} rupees",
        "Prescription {} Rs {}"
    ],
    "Education": [
        "Course fee payment Rs {}",
        "Book purchase Rs {}",
        "Udemy course Rs {}",
        "Tuition fee Rs {}",
        "Coursera subscription Rs {}",
        "Stationery {} rupees",
        "College fee Rs {}",
        "Online class Rs {}",
        "Study material Rs {}",
        "Exam fee Rs {}"
    ],
    "Others": [
        "ATM withdrawal Rs {}",
        "Bank charges Rs {}",
        "Donation Rs {}",
        "Gift purchase Rs {}",
        "Miscellaneous payment Rs {}",
        "Online payment Rs {}",
        "UPI transfer Rs {}",
        "Cash withdrawal Rs {}",
        "Service charge Rs {}",
        "Other expense Rs {}"
    ]
}

# Brand names (Simplified)
brands = {
    "Food & Dining": ["Burger King", "KFC", "Subway"],
    "Shopping": ["Reliance Digital", "Croma"],
    "Bills & Utilities": ["Airtel", "Jio"],
    "Healthcare": ["Apollo", "MedPlus"],
    "Education": ["BYJU'S", "Vedantu"],
    "Transportation": ["IOCL", "BPCL"],
    "Others": ["NGO", "Bank"],
    "Entertainment": ["PVR", "INOX"]
}


# --- Function to Run Data Generation and Training (Simplified) ---

def generate_and_train(templates, brands, data_file, model_file, vectorizer_file):
    
    
    # Generate dataset
    data = []
    for category, template_list in templates.items():
        for _ in range(SAMPLES_PER_CATEGORY):
            template = random.choice(template_list)
            
            # Generate amount (Simplified, no complex formatting)
            if category in ["Bills & Utilities", "Shopping", "Healthcare"]:
                amount = random.randint(500, 3000)
            elif category == "Education":
                amount = random.randint(1000, 10000)
            else:
                amount = random.randint(50, 1000)
                
            # Fill template logic (Simplified to use base format)
            if template.count('{}') == 2:
                brand = random.choice(brands.get(category, ["Merchant ABC"]))
                text = template.format(brand, amount)
            else:
                text = template.format(amount)

            data.append({'transaction_text': text, 'category': category})

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(data_file, index=False)

    # --- Training Pipeline ---
    X = df['transaction_text']
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['category'])

    # Simplified vectorizer for clean data
    vectorizer = TfidfVectorizer(
        max_features=500, 
        ngram_range=(1, 2), # Less complex n-grams needed
        lowercase=True,
        stop_words='english'
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    # Using default max_iter since dataset is smaller
    model = LogisticRegression(random_state=42, C=1.0) 
    model.fit(X_train_tfidf, y_train)

    # Save the trained model and vectorizer
    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)

    st.success(f"Training complete! {len(df)} transactions generated. Model saved.")
    return model, vectorizer


# --- Streamlit UI and Prediction Logic ---

# Use st.cache_resource for objects that should be loaded only once across sessions
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


st.title("💸 Simple Transaction Category Classifier")
st.markdown("---")
st.header("Predict Your Expense Category")
st.markdown("Enter a transaction message (e.g., *Uber trip completed Rs 350*) and click Predict.")

# Text input for the user
user_input = st.text_input(
    "Enter Transaction Message:",
    placeholder="e.g., Swiggy order delivered Rs 450",
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
Uber trip completed Rs 280
Amazon purchase Rs 1500
Electricity bill paid Rs 850
Movie ticket Rs 300
""")