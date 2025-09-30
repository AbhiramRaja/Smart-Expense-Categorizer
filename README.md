💸 Smart Expense Categorizer
An interactive web application that uses a robust Machine Learning classifier to instantly categorize financial transaction text into specific expense types.

Live Demo
https://smart--expense--tracker.streamlit.app

💡 Overview
This project tackles the challenge of messy, real-world financial transaction data by building a high-accuracy text classifier. It demonstrates a complete end-to-end ML pipeline, from synthetic data generation to interactive web deployment, all condensed into a single Python file.

Key Features

Instant Classification: Predicts the expense category (e.g., Food & Dining, Transportation, Bills & Utilities) from any user-input transaction message.

Robustness via Noise: The model is trained on a deliberately noisy, synthetically generated dataset that includes common real-world issues like typos, mixed casing, abbreviations ("pmt", "dlvrd"), and varied currency formats.

Advanced Feature Engineering: Utilizes a highly tuned TF-IDF Vectorizer with an expanded feature space (2000 features and up to 3-grams) to capture complex patterns and subtle keyword associations.

Self-Contained Deployment: The entire system (data generation, model training, and Streamlit UI) is contained in a single app.py file, enabling streamlined deployment via Streamlit Community Cloud.

📈 Model Performance
The classifier, a Logistic Regression model from scikit-learn, achieved excellent results on the dedicated test set:

Metric	Value
Model	Logistic Regression
Dataset Size	800 Transactions (Balanced)
Test Accuracy	99.38%
Methodology	Model trained using stratified train/test split on synthetically generated data.
🛠️ Technology Stack
Machine Learning: Scikit-learn (Logistic Regression, TfidfVectorizer)

Web Framework: Streamlit

Data Handling: Pandas, NumPy

Language: Python 3.x

Deployment: Streamlit Community Cloud

⚙️ Local Setup & Execution
Follow these steps to get a local copy of the project running on your machine.

Prerequisites

You must have Python 3.x and pip installed.

1. Clone the Repository

Bash
git clone https://github.com/AbhiramRaja/Smart-Expense-Categorizer.git
cd Smart-Expense-Categorizer
2. Set up Virtual Environment (Recommended)

Bash
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies

Install all required libraries specified in the requirements.txt file:

Bash
pip install -r requirements.txt
4. Run the Application

Execute the main application file using Streamlit:

Bash
streamlit run app.py
Note on First Run: If the model files (.pkl) do not exist, the script will automatically pause to generate the 800-transaction dataset, train the model, and save the assets. The app will then launch. On subsequent runs, it will load the saved model instantly.

The application will automatically open in your web browser at http://localhost:8501.
