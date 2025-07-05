import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Set up paths
DATA_PATH = Path(r'C:\Users\Daniel.Temesgen\Desktop\KIAM-Rsc\complaints.csv')
OUTPUT_PATH = DATA_PATH / 'filtered_complaints.csv'
DATA_PATH.mkdir(exist_ok=True)

# Load dataset (assuming CSV file is available; adjust path as needed)
# Load dataset
try:
    df = pd.read_csv(DATA_PATH / 'complaints.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Please place the CFPB complaints dataset on your Desktop.")
    exit(1)

# EDA
def perform_eda(df):
    print("Dataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    
    # Distribution of complaints by product
    product_counts = df['Product'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=product_counts.values, y=product_counts.index)
    plt.title('Distribution of Complaints by Product')
    plt.xlabel('Number of Complaints')
    plt.savefig('notebooks/product_distribution.png')
    plt.close()
    
    # Analyze narrative lengths
    df['narrative_length'] = df['Consumer complaint narrative'].dropna().apply(lambda x: len(word_tokenize(str(x))))
    print("\nNarrative Length Statistics:")
    print(df['narrative_length'].describe())
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['narrative_length'].dropna(), bins=50)
    plt.title('Distribution of Narrative Word Counts')
    plt.xlabel('Word Count')
    plt.savefig('notebooks/narrative_length_distribution.png')
    plt.close()
    
    # Complaints with and without narratives
    narrative_counts = df['Consumer complaint narrative'].isna().value_counts()
    print("\nComplaints with/without narratives:")
    print(f"With narratives: {narrative_counts.get(False, 0)}")
    print(f"Without narratives: {narrative_counts.get(True, 0)}")

# Text cleaning function
def clean_narrative(text):
    if pd.isna(text):
        return text
    # Lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove boilerplate phrases
    boilerplate_phrases = [
        r'i am writing to file a complaint',
        r'please assist',
        r'thank you for your attention'
    ]
    for phrase in boilerplate_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Main processing
def preprocess_data(df):
    # Filter for specified products
    target_products = [
        'Credit card',
        'Personal loan',
        'Buy Now, Pay Later (BNPL)',
        'Savings account',
        'Money transfers'
    ]
    df_filtered = df[df['Product'].isin(target_products)].copy()
    
    # Remove records with empty narratives
    df_filtered = df_filtered.dropna(subset=['Consumer complaint narrative'])
    
    # Clean narratives
    df_filtered['Consumer complaint narrative'] = df_filtered['Consumer complaint narrative'].apply(clean_narrative)
    
    # Remove any narratives that became empty after cleaning
    df_filtered = df_filtered[df_filtered['Consumer complaint narrative'].str.strip() != '']
    
    return df_filtered

def main():
    # Create notebooks directory for plots
    Path('notebooks').mkdir(exist_ok=True)
    
    # Perform EDA
    print("Performing Exploratory Data Analysis...")
    perform_eda(df)
    
    # Preprocess data
    print("\nPreprocessing data...")
    df_processed = preprocess_data(df)
    
    # Save filtered dataset
    df_processed.to_csv(OUTPUT_PATH, index=False)
    print(f"\nFiltered dataset saved to {OUTPUT_PATH}")
    print("Processed dataset shape:", df_processed.shape)

if __name__ == "__main__":
    main()