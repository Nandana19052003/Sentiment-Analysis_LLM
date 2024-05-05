import streamlit as st
import pickle
import re

# Load the function to recreate the model
with open('sentiment1_model.pkl', 'rb') as f:
    create_model = pickle.load(f)

# Define a function for text cleaning
def clean_text(text):
    # Remove special characters and punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]*>", " ", text)

    # Lowercase the text
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Trim leading and trailing spaces
    text = text.strip()

    return text

# Define a function to analyze sentiment
def analyze_sentiment(review):
    model = create_model()
    prompt = f"""
    As an expert linguist skilled in sentiment analysis, your task is to classify customer reviews into Positive (label=1) and Negative (label=0) sentiments.

The customer reviews are provided in JSON format between three backticks.

Your task is to update the predicted labels under the 'pred_label' field in the JSON code.

Please ensure that you maintain the original JSON code format and only modify the 'pred_label' field.

    ```
    {{
      "clean_reviews": "{review}",
      "pred_label": ""
    }}
    ```
    """
    response = model.generate_content(prompt)
    return response.text.strip("`")

# Define the Streamlit app
def main():
    st.title("Sentiment Analysis with Gemini")

    # Text input for customer review
    customer_review = st.text_area("Enter Customer Review", "")

    if st.button("Analyze Sentiment"):
        # Clean the input text
        cleaned_review = clean_text(customer_review)

        # Analyze sentiment
        predicted_label = analyze_sentiment(cleaned_review)

        # Display predicted label
        st.write("Predicted Label:")
        st.write(predicted_label)

if __name__ == "__main__":
    main()
