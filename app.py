from transformers import pipeline
import gradio as gr

# Load the sentiment analysis model
sentiment_analysis = pipeline("sentiment-analysis")

# Define a function to process the sentiment of the input text
def analyze_sentiment(text):
    try:
        # Get prediction result from the model
        result = sentiment_analysis(text)[0]
        # Return label and score
        return result['label'], round(result['score'], 2)
    except Exception as e:
        return "Error", str(e)

# Create Gradio interface
interface = gr.Interface(
    fn=analyze_sentiment,             # Function that handles input and returns sentiment
    inputs="text",                    # Text input field
    outputs=[gr.Textbox(), gr.Textbox()],  # Two output boxes: one for the label and one for the confidence score
    title="Sentiment Analysis",       # Title of the app
    description="Enter a text to determine its sentiment (POSITIVE/NEGATIVE)."
)

# Launch the interface
if __name__ == "__main__":
    interface.launch(share=True)  # `share=True` generates a public link
