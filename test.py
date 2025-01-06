from transformers import pipeline

# Load the model and pipeline
sentiment_analysis = pipeline("sentiment-analysis")

# Test with a sample text
print(sentiment_analysis("This is the worst thing I ever bought."))
