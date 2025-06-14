from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "facebook/bart-large-mnli"

# Downloads and caches locally
AutoTokenizer.from_pretrained(model_name)
AutoModelForSequenceClassification.from_pretrained(model_name)
