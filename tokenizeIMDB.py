# Import necessary libraries and modules
from transformers import DistilBertTokenizerFast
# Import the readMovieData module where the IMDB dataset reading function is defined.
import readMovieData

# Initialize the DistilBERT tokenizer. 
# We're using the "distilbert-base-uncased" version which is a smaller, faster, cheaper version of BERT.
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize the training data.
# `truncation=True` ensures that inputs longer than the model's maximum input length are truncated.
# `padding=True` ensures that shorter sequences are padded up to the model's maximum input length.
train_encodings = tokenizer(readMovieData.train_texts, truncation=True, padding=True)

# Tokenize the validation data.
val_encodings = tokenizer(readMovieData.val_texts, truncation=True, padding=True)

# Tokenize the test data.
test_encodings = tokenizer(readMovieData.test_texts, truncation=True, padding=True)
