from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

def read_imdb_split(split_dir):
    # Convert the directory string to a Path object for easier path operations.
    split_dir = Path(split_dir)
    texts = []
    labels = []

    # Loop over both positive and negative reviews.
    for label_dir in ["pos", "neg"]:
        # Get all files for the current label (either positive or negative).
        all_files = list((split_dir/label_dir).iterdir())
        
        # Loop over each file with a progress bar.
        for text_file in tqdm(all_files, total=len(all_files), desc=f"Processing {label_dir} files"):
            # Read the file content with utf-8 encoding and append to texts.
            texts.append(text_file.read_text(encoding="utf-8"))
            # Append the label to labels. 0 for negative and 1 for positive.
            labels.append(0 if label_dir == "neg" else 1)

    return texts, labels

# Read IMDB dataset for both training and test data.
train_texts, train_labels = read_imdb_split('aclImdb/train')
test_texts, test_labels = read_imdb_split('aclImdb/test')

# Split the training data into training and validation sets.
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

# Print the number of texts and labels for each set.
print(f"Number of training texts: {len(train_texts)}")
print(f"Number of training labels: {len(train_labels)}")
print(f"Number of validation texts: {len(val_texts)}")
print(f"Number of validation labels: {len(val_labels)}")
print(f"Number of test texts: {len(test_texts)}")
print(f"Number of test labels: {len(test_labels)}")
