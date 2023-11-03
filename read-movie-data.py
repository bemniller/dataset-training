from tqdm import tqdm
from pathlib import Path

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in tqdm ((split_dir/label_dir).iterdir(), total=len(list((split_dir/label_dir).iterdir())), desc=f"Processing {label_dir} files") :
            texts.append(text_file.read_text(encoding="utf-8"))
            labels.append(0 if label_dir == "neg" else 1)

    return texts, labels

train_texts, train_labels = read_imdb_split('aclImdb/train')
test_texts, test_labels = read_imdb_split('aclImdb/test')

print(f"Number of training texts: {len(train_texts)}")
print(f"Number of training labels: {len(train_labels)}")
print(f"Number of test texts: {len(test_texts)}")
print(f"Number of test labels: {len(test_labels)}")