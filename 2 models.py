# Importing libreries
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

# function to load data from the dataset provided
def load_data(folder_path, queries=None):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # Ensuring to read only text files
            try:
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                    content = file.read()
                    if queries:
                        for query in queries:
                            if query.lower() in content.lower():
                                data.append(content)
                                break  # Breaks to avoid duplicate entries for a file matching multiple queries
                    else:
                        data.append(content)
            except UnicodeDecodeError as e:
                print(f"Error decoding file {filename}: {str(e)}")
            except Exception as e:
                print(f"Error reading file {filename}: {str(e)}")
    return data

# Paths to your data folders
# current_dir = os.getcwd()  # Changed to getcwd for compatibility with notebooks
current_dir = os.getcwd()
dataset_folder = os.path.join(current_dir, '')
train_pos_path = os.path.join(dataset_folder, 'train', 'pos')
train_neg_path = os.path.join(dataset_folder, 'train', 'neg')
test_pos_path = os.path.join(dataset_folder, 'test', 'pos')
test_neg_path = os.path.join(dataset_folder, 'test', 'neg')

queries_to_search = ["great", "disappointing", "awesome"]
train_pos = load_data(train_pos_path, queries_to_search)
train_neg = load_data(train_neg_path, queries_to_search)
test_pos = load_data(test_pos_path, queries_to_search)
test_neg = load_data(test_neg_path, queries_to_search)

# declaring mode 1 : DistilBERT by Bhadresh Savani
distilbert_model_name = 'bhadresh-savani/distilbert-base-uncased-emotion'
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model_name)
distilbert_model = DistilBertForSequenceClassification.from_pretrained(distilbert_model_name)

# declaring mode 2: BERT by NLPtown
bert_model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name)

# Assuming train_pos, train_neg, test_pos, test_neg are loaded as in the previous steps
# Assign labels: 1 for positive, 0 for negative
train_labels = [1] * len(train_pos) + [0] * len(train_neg)
test_labels = [1] * len(test_pos) + [0] * len(test_neg)

# Concatenate positive and negative reviews for training and testing
train_texts = train_pos + train_neg
test_texts = test_pos + test_neg

# Tokenize the texts using the DistilBERT tokenizer
train_encodings_distilbert = distilbert_tokenizer(train_texts, truncation=True, padding=True)
test_encodings_distilbert = distilbert_tokenizer(test_texts, truncation=True, padding=True)

# Tokenize the texts using the BERT tokenizer
train_encodings_bert = bert_tokenizer(train_texts, truncation=True, padding=True)
test_encodings_bert = bert_tokenizer(test_texts, truncation=True, padding=True)


class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
    
# for model 1: distilbert
train_dataset_distilbert = SentimentDataset(train_encodings_distilbert, train_labels)
test_dataset_distilbert = SentimentDataset(test_encodings_distilbert, test_labels)

train_loader_distilbert = DataLoader(train_dataset_distilbert, batch_size=8, shuffle=True)
test_loader_distilbert = DataLoader(test_dataset_distilbert, batch_size=8, shuffle=False)

# for model 1: bert
train_dataset_bert = SentimentDataset(train_encodings_bert, train_labels)
test_dataset_bert = SentimentDataset(test_encodings_bert, test_labels)

train_loader_bert = DataLoader(train_dataset_bert, batch_size=8, shuffle=True)
test_loader_bert = DataLoader(test_dataset_bert, batch_size=8, shuffle=False)

training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    logging_steps=10,
)


#trainer for distilbert model (Model 1)
trainer_distilbert = Trainer(
    model=distilbert_model,  # the instantiated Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset_distilbert,  # training dataset
    eval_dataset=test_dataset_distilbert  # evaluation dataset
)

#trainer for bert model (Model 2)
trainer_bert = Trainer(
    model=bert_model,  # the instantiated Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset_bert,  # training dataset
    eval_dataset=test_dataset_bert  # evaluation dataset
)

# running trainer of model 1
trainer_distilbert.train()


# running trainer for model 2
trainer_bert.train()


#evaluating the results of Model 1
print("Results of Model 1: DistilBERT by Bhradresh Savani")
# Evaluate the model on the test set
result_distilbert = trainer_distilbert.evaluate()
print(result_distilbert)

# Get predictions for the test set
predictions_distilbert = trainer_distilbert.predict(test_dataset=test_dataset_distilbert)
predicted_classes_distilbert = predictions_distilbert.predictions.argmax(axis=1)
true_labels_distilbert = test_labels

# Calculate accuracy
accuracy_distilbert = accuracy_score(true_labels_distilbert, predicted_classes_distilbert)
print(f"Accuracy: {accuracy_distilbert}")

# Generate classification report
class_names = ['Negative', 'Positive']
print(classification_report(true_labels_distilbert, predicted_classes_distilbert, target_names=class_names))

#Evaluating the results of model 2
print("Results of model 2: BERT by NLPtown")
# Evaluate the model on the test set
result_bert = trainer_bert.evaluate()
print(result_bert)

# Get predictions for the test set
predictions_bert = trainer_bert.predict(test_dataset=test_dataset_bert)
predicted_classes_bert = predictions_bert.predictions.argmax(axis=1)
true_labels_bert = test_labels

# Calculate accuracy
accuracy_bert = accuracy_score(true_labels_bert, predicted_classes_bert)
print(f"Accuracy: {accuracy_bert}")

# Generate classification report
class_names = ['Negative', 'Positive']
print(classification_report(true_labels_bert, predicted_classes_bert, target_names=class_names))