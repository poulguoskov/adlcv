from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import random
import numpy as np

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(1)

# Load IMDB dataset
dataset = load_dataset('imdb')

train_data = dataset['train'].shuffle(seed=1).select(range(4000))
test_data = dataset['test'].shuffle(seed=1).select(range(1000))

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)

train_data = train_data.map(tokenize, batched=True)
test_data = test_data.map(tokenize, batched=True)

train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model = model.to(device)

opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

loss_function = nn.CrossEntropyLoss()
num_epochs = 10  # fine-tuning needs far fewer epochs

for e in range(num_epochs):
    print(f'\n epoch {e}')
    model.train()
    for batch in train_loader:
        opt.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_function(outputs.logits, labels)
        loss.backward()
        opt.step()

    with torch.no_grad():
        model.eval()
        tot, cor = 0.0, 0.0
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1)
            tot += float(labels.size(0))
            cor += float((preds == labels).sum().item())

        acc = cor / tot
        print(f'-- validation accuracy {acc:.3f}')