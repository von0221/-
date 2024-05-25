import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from tqdm import tqdm
import os
import warnings
from transformers import BertModel
import torch.nn as nn

# 忽略特定警告
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.file_utils")

# Define the label map
label_map = {'因果': 0, '时序': 1}
inverse_label_map = {v: k for k, v in label_map.items()}


class BertWithBiLSTM(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, num_labels):
        super(BertWithBiLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bilstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                              hidden_size=hidden_dim,
                              num_layers=1,
                              bidirectional=True,
                              batch_first=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]

        # Use the BiLSTM layer
        lstm_output, _ = self.bilstm(sequence_output)  # [batch_size, seq_length, hidden_dim*2]

        # Use the hidden state of the last LSTM layer
        cls_output = lstm_output[:, 0, :]  # [batch_size, hidden_dim*2]

        logits = self.classifier(cls_output)  # [batch_size, num_labels]

        return logits


class EventDataset(Dataset):
    def __init__(self, events, relations, tokenizer, max_len):
        self.events = events
        self.relations = relations
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.relations)

    def __getitem__(self, idx):
        relation = self.relations[idx]
        event1 = self.events[relation['one_event_id']]
        event2 = self.events[relation['other_event_id']]

        tokens = self.tokenizer.encode_plus(
            event1['event-information']['trigger'][0]['text'],
            event2['event-information']['trigger'][0]['text'],
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        if 'relation_type' in relation:
            labels = torch.tensor(label_map[relation['relation_type']], dtype=torch.long)
        else:
            labels = torch.tensor(-1, dtype=torch.long)  # Dummy label for test data

        return {
            'input_ids': tokens['input_ids'].flatten(),
            'attention_mask': tokens['attention_mask'].flatten(),
            'labels': labels
        }


def load_data(file_path, is_train=True):
    events = {}
    relations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data = json.loads(line.strip())
                events.update({event['id']: event for event in data['events']})
                if is_train and 'relations' in data:
                    relations.extend(data['relations'])
    return events, relations


def train_epoch(model, data_loader, optimizer, device, criterion):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader):
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, device, criterion):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def predict(model, data_loader, device):
    model = model.eval()
    predictions = []

    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    return predictions


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # 设置文件路径
    train_file_path = 'train.json'
    test_file_path = 'testa.json'
    result_file_path = 'result.txt'

    # 加载训练数据
    events, relations = load_data(train_file_path)

    # 初始化 Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 创建 Dataset 和 DataLoader
    dataset = EventDataset(events, relations, tokenizer, max_len=128)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 初始化自定义模型
    model = BertWithBiLSTM(bert_model_name='bert-base-chinese', hidden_dim=256, num_labels=len(label_map))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)

    # 初始化优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    best_accuracy = 0
    for epoch in range(10):  # 训练10个epoch，可以调整
        print(f'Epoch {epoch + 1}/10')
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, device, criterion)
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(model, val_loader, device, criterion)
        print(f'Val loss {val_loss} accuracy {val_acc}')

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model_state.bin'))

    # 加载测试数据
    test_events, _ = load_data(test_file_path, is_train=False)

    test_relations = []
    for event_id_1 in test_events:
        for event_id_2 in test_events:
            if event_id_1 != event_id_2:
                test_relations.append({
                    'one_event_id': event_id_1,
                    'other_event_id': event_id_2
                })

    test_dataset = EventDataset(test_events, test_relations, tokenizer, max_len=128)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    predictions = predict(model, test_loader, device)

    # 保存结果
    with open(result_file_path, 'w', encoding='utf-8') as f:
        result = {'relations': []}
        for i, relation in enumerate(test_relations):
            relation['relation_type'] = inverse_label_map[predictions[i]]
            result['relations'].append(relation)
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
