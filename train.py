import sys
import pickle as pkl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import MovieReviewsDataset, moviereviews_collate_func
from model import BagOfWords

def compute_acc(outputs, labels):
    predicted = outputs.max(1, keepdim=True)[1]
    correct = predicted.eq(labels.view_as(predicted)).sum().item()
    return correct

def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data, lengths, labels in loader:
            data_batch, length_batch, label_batch = data, lengths, labels
            outputs = F.softmax(model(data_batch, length_batch), dim=1)
            correct += compute_acc(outputs, labels)
            total += labels.size(0)
    return (100 * correct / total)

def main(ngram, max_vocab_size, emb_dim):
    # load data
    train_data_indices, train_targets = pkl.load(open(f"data/processed/train_data_indicies_{ngram}gram_{max_vocab_size}vocab.p", "rb"))
    val_data_indices, val_targets = pkl.load(open(f"data/processed/val_data_indicies_{ngram}gram_{max_vocab_size}vocab.p", "rb"))

    BATCH_SIZE = 64
    train_dataset = MovieReviewsDataset(train_data_indices, train_targets)
    train_loader = DataLoader(dataset=train_dataset,
                               batch_size=BATCH_SIZE,
                               collate_fn=moviereviews_collate_func,
                               shuffle=True)
                               #num_workers=4)

    val_dataset = MovieReviewsDataset(val_data_indices, val_targets)
    val_loader = DataLoader(dataset=val_dataset,
                           batch_size=BATCH_SIZE,
                           collate_fn=moviereviews_collate_func,
                           shuffle=True)
                           #num_workers=4)
    # SGD vs Adam
    # Learn rate
    # Linear annealing
    # params
    model = BagOfWords(max_vocab_size + 2, emb_dim)

    learning_rate = 0.01
    num_epochs = 100  # number epoch to train

    # Criterion and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    bad_performance_count = 0
    max_bad_performance = 25
    pre_val_acc = 0
    best_acc = 0
    running_correct = 0
    running_total = 0
    for epoch in range(num_epochs):
        for i, (data, lengths, labels) in enumerate(train_loader):
            model.train()
            data_batch, length_batch, label_batch = data, lengths, labels
            optimizer.zero_grad()
            outputs = model(data_batch, length_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()

            running_correct += compute_acc(outputs, labels)
            running_total += labels.size(0)

            # validate every 100 iterations
            if i > 0 and i % 100 == 0:
                # validate
                train_acc = 100 * running_correct / running_total
                running_total = 0
                running_correct = 0
                val_acc = test_model(val_loader, model)
                print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format(
                           epoch+1, num_epochs, i+1, len(train_loader), val_acc))
                print('Epoch: [{}/{}], Step: [{}/{}], Train Acc: {:.2f}'.format(
                           epoch+1, num_epochs, i+1, len(train_loader), train_acc))
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), f"models/model_{ngram}gram_{max_vocab_size}vocab_{emb_dim}embed.pth")
                elif pre_val_acc - val_acc > .1:
                    bad_performance_count += 1
                    if bad_performance_count >= max_bad_performance:
                        break
                pre_val_acc = val_acc
        if bad_performance_count >= max_bad_performance:
            break
    print("Best val accuracy: ", best_acc)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: train.py ngram max_vocab_size emb_dim")
    else:
        ngram = int(sys.argv[1])
        max_vocab_size = int(sys.argv[2])
        emb_dim = int(sys.argv[3])
        main(ngram, max_vocab_size, emb_dim)
