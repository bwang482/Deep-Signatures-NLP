import torch
import torch.optim as optim
import time
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(preds, y):
    """accuracy function for binary classification problem (sentiment analysis)"""
    #rounded_preds = torch.round(torch.sigmoid(preds))
    #correct = (rounded_preds == y).float()     
    #acc = correct.sum() / len(correct)

    ##Multiclass accuracy
    _, outputs = torch.max(preds, 1)
    acc = torch.sum(outputs==y).type(torch.float)/torch.tensor(y.size()[0], dtype=torch.float)

    return acc
    

def train(model, iterator, optimizer, criterion, need_lengths):
    """train for one batch"""
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        if need_lengths:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
        else:
            predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label.long())
        acc = accuracy(predictions, batch.label.long())
        #loss = criterion(predictions, batch.label)
        #acc = accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, need_lengths):
    """evaluate for one batch with no gradient required"""
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            if need_lengths:
                text, text_lengths = batch.text
                predictions = model(text, text_lengths).squeeze(1)
            else:
                predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label.long())
            acc = accuracy(predictions, batch.label.long())
            #loss = criterion(predictions, batch.label)
            #acc = accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    """how much time does an epoch take"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def fit(model, train_iterator, valid_iterator, criterion, epochs, model_name, history,
        #lr=0.001, momentum=0.8, 
        freeze_embedding=True, save_best_model=False, need_lengths=True):
    """fit across all epochs and save the best model"""

    best_valid_loss = float('inf')

    if freeze_embedding:
        model.embedding.weight.requires_grad = False
        optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad==True])
        #optimizer = optim.SGD([param for param in model.parameters() if param.requires_grad==True], lr=0.001)
    
    else:
        optimizer = optim.Adam(model.parameters())
        #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    history[model_name] = []

    for epoch in tqdm(range(epochs)):

        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, need_lengths=need_lengths)
        middle_time = time.time()
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, need_lengths=need_lengths)
        end_time = time.time()
        epoch_mins_t, epoch_secs_t = epoch_time(start_time, middle_time)
        epoch_mins_e, epoch_secs_e = epoch_time(middle_time, end_time)
        history[model_name].append(valid_loss)
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if save_best_model:
                torch.save(model.state_dict(), './figures/sentiment analysis/' + model_name + '.pt')
    
        print(f'Epoch: {epoch+1:02} | Epoch Time for training: {epoch_mins_t}m {epoch_secs_t}s')
        print(f'Epoch: {epoch+1:02} | Epoch Time for evaluating: {epoch_mins_e}m {epoch_secs_e}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')