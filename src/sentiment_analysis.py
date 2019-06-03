import torch
from torchtext import data
from torchtext import datasets
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_batched_data(batch_size, pre_trained=False, max_voc_size=5_000, TREC=False):
    """returns necessary dataloaders for sentiment analysis with IMDB data"""

    # set-up using torchtext
    SEED = 1234
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True #setting to True for reproducibility

    TEXT = data.Field(tokenize = 'spacy', include_lengths=True)
    LABEL = data.LabelField(dtype = torch.float)

    if TREC: # TREC dataset (question classification)
        train_data, test_data = datasets.TREC.splits(TEXT, 
                                                     LABEL, 
                                                     root='.data', 
                                                     train='train_5500.label', 
                                                     test='TREC_10.label')

    else: # IMD dataset (film reviews)
        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
        
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))
     
    if pre_trained: # pre-trained word embedding (Glove)
        TEXT.build_vocab(train_data, 
                         max_size = max_voc_size, 
                         vectors = "glove.6B.50d", 
                         unk_init = torch.Tensor.normal_)
    else: # not pre-trained embedding
        TEXT.build_vocab(train_data, max_size=max_voc_size)

    LABEL.build_vocab(train_data)

    # define iterators
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = batch_size,
        sort_within_batch = True,
        device = device)

    return TEXT, LABEL, train_iterator, valid_iterator, test_iterator