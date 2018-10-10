import sys
import pickle as pkl
from collections import Counter

# save index 0 for unk and 1 for pad
PAD_IDX = 0
UNK_IDX = 1

def token2index_dataset(tokens_data, token2id):
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[token] if token in token2id else UNK_IDX for token in tokens]
        indices_data.append(index_list)
    return indices_data

def build_vocab(all_tokens):
    # Returns:
    # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
    # token2id: dictionary where keys represent tokens and corresponding values represent indices
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(max_vocab_size))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2,2+len(vocab))))
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token

def main(ngram, max_vocab_size):
    print(f"Argin: \n\tngram: {ngram}\n\tmax_vocab_size: {max_vocab_size}")

    train_data_tokens, train_targets = pkl.load(open(f"data/processed/train_data_{ngram}gram.p", "rb"))
    val_data_tokens, val_targets = pkl.load(open(f"data/processed/val_data_{ngram}gram.p", "rb"))
    test_data_tokens, test_targets = pkl.load(open(f"data/processed/test_data_{ngram}gram.p", "rb"))
    all_train_tokens = pkl.load(open(f"data/processed/all_train_{ngram}grams.p", "rb"))

    # double checking
    print ("Train dataset size is {}".format(len(train_data_tokens)))
    print ("Val dataset size is {}".format(len(val_data_tokens)))
    print ("Test dataset size is {}".format(len(test_data_tokens)))

    expected_n_tokens = 4795739 - (ngram-1) * 20000
    assert(len(all_train_tokens) == expected_n_tokens)

    token2id, id2token = build_vocab(all_train_tokens)

    train_data_indices = token2index_dataset(train_data_tokens, token2id)
    val_data_indices = token2index_dataset(val_data_tokens, token2id)
    test_data_indices = token2index_dataset(test_data_tokens, token2id)

    # double checking
    print ("Train dataset size is {}".format(len(train_data_indices)))
    print ("Val dataset size is {}".format(len(val_data_indices)))
    print ("Test dataset size is {}".format(len(test_data_indices)))

    pkl.dump([train_data_indices, train_targets], open(f"data/processed/train_data_indicies_{ngram}gram_{max_vocab_size}vocab.p", "wb"))
    pkl.dump([val_data_indices, val_targets], open(f"data/processed/val_data_indicies_{ngram}gram_{max_vocab_size}vocab.p", "wb"))
    pkl.dump([test_data_indices, test_targets], open(f"data/processed/test_data_indicies_{ngram}gram_{max_vocab_size}vocab.p", "wb"))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: index_data.py ngram max_vocab_size")
    else:
        ngram = int(sys.argv[1])
        max_vocab_size = int(sys.argv[2])
        main(ngram, max_vocab_size)

