import sentencepiece as spm
import torch
from utils.base_utils import Config
from utils.loader import MovieDataSet
from utils.preprocess_utils import movie_collate_fn
from models.transformer import MovieClassification
from evaluate import eval_epoch
from train import train_epoch
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import argparse

DATA_DIR = './dataset/transformer-evolution'


def main():
    parser = argparse.ArgumentParser(description='PyTorch High Performance Content Project')
    parser.add_argument('--batch_size', type=int, default=640, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--n_epoch', type=int, default=3000, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-03, metavar='LR',
                        help='learning rate (default: 0.01)')

    args = parser.parse_args()

    vocab_file = f"{DATA_DIR}/kowiki.model"
    vocab = spm.SentencePieceProcessor()
    vocab.load(vocab_file)

    train_dataset = MovieDataSet(vocab, f"{DATA_DIR}/ratings_train.json")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, collate_fn=movie_collate_fn)
    test_dataset = MovieDataSet(vocab, f"{DATA_DIR}/ratings_test.json")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False, collate_fn=movie_collate_fn)

    config = Config({
        "n_enc_vocab": len(vocab),
        "n_dec_vocab": len(vocab),
        "n_enc_seq": 256,
        "n_dec_seq": 256,
        "n_layer": 6,
        "d_hidn": 256,
        "i_pad": 0,
        "d_ff": 1024,
        "n_head": 4,
        "d_head": 64,
        "dropout": 0.1,
        "layer_norm_epsilon": 1e-12
    })
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.n_output = 2
    print(config)

    model = MovieClassification(config)
    model.to(config.device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_epoch, best_loss, best_score = 0, 0, 0
    losses, scores = [], []
    for epoch in range(args.n_epoch):
        loss = train_epoch(config, epoch, model, criterion, optimizer, train_loader)
        score = eval_epoch(config, model, test_loader)

        losses.append(loss)
        scores.append(score)

        if best_score < score:
            best_epoch, best_loss, best_score = epoch, loss, score
    print(f">>>> epoch={best_epoch}, loss={best_loss:.5f}, socre={best_score:.5f}")

    # table
    data = {
        "loss": losses,
        "score": scores
    }
    df = pd.DataFrame(data)
    display(df)

    # graph
    plt.figure(figsize=[12, 4])
    plt.plot(losses, label="loss")
    plt.plot(scores, label="score")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.show()


if __name__ == "__main__":
    main()
