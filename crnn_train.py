import argparse
from crnn_tool import mapping_seq
from models.crnn.crnn_model import CRNN
from data.datasets import CRNN_Dataset
import itertools
import torch
from torch import nn, optim
from tqdm import tqdm


def save_model(crnn_model, save_path, train_loader, device):
    torch.save(crnn_model.state_dict(), save_path)
    print("----------------------------\n\nsave model : ", save_path)

    acc=[]
    for i, (xx, yy) in enumerate(train_loader):
        xx = xx.to(device)
        yy = [[int(char) for char in text.split(",")] for text in yy]

        preds = crnn_model(xx)
        predict = mapping_seq(preds)

        if i == 0:
            print("\npredict 예시")
            print(predict[:10])

        count = 0
        for yy_i, label in enumerate(yy):
            if predict[yy_i] == label:
                count += 1
        acc.append(float(count)/len(yy))

    print("\naccuracy : ", sum(acc)/len(acc), "\n\n----------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert text to image file")
    parser.add_argument("--load_model", type=str, default="./weights/crnn.pth")
    parser.add_argument("--save_model", type=str, default="./weights/crnn.pth")
    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--n_iter", type=int, default=500)

    args = parser.parse_args()

    print("Data Loading ...\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = CRNN_Dataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize)

    crnn = CRNN(64, 3, len(train_dataset.label_dict), 256).to(device)
    if args.load_model:
        print("load ", args.load_model)
        crnn.load_state_dict(torch.load(args.load_model, map_location=device))

    optimizer = optim.Adam(crnn.parameters())
    loss_fn = nn.CTCLoss()

    print("train start !!!\n==================================================================")

    train_losses = []

    for epoch in range(args.n_iter):
        running_loss = 0.0
        crnn.train()
        n = 0

        for i, (xx, yy) in tqdm(enumerate(train_loader), total=len(train_loader)):
            xx = xx.to(device)
            yy = [[int(char) for char in text.split(",")] for text in yy]
            batch_size = xx.size(0)

            preds = crnn(xx)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            labels = torch.IntTensor(list(itertools.chain.from_iterable(yy)))
            labels_length = torch.IntTensor([len(text) for text in yy])
            loss = loss_fn(preds.cpu(), labels, preds_size, labels_length)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            temp_loss = loss.item()
            running_loss += loss.item()
            n += batch_size
        train_losses.append(running_loss / (i+1))

        print("epoch : ", str(epoch), ",  avg loss : ", train_losses[-1], ",  temp loss : ", temp_loss, flush=True)

        if epoch%25 == 0:
            save_model(crnn, args.save_model, train_loader, device)
        if train_losses[-1] < 0.05:
            break
    
    save_model(crnn, args.save_model, train_loader, device)
    print("\n\n --- finish !! ---")
