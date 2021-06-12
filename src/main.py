import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from Utils import *
from Datasets import LeavesData, Class_To_Num
from tqdm import tqdm
import pandas as pd
import sys, getopt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

conf = Conf()
data_path = conf.get("MyConfig", "data_path")
learning_rate = conf.getfloat("MyConfig", "learning_rate")
weight_decay = conf.getfloat("MyConfig", "weight_decay")
epoch = conf.getint("MyConfig", "epoch")
model_path = conf.get("MyConfig", "model_path")
saveFileName = conf.get("MyConfig", "submission_file")

class_to_num = Class_To_Num(data_path+"train.csv")
num_to_class = {v : k for k, v in class_to_num.items()}

# 加载数据集
train_dataset = LeavesData(data_path, mode="train")
val_dataset = LeavesData(data_path, mode="valid")
test_dataset = LeavesData(data_path, mode="test")

# 定义data loader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2
)

def freezing(model):
    for param in model.parameters():
        param.requires_grad = False

def res_model(num_classer, feature_exttract=False, pretrained=True):
    model_ft = models.resnet34(pretrained=pretrained)
    if feature_exttract:
        freezing(model_ft)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classer))
    return model_ft

# 定义模型、损失函数、优化器
model = res_model(176).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def train():
    # 训练
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    for index in range(epoch):
        model.train()
        train_acc = 0
        train_loss = 0
        for batch in tqdm(train_loader):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc += (logits.argmax(dim=1) == labels).float().sum()
            train_loss += loss.item()

        train_acc /= train_size
        train_loss /= train_size

        print(f"[ Train | {index + 1:03d}/{epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        model.eval()
        val_acc = 0
        val_loss = 0
        for batch in tqdm(val_loader):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                logits = model(imgs)
            loss = loss_fn(logits, labels)
            val_acc += (logits.argmax(dim=1) == labels).float().sum()
            val_loss += loss.item()
        val_acc /= val_size
        val_loss /= val_size
        print(f"[ Valid | {index + 1:03d}/{epoch:03d} ] loss = {val_loss:.5f}, acc = {val_acc:.5f}")

    # 保存模型
    torch.save(model.state_dict(), model_path)
    print('save model to {}'.format(model_path))

def test():
    # 载入模型
    model.load_state_dict(torch.load(model_path))

    model.eval()
    predictions = []
    for batch in tqdm(test_loader):
        imgs = batch
        imgs = imgs.to(device)

        logits = model(imgs)
        predictions.extend(logits.argmax(1).cpu().numpy().tolist())
    
    class_pred = [num_to_class[i] for i in predictions]

    test_data = pd.read_csv(data_path+"sample_submission.csv")
    test_data['label'] = pd.Series(class_pred)
    submission = pd.concat([test_data['image'], test_data['label']], axis=1)
    submission.to_csv(saveFileName, index=False)
    print("Finish!")

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hm:", ["mode="])
    except getopt.GetoptError:
        print("Please enter the following command : [main.py -m <train/test>]")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("Please enter the following command : [main.py -m <train/test>]")
            sys.exit()
        elif opt in ("-m", "--mode"):
            if arg == "train":
                train()
            elif arg == "test":
                test()
            else:
                print("Please enter the following command : [main.py -m <train/test>]")
                sys.exit()

if __name__ == "__main__":
    main(sys.argv[1:])