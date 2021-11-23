import os
import cv2
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import albumentations as A
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer

from CFG import CFG
from AvgMeter import AvgMeter
from CLIPDataset import CLIPDataset
from Encoders import ImageEncoder, TextEncoder
from ProjectionHead import ProjectionHead
from CLIPModel import CLIPModel


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )


def preprocess_desc(df):
    """wash dataset"""
    unwanted_words = ["auchan"]  # list of unwanted words (auchan, etc)
    for i in range(len(df.shape[0])):
        df['caption'][i] = df['caption'][i].lower()
        for word in unwanted_words:
            df['caption'][i] = df['caption'][i].replace(word, '')


def make_train_valid_dfs():
    # dataframe = pd.DataFrame(image_filenames, columns=["image"])
    # dataframe["id"] = [id_ for id_ in range(dataframe.shape[0])]
    # dataframe["caption"] = [img[:img.index('_')] for img in image_filenames]
    filenames = os.listdir(CFG.path)
    dataframe = pd.read_csv(CFG.csv_path)
    dataframe = dataframe[dataframe.image.isin(filenames)]
    dataframe.dropna(subset = ["caption"], inplace = True)

    # max_id = dataframe["id"].max() + 1 if not CFG.debug else 100
    # image_ids = np.arange(0, max_id)

    # dataframe = preprocess_desc(dataframe)
    image_ids = dataframe['id']
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(dataframe.shape[0])), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=1,

        shuffle=True if mode == "train" else False,
    )
    return dataloader


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

    if (CFG.train == True):
        model = CLIPModel().to(CFG.device)
        params = [
            {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
            {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
            {"params": itertools.chain(
                model.image_projection.parameters(), model.text_projection.parameters()
            ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=0.)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
        )
        step = "epoch"
        # model = nn.DataParallel(model)

        best_loss = float('inf')
        for epoch in range(CFG.epochs):
            print(f"Epoch: {epoch + 1}")
            model.train()
            train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
            model.eval()
            with torch.no_grad():
                valid_loss = valid_epoch(model, valid_loader)

            if valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                torch.save(model.state_dict(), "best.pt")
                print("Saved Best Model!")

            lr_scheduler.step(valid_loss.avg)


def get_image_embeddings(valid_df, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()

    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings)


def find_matches(model, image_embeddings, query, image_filenames, n=9):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T

    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]

    _, axes = plt.subplots(3, 1, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        print(os.path.join(CFG.path, match[:match.index('_')], match))
        image = cv2.imread(os.path.join(CFG.path, match[:match.index('_')], match))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")

    plt.show()


if __name__ == '__main__':
    main()
    # _, valid_df = make_train_valid_dfs()
    # model, image_embeddings = get_image_embeddings(valid_df, "best.pt")
    # find_matches(model,
    #                 image_embeddings,
    #                 query="Lait",
    #                 image_filenames=valid_df['image'].values,
    #                 n=3)
