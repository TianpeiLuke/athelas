import dataclasses

import polars as pl
import torch
import torch.nn as nn
import torch.utils.data as data

from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import OneCycleLR
from torcheval.metrics import BinaryAUROC
from tqdm import tqdm

import lstm2risk
import transformer2risk

from dataset import TextDataset, TabularDataset
from logger import JSONLogger
from tokenizer import OrderTextTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

BATCH_SIZE = 512
EPOCHS = 10

REQUIRED_FEATURES = {
    "objectId",
    "customerId",
    "transactionDate",
    "marketplaceCountryCode",
    "emailAddress",
    "billingAddressName",
    "customerName",
    "paymentAccountHolderName",
    "emailDomain",
    "status",
    "orderDate",
    "daysSinceFirstCompletedOrder",
}


def train_loop(model, dataloader, loss_fn, optimizer, scheduler):
    model.train()
    auc = BinaryAUROC()
    pbar = tqdm(dataloader)

    for batch in pbar:
        batch_auc = BinaryAUROC()
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        pred = model(batch)

        loss = loss_fn(pred, batch["label"])

        auc.update(pred.view(-1), batch["label"].view(-1))
        batch_auc.update(pred.view(-1), batch["label"].view(-1))

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        lr, *_ = scheduler.get_last_lr()
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.3f}",
                "lr": f"{lr:.1e}",
                "auc": f"{batch_auc.compute():.3f}",
            }
        )

    return auc.compute()


@torch.no_grad()
def test_loop(model, dataloader):
    model.eval()
    auc = BinaryAUROC()

    for batch in tqdm(dataloader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        pred = model(batch)
        auc.update(pred.view(-1), batch["label"].view(-1))

    return auc.compute()


def main():

    tabular_features = set()

    with open("features/DigitalModelNA.txt") as file:
        tabular_features.update(line.strip() for line in file)

    with open("features/DigitalModelEU.txt") as file:
        tabular_features = tabular_features.intersection({line.strip() for line in file})

    with open("features/DigitalModelJP.txt") as file:
        tabular_features = tabular_features.intersection({line.strip() for line in file})

    features = REQUIRED_FEATURES.union(tabular_features)

    print("Loading data...")

    df = (
        pl.concat(
            [
                pl.scan_parquet("data/2025-08-20T00-40-17_NA.parquet")
                .select(features)
                .with_columns(region=pl.lit("NA")),
                pl.scan_parquet("data/2025-08-20T00-47-40_EU.parquet")
                .select(features)
                .with_columns(region=pl.lit("EU")),
                pl.scan_parquet("data/2025-08-20T00-50-46_FE.parquet")
                .select(features)
                .with_columns(region=pl.lit("FE")),            
            ],
            how="vertical_relaxed",
        )
        .filter(
            ~pl.col("emailDomain").str.to_lowercase().str.contains("amazon."),
        )
        .with_columns(
            label=pl.when(pl.col("status").is_in(["F", "I"]))
            .then(1)
            .when(pl.col("status") == "N")
            .then(0)
            .otherwise(-1),
            text=pl.concat_str(
                [
                    pl.col("emailAddress").fill_null("[MISSING]"),
                    pl.col("billingAddressName").fill_null("[MISSING]"),
                    pl.col("customerName").fill_null("[MISSING]"),
                    pl.col("paymentAccountHolderName").fill_null("[MISSING]"),
                ],
                separator="|",
            ),
        )
        .filter(
            pl.col("label") >= 0,
        )
        .sort("orderDate")
        .unique(
            "customerId", keep="first"
        )  # the sql query only the first order by region. this will dedup WW.
        .collect(engine="streaming")
    )

    print(
        df
        .lazy()
        .group_by("marketplaceCountryCode")
        .agg(pl.len())
        .with_columns(ratio=pl.col("len")/pl.sum("len"))
        .sort("len", descending=True)
        .collect()
    )

    print(
        df
        .lazy()
        .group_by("label")
        .agg(pl.len())
        .with_columns(ratio=pl.col("len")/pl.sum("len"))
        .collect()
    )

    tabular_features = {
        feature for feature in tabular_features if df[feature].dtype.is_numeric()
    }

    df_train, df_test = train_test_split(
        df.sort("transactionDate"), test_size=0.05, shuffle=False
    )

    print("Training tokenizer...")

    config = lstm2risk.LSTMConfig(n_tab_features=len(tabular_features))
    # config = transformer2risk.TransformerConfig(n_tab_features=len(tabular_features))
    
    tokenizer = OrderTextTokenizer().train(
        df_train
        .get_column("text").to_list()
    )

    config.n_embed = tokenizer.vocab_size

    print(f"Vocab size: {config.n_embed}")

    training_dataset = data.StackDataset(
        text=TextDataset(df_train.get_column("text").to_list(), tokenizer),
        tabular=TabularDataset(
            df_train.select(pl.col(tabular_features).fill_null(-1)).to_torch(
                dtype=pl.Float32
            )
        ),
        label=TabularDataset(df_train.select("label").to_torch(dtype=pl.Float32)),
    )

    testing_dataset = data.StackDataset(
        text=TextDataset(df_test.get_column("text").to_list(), tokenizer),
        tabular=TabularDataset(
            df_test.select(pl.col(tabular_features).fill_null(-1)).to_torch(
                dtype=pl.Float32
            )
        ),
        label=TabularDataset(df_test.select("label").to_torch(dtype=pl.Float32)),
    )

    model = lstm2risk.LSTM2Risk(config).to(DEVICE)
    # model = transformer2risk.Transformer2Risk(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = nn.BCELoss()

    training_dataloader, testing_dataloader = (
        data.DataLoader(
            training_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=model.create_collate_fn(tokenizer.pad_token),
        ),
        data.DataLoader(
            testing_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=model.create_collate_fn(tokenizer.pad_token),
        ),
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=EPOCHS * len(training_dataloader),
        pct_start=0.1,
    )

    logger = JSONLogger()

    logger.log_params(
        dataclasses.asdict(config)
    )

    print(f"Number of Parameters: {sum(p.numel() for p in model.parameters()):,} (Text proj: {sum(p.numel() for p in model.text_projection.parameters()):,}, Tabular proj: {sum(p.numel() for p in model.tab_projection.parameters()):,})")

    for epoch in range(EPOCHS):

        train_auc = train_loop(
            model, training_dataloader, loss_fn, optimizer, scheduler
        )
        test_auc = test_loop(model, testing_dataloader)
        lr, *_ = scheduler.get_last_lr()

        torch.save(model.state_dict(), f"models/model_{epoch}.pt")

        print(f"Epoch {epoch}: {train_auc:.4f} {test_auc:.4f}, {lr:.1e}")


        mp_auc = {}
        for (mp,), df_mp in df_test.group_by("marketplaceCountryCode"):
            dataset = data.StackDataset(
                text=TextDataset(df_mp.get_column("text").to_list(), tokenizer),
                tabular=TabularDataset(
                    df_mp.select(pl.col(tabular_features).fill_null(-1)).to_torch(
                        dtype=pl.Float32
                    )
                ),
                label=TabularDataset(df_mp.select("label").to_torch(dtype=pl.Float32)),
            )

            dataloader = data.DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=model.create_collate_fn(tokenizer.pad_token),
            )

            mp_auc[f"{mp}_auc"] = test_loop(model, dataloader).item()

        logger.log_metrics(
            epoch,
            lr=lr,
            train_auc=train_auc.item(),
            test_auc=test_auc.item(),
            **mp_auc
        )


if __name__ == "__main__":
    main()
