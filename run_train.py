import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from glob import iglob
from sklearn.model_selection import train_test_split

import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

import pytorch_lightning as lit
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar

from dataset import Dataset
from model_lit import LitClassifier

torch.set_float32_matmul_precision("medium")
import matplotlib.pyplot as plt


torch.set_float32_matmul_precision("medium")

def main(config, log_path="path_to_log_dir"):
    # Set seed for reproducibility
    seed_everything(config["parameters"]["seed"], workers=True)

    # Get a list of all files in the data path
    all_files = list(iglob(config["paths"]["data_path"] + os.sep + "*.tif"))

    # Create a DataFrame to store file information
    file_df = pd.DataFrame({"file_path": all_files})

    # Extract file name components and create columns in the DataFrame
    file_df["file_name"] = file_df["file_path"].apply(os.path.basename)
    file_df[["city", "x0_", "x0", "x1_", "x1", "dst_", "bg", "urban", "vegetation", "water", "slum", "ending_"]] = file_df["file_name"].str.split("_", expand=True)
    file_df.drop(columns=["x0_", "x1_", "dst_", "ending_"], inplace=True, errors="ignore")

    # Create the 'class' column based on the maximum value among 'bg', 'urban', 'vegetation', 'water', and 'slum'
    file_df["class_name"] = (file_df[["bg", "urban", "vegetation", "water", "slum"]].astype(float)).idxmax(axis=1)
    file_df["class"] = file_df["class_name"].map({"bg": 0, "urban": 1, "vegetation": 2, "water": 3, "slum": 4})

    # Adjust the 'slum' class threshold and update the 'class' column accordingly
    def adjust_slum_class_threshold(file_df, threshold):
        file_df["class"] = file_df.apply(
            lambda x: (
                4 if (isinstance(x["slum"], bool) and x["slum"]) or (isinstance(x["slum"], str) and int(x["slum"]) >= threshold)
                else x["class"]
            ),
            axis=1,
        )
        num_class_4 = len(file_df[file_df["class"] == 4])
        return file_df, num_class_4

    # Adjust the slum class threshold to 10 and check the number of class 4 samples
    file_df, num_class_4 = adjust_slum_class_threshold(file_df, 10)

    # If the number of class 4 samples is less than or equal to 100, adjust the threshold to 5
    if num_class_4 <= 100:
        file_df, num_class_4 = adjust_slum_class_threshold(file_df, 5)

    # Get class counts sorted
    class_counts = file_df["class"].value_counts()

    # Check if any class has fewer than 10 unique items
    for class_label, class_count in class_counts.items():
        if class_count < 10:
            # Get all rows of class_label
            class_rows = file_df[file_df["class"] == class_label]
            # Concatenate class_rows four times to file_df
            file_df = pd.concat([file_df, class_rows, class_rows, class_rows, class_rows])

    # Drop all rows with class 0
    file_df = file_df[file_df["class"] != 0]

    # Print final class counts sorted
    class_counts = file_df["class"].value_counts().sort_index()
    print("Class Counts:")
    for class_label, count in class_counts.items():
        print(f"Class {class_label} count: {count}")

    # Split the data into train, val, and test
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()
    groups = file_df.groupby("class", group_keys=False)
    for class_id, group in groups:
        if class_id == 4:
            n_samples = 100
        else:
            n_samples = 33
        group_train, group_temp = train_test_split(group, test_size=0.4, random_state=config["parameters"]["seed"])
        group_val, group_test = train_test_split(group_temp, test_size=0.5, random_state=config["parameters"]["seed"])
        # If pretrain imbalanced & if transfer balanced class dataset
        if isinstance(config["parameters"]["pretrained"], str) and not config["parameters"]["pretrained"].lower() == "true":
            group_train = group_train.sample(n=n_samples, random_state=config["parameters"]["seed"], replace=True)
            group_val = group_val.sample(n=n_samples, random_state=config["parameters"]["seed"], replace=True)
            group_test = group_test.sample(n=n_samples, random_state=config["parameters"]["seed"], replace=True)
        df_train = pd.concat([df_train, group_train])
        df_val = pd.concat([df_val, group_val])
        df_test = pd.concat([df_test, group_test])

    # Print class counts
    print("There are {} slum tiles in df_train:", len(df_train[df_train["class"] == 4]))
    print("Class 4 counts in df_val:", len(df_val[df_val["class"] == 4]))
    # Drop all rows with class 0
    file_df = file_df[file_df["class"] != 0]

    # print fial class counts sorted
    class_counts = file_df["class"].value_counts().sort_index()
    print("Class Counts:")
    for class_label, count in class_counts.items():
        print(f"Class {class_label} count: {count}")

    # Split the data into train, val, and test
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()
    groups = file_df.groupby("class", group_keys=False)
    for class_id, group in groups:
        if class_id == 4:
            n_samples = 100
        else:
            n_samples = 33
        group_train, group_temp = train_test_split(
            group, test_size=0.4, random_state=config["parameters"]["seed"]
        )
        group_val, group_test = train_test_split(
            group_temp, test_size=0.5, random_state=config["parameters"]["seed"]
        )
        # if pretrain imbalanced & if transfer balanced class dataset
        if (
            isinstance(config["parameters"]["pretrained"], str)
            and not config["parameters"]["pretrained"].lower() == "true"
        ):
            group_train = group_train.sample(
                n=n_samples, random_state=config["parameters"]["seed"], replace=True
            )
            group_val = group_val.sample(
                n=n_samples, random_state=config["parameters"]["seed"], replace=True
            )
            group_test = group_test.sample(
                n=n_samples, random_state=config["parameters"]["seed"], replace=True
            )
        df_train = pd.concat([df_train, group_train])
        df_val = pd.concat([df_val, group_val])
        df_test = pd.concat([df_test, group_test])

    # Print class counts
    print("There are {} slum tiles in df_train:", len(df_train[df_train["class"] == 4]))
    print("Class 4 counts in df_val:", len(df_val[df_val["class"] == 4]))
    print("Class 4 counts in df_test:", len(df_test[df_test["class"] == 4]))

    # Get data
    train_set = Dataset(
        path=df_train["file_path"].tolist(),
        class_labels=df_train["class"].tolist(),
        config=config,
        norm=config["parameters"]["normalization"],
    )
    val_set = Dataset(
        path=df_val["file_path"].tolist(),
        class_labels=df_val["class"].tolist(),
        config=config,
        norm=config["parameters"]["normalization"],
    )
    test_set = Dataset(
        path=df_test["file_path"].tolist(),
        class_labels=df_test["class"].tolist(),
        config=config,
        norm=config["parameters"]["normalization"],
    )

    if config["parameters"]["verbose"] >= 1:
        grouped_df = df_train.groupby("class").nunique()
        randints = torch.randint(low=0, high=len(train_set), size=(10,))
        for i in randints:
            image, label = train_set[i]
            plt.imshow(image.numpy()[0, :, :])
            plt.title(str(label))
            plt.show()
            plt.close()

    # Initialize DataLoader
    n_cpu = os.cpu_count()
    batch_size = config["parameters"]["batch_size"]
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=10,
        persistent_workers=False,
        pin_memory=False,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=10,
        persistent_workers=False,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=10,
        persistent_workers=False,
        pin_memory=False,
    )

    # Logger and Callbacks
    if not config["parameters"]["progressbar"]:

        class NoValidationProgressBar(TQDMProgressBar):
            def init_validation_tqdm(self):
                bar = super().init_validation_tqdm()
                bar.disable = True
                return bar

        tqdm_refreshrate = int((len(train_set) / batch_size) / 5)

    if config["parameters"]["pretrained"]:
        name_str = config["parameters"]["model"]
    else:
        name_str = config["parameters"]["model"] + file_df["city"][0]

    logger = TensorBoardLogger(
        save_dir=log_path,
        name=name_str,
        # version=version_int,
    )

    checkpoint = lit.callbacks.ModelCheckpoint(
        save_top_k=1,
        every_n_epochs=None,
        every_n_train_steps=None,
        train_time_interval=None,
        save_on_train_epoch_end=True,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.01,
        patience=config["parameters"]["epochs"] / 3,
        verbose=True,
        mode="min",
    )
    if not config["parameters"]["progressbar"]:
        callbacks = [
            early_stop_callback,
            checkpoint,
            TQDMProgressBar(refresh_rate=tqdm_refreshrate),
        ]
    else:
        callbacks = [early_stop_callback, checkpoint]

    # Model
    if config["parameters"]["transfer"] and not config["parameters"]["pretrained"]:
        ckpt_path = list(
            iglob(
                os.path.join(
                    config["paths"]["pretrain_base_dir"],
                    "*_" + config["parameters"]["model"],
                    "**",
                    "**",
                    "checkpoints",
                    "*.ckpt",
                )
            )
        )[0]
        print("Checkpoint path:", ckpt_path)
        model = LitClassifier.load_from_checkpoint(ckpt_path, config=config)
    else:
        model = LitClassifier(config=config)

    # Trainer
    trainer = lit.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        max_epochs=config["parameters"]["epochs"],
        logger=logger,
        callbacks=callbacks,
        deterministic=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/mnt/data1/2023_P3_TDGUP/data/transfer_data/Abidjan_00076_21602/config_transfer_resnet18.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # create log info
    model_name = config["parameters"]["model"]
    if config["parameters"]["transfer"]:
        city_name = config["paths"]["log_path"].split("/")[-1].split("_")[0].lower()
        log_name = "transfer_" + model_name + "_" + city_name
    else:
        log_name = "pretrained_" + model_name

    log_path = os.path.join(config["paths"]["log_path"], "logs", log_name)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
        # run training
    main(config, log_path)
