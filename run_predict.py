import argparse
import yaml
from tqdm import tqdm
from glob import iglob
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import tifffile as tif
from pathlib import Path

import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
import pytorch_lightning as lit

from dataset import Dataset
from model_lit import LitClassifier

# from utils_get_console_output import get_console_output

# from confusion_matrix import cm_analysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    top_k_accuracy_score,
    cohen_kappa_score,
)
from shutil import rmtree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import rasterio

torch.set_float32_matmul_precision("medium")


def run_predict(config):

    seed_everything(config["parameters"]["seed"], workers=True)

    all_files = list(iglob(config["paths"]["data_path"] + os.sep + "*.tif"))
    print("Number of files:", len(all_files))
    file_df = pd.DataFrame({"file_path": all_files})
    file_df["file_name"] = file_df["file_path"].apply(os.path.basename)
    file_df[
        [
            "city",
            "x0_",
            "x0",
            "x1_",
            "x1",
            "dst_",
            "bg",
            "urban",
            "vegetation",
            "water",
            "slum",
            "ending_",
        ]
    ] = file_df["file_name"].str.split("_", expand=True)
    file_df.drop(
        columns=["x0_", "x1_", "dst_", "ending_"], inplace=True, errors="ignore"
    )
    # Find the maximum values of x0 and x1
    max_x0 = file_df["x0"].astype(int).max()
    max_x1 = file_df["x1"].astype(int).max()

    # Create the 'class' column
    file_df["class_name"] = (
        file_df[["bg", "urban", "vegetation", "water", "slum"]]
        .astype(float)
        .idxmax(axis=1)
    )
    file_df["class"] = file_df["class_name"].map(
        {"bg": 0, "urban": 1, "vegetation": 2, "water": 3, "slum": 4}
    )
    # Change the slum class behaviour for lower pixel percentage threshold

    def adjust_slum_class_threshold(file_df, threshold):
        file_df["class"] = file_df.apply(
            lambda x: (
                4
                if (isinstance(x["slum"], bool) and x["slum"])
                or (isinstance(x["slum"], str) and int(x["slum"]) >= threshold)
                else x["class"]
            ),
            axis=1,
        )
        num_class_4 = len(file_df[file_df["class"] == 4])
        return file_df, num_class_4

    file_df, num_class_4 = adjust_slum_class_threshold(file_df, 10)

    # get class counts sorted
    class_counts = file_df["class"].value_counts()

    # Split the data into train, val, and test
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()
    groups = file_df.groupby("class", group_keys=False)
    for _, group in groups:
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
                n=100, random_state=config["parameters"]["seed"], replace=True
            )
            group_val = group_val.sample(
                n=100, random_state=config["parameters"]["seed"], replace=True
            )
            group_test = group_test.sample(
                n=100, random_state=config["parameters"]["seed"], replace=True
            )
        # _ = pd.concat([df_train, group_train])
        # _ = pd.concat([df_val, group_val])
        df_test = pd.concat([df_test, group_test])
        df_test_indices = df_test.index.tolist()

    print(df_test.head())

    inference_set = Dataset(
        path=file_df["file_path"].tolist(),
        class_labels=file_df["class"].tolist(),
        config=config,
        norm=config["parameters"]["normalization"],
    )

    print("Length of inference_set:", len(inference_set))

    all_logits = []
    slum_logits = []
    for i in range(config["parameters"]["test_time_iterations"]):
        print(
            "Monete Carlo Iteration: {}/{}".format(
                i + 1, config["parameters"]["test_time_iterations"]
            )
        )
        inference_loader = DataLoader(
            inference_set,
            batch_size=config["parameters"]["batch_size"],
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            shuffle=False,
        )
        ckpt_base_path = config["paths"]["data_path"].replace("images", "logs")
        ckpt_base_path = (
            ckpt_base_path
            + "/*_"
            + config["parameters"]["model"]
            + "*/**/**/checkpoints/*.ckpt"
        )
        ckpt_path = list(iglob(ckpt_base_path))[-1]
        ckpt_path = Path(ckpt_path)
        print("Checkpoint path:", ckpt_path)
        model = LitClassifier.load_from_checkpoint(
            checkpoint_path=ckpt_path, config=config
        )

        # Set the model in training mode to enable dropout during prediction
        model.train()

        trainer = lit.Trainer(
            accelerator="gpu", devices=torch.cuda.device_count(), deterministic=False
        )

        logits = trainer.predict(model, inference_loader)

        # Step 1: Concatenate all batched logits into a single tensor
        concatenated_logits = torch.cat(logits, dim=0)

        # Step 2: Convert the concatenated tensor to a NumPy array
        preds_array_mci = concatenated_logits.numpy()

        # Step 3: Perform argmax to get class logits for each item
        preds_class_mci = np.argmax(preds_array_mci, axis=1)

        # Get only slum class activations
        slum_class_mci = preds_array_mci[:, 4]

        # Save the logits for this iteration
        all_logits.append(preds_class_mci)
        slum_logits.append(slum_class_mci)

    # Calculate the mean logits across all iterations
    preds_class_arr, counts_class = stats.mode(all_logits, axis=0)
    preds_class = preds_class_arr.squeeze().tolist()
    preds_class_probability = [
        f / config["parameters"]["test_time_iterations"] for f in counts_class
    ]

    # Calculate the slum probability from monte carlo iterations
    slum_logits = np.array(slum_logits)
    slum_sigmoid = 1 / (1 + np.exp(-slum_logits))
    slum_class_probability = np.mean(slum_sigmoid, axis=0)

    # Get labels
    labels_class = []
    for image, label in inference_set:
        labels_class.append(label.squeeze().tolist())

    # Create a DataFrame
    data = {
        "labels": labels_class,
        "prediction": preds_class,
        "probabilities": preds_class_probability,
        "slum_probabilities": slum_class_probability,
    }
    df_preds = pd.DataFrame(data)

    print(df_preds.head())

    # Inference
    # Create empty array with the same dimensions as the planet image
    planet_path = config["paths"]["planet_path"]
    planet_image = tif.imread(planet_path)
    height, width, _ = planet_image.shape
    print("Planet image shape:", planet_image.shape)

    probability_array = np.full((height, width), np.nan, dtype=np.float32)
    classification_array = np.full((height, width), np.nan, dtype=np.float32)
    label_array = np.full((height, width), np.nan, dtype=np.float32)

    # Add x0 and x1 to the dataframe
    df_preds = df_preds.join(file_df, how="inner")
    print(df_preds.head())

    # Define a function to fill the empty image
    def fill_image(row):
        x0, x1 = row["x0"], row["x1"]
        x0 = int(x0)
        x1 = int(x1)

        # Creating a smaller 2D array of size 224x224 filled with the probability value
        probability_tile = np.full(
            (config["parameters"]["image_size"], config["parameters"]["image_size"]),
            row["slum_probabilities"],
        )
        classification_tile = np.full(
            (config["parameters"]["image_size"], config["parameters"]["image_size"]),
            row["prediction"],
        )
        label_tile = np.full(
            (config["parameters"]["image_size"], config["parameters"]["image_size"]),
            row["labels"],
        )

        # Make sure the placement is within the bounds of the empty image
        if (
            0 <= x0 < height - config["parameters"]["image_size"]
            and 0 <= x1 < width - config["parameters"]["image_size"]
        ):

            # Extract the region from the existing image
            existing_values_probability = probability_array[
                x0 : x0 + config["parameters"]["image_size"],
                x1 : x1 + config["parameters"]["image_size"],
            ]
            existing_values_classification = classification_array[
                x0 : x0 + config["parameters"]["image_size"],
                x1 : x1 + config["parameters"]["image_size"],
            ]
            existing_values_label = label_array[
                x0 : x0 + config["parameters"]["image_size"],
                x1 : x1 + config["parameters"]["image_size"],
            ]

            # Calculate the mean between the existing values and the new values
            overlap_values_probabilities = np.nanmean(
                np.stack([existing_values_probability, probability_tile]), axis=0
            )
            # overlap_values_probabilities = np.nanmax(np.stack([existing_values_probability, probability_tile]), axis=0)
            overlap_values_classification = np.nanmean(
                np.stack([existing_values_classification, classification_tile]), axis=0
            )
            overlap_values_label = np.nanmean(
                np.stack([existing_values_label, label_tile]), axis=0
            )

            # Assign the mean values back to the empty image
            crop_size = config["parameters"]["image_size"] // 5
            crop_start_x0 = x0 + crop_size
            crop_end_x0 = x0 + config["parameters"]["image_size"] - crop_size
            crop_start_x1 = x1 + crop_size
            crop_end_x1 = x1 + config["parameters"]["image_size"] - crop_size
            probability_array[crop_start_x0:crop_end_x0, crop_start_x1:crop_end_x1] = (
                overlap_values_probabilities[
                    crop_start_x0 - x0 : crop_end_x0 - x0,
                    crop_start_x1 - x1 : crop_end_x1 - x1,
                ]
            )
            classification_array[
                crop_start_x0:crop_end_x0, crop_start_x1:crop_end_x1
            ] = overlap_values_classification[
                crop_start_x0 - x0 : crop_end_x0 - x0,
                crop_start_x1 - x1 : crop_end_x1 - x1,
            ]
            label_array[crop_start_x0:crop_end_x0, crop_start_x1:crop_end_x1] = (
                overlap_values_label[
                    crop_start_x0 - x0 : crop_end_x0 - x0,
                    crop_start_x1 - x1 : crop_end_x1 - x1,
                ]
            )

    # Apply the function to each row of the DataFrame
    df_preds.apply(lambda row: fill_image(row), axis=1)

    # Save the image with the same georeference as planet raster using rasterio
    # Open the planet raster file
    results_arrays = [
        classification_array,
        label_array,
    ]  # [probability_array, classification_array, label_array]
    result_array_names = ["classification", "label"]
    for i in range(len(results_arrays)):
        base_path, _ = os.path.split(config["paths"]["planet_path"])
        output_path = os.path.join(
            base_path,
            result_array_names[i] + "_" + str(config["parameters"]["model"]) + ".tif",
        )
        with rasterio.open(planet_path) as src:
            # Create a new raster file with the same georeference as the planet raster
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                width=src.width,
                height=src.height,
                count=1,
                dtype=results_arrays[i].dtype,
                crs=src.crs,
                transform=src.transform,
            ) as dst:
                # Write the probability_array to the new raster file
                dst.write(results_arrays[i], 1)

    # only get predictions for the test set using the index from df_tes_indices

    labels_class = [labels_class[i] for i in df_test_indices]
    preds_class = [preds_class[i] for i in df_test_indices]

    # calculate accuracy metrics only for df_test2 (latter half of the data)
    acc_balanced = balanced_accuracy_score(
        labels_class[len(labels_class) // 2 :], preds_class[len(preds_class) // 2 :]
    )
    acc_kappa = cohen_kappa_score(
        labels_class[len(labels_class) // 2 :], preds_class[len(preds_class) // 2 :]
    )
    print("Overall Accuracy:  {:.2f}".format(acc_balanced * 100))
    print("Overall Kappa:  {:.2f}".format(acc_kappa * 100))

    # Filter the predictions and labels for class 4
    preds_slum = [1 if f == 4 else 0 for f in preds_class]
    labels_slum = [1 if f == 4 else 0 for f in labels_class]

    # Calculate accuracy
    accuracy = accuracy_score(
        labels_slum[len(labels_slum) // 2 :], preds_slum[len(preds_slum) // 2 :]
    )

    # Calculate precision
    precision = precision_score(
        labels_slum[len(labels_slum) // 2 :], preds_slum[len(preds_slum) // 2 :]
    )

    # Calculate recall
    recall = recall_score(
        labels_slum[len(labels_slum) // 2 :], preds_slum[len(preds_slum) // 2 :]
    )

    # Calculate F1 score
    f1 = f1_score(
        labels_slum[len(labels_slum) // 2 :], preds_slum[len(preds_slum) // 2 :]
    )

    print("Class 4 Accuracy: {:.2f}".format(accuracy * 100))
    print("Class 4 Precision: {:.2f}".format(precision * 100))
    print("Class 4 Recall: {:.2f}".format(recall * 100))
    print("Class 4 F1 Score: {:.2f}".format(f1 * 100))

    # Write the print statements to a text file
    base_path, _ = os.path.split(config["paths"]["planet_path"])
    output_file = os.path.join(
        base_path, "accuaracy_metrics_" + str(config["parameters"]["model"]) + ".txt"
    )
    with open(output_file, "w") as f:
        f.write("Overall Accuracy: {:.2f}\n".format(acc_balanced * 100))
        f.write("Overall Kappa: {:.2f}\n".format(acc_kappa * 100))
        f.write("Class 4 Accuracy: {:.2f}\n".format(accuracy * 100))
        f.write("Class 4 Precision: {:.2f}\n".format(precision * 100))
        f.write("Class 4 Recall: {:.2f}\n".format(recall * 100))
        f.write("Class 4 F1 Score: {:.2f}\n".format(f1 * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/mnt/data1/2023_P3_TDGUP/data/transfer_data/Abidjan_00076_21602/config_predict_resnet18.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # get predictions
    run_predict(config)

    # remove lightninglogdir
    lighntinglogdir = os.path.join(os.getcwd(), "lightning_logs")
    rmtree(lighntinglogdir)
