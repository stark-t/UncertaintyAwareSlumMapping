import argparse
import os
from glob import glob
import yaml
import numpy as np
from tqdm import tqdm
import tifffile as tif
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def image_padding(img, image_size):
    """
    Pad the image to the specified image size.
    :param img: The input image.
    :param image_size: The desired image size.
    :return: The padded image.
    """
    if len(img.shape) == 3:  # Check if the image has multiple channels
        img_pad = np.zeros((image_size, image_size, img.shape[2]), dtype=np.uint8)
        for dim in range(img.shape[2]):
            img_pad[: img.shape[0], : img.shape[1], dim] = img[:, :, dim]
    else:  # If the image is 2D, assume it has a single channel
        img_pad = np.zeros((image_size, image_size), dtype=np.uint8)
        img_pad[: img.shape[0], : img.shape[1]] = img

    return img_pad


def split_raster(
    planet_raster_path,
    lcz_raster_path,
    reference_raster_path,
    output_path,
    overlap,
    config,
):
    """Split a raster into multiple smaller parts."""
    # Create the output directory
    os.makedirs(output_path, exist_ok=True)

    # Get the city name
    city_name = planet_raster_path.split(os.sep)[-1].split("_")[1].split(".")[0]

    # read rasters
    planet_raster = tif.imread(planet_raster_path)
    # !TODO line 47 to 54 is only for example data!
    if city_name == "caracas":
        planet_raster = planet_raster[:, :, [0, 1, 2]]
        planet_raster = np.where(planet_raster == 255, np.nan, planet_raster)
        planet_raster = cv2.normalize(planet_raster, None, 0, 255, cv2.NORM_MINMAX)
        planet_raster = np.where(np.isnan(planet_raster), 255, planet_raster)
    else:
        planet_raster = planet_raster[:, :, [3, 2, 1]]
        planet_raster = np.where(planet_raster == 65535, np.nan, planet_raster)
        planet_raster = cv2.normalize(planet_raster, None, 0, 255, cv2.NORM_MINMAX)
        planet_raster = np.where(np.isnan(planet_raster), 255, planet_raster)

    lcz_raster = tif.imread(lcz_raster_path)
    lcz_raster = lcz_raster.astype(np.uint8)
    reference_raster = tif.imread(reference_raster_path)
    reference_raster = reference_raster.astype(np.uint8)

    # get image stats
    image_stats = []
    for i in range(planet_raster.shape[-1]):
        image_mean_ch = np.mean(planet_raster[:, :, i])
        image_std_ch = np.std(planet_raster[:, :, i])

        record = {
            "city": city_name,
            "channel": i,
            "mean": image_mean_ch,
            "std": image_std_ch,
        }

        image_stats.append(record)

    # write image stats to dataframe and save to csv in output_path
    image_stats_df = pd.DataFrame(image_stats)

    # save dataframe to CSV
    image_stats_csv_path = os.path.join(output_path, f"{city_name}_image_stats.csv")
    image_stats_df.to_csv(image_stats_csv_path, index=False)

    i = 0
    class_distribution_records = []
    for ax0 in tqdm(
        range(
            0,
            int(planet_raster.shape[0]),
            int(config["parameters"]["image_size"] / overlap),
        )
    ):
        for ax1 in range(
            0,
            int(planet_raster.shape[1]),
            int(config["parameters"]["image_size"] / overlap),
        ):
            # new iterator id
            i += 1
            # get image content
            img = planet_raster[
                ax0 : ax0 + config["parameters"]["image_size"],
                ax1 : ax1 + config["parameters"]["image_size"],
                0:3,
            ]
            img_lcz = lcz_raster[
                ax0 : ax0 + config["parameters"]["image_size"],
                ax1 : ax1 + config["parameters"]["image_size"],
            ]
            img_ref = reference_raster[
                ax0 : ax0 + config["parameters"]["image_size"],
                ax1 : ax1 + config["parameters"]["image_size"],
            ]

            # Skip if no lcz data
            if 0 in img_lcz:
                continue

            # handle uneven images
            if img.shape[:-1] != (
                config["parameters"]["image_size"],
                config["parameters"]["image_size"],
            ):
                # zero padding uneven image-tile
                img = image_padding(img, config["parameters"]["image_size"])
                img_lcz = image_padding(img_lcz, config["parameters"]["image_size"])
                img_ref = image_padding(img_ref, config["parameters"]["image_size"])

            # Change the LCZ classes to 4 classes: Background, Urban, Vegetation, Water
            img_lcz_reclassified = np.copy(img_lcz)
            img_lcz_reclassified[
                (img_lcz_reclassified >= 1) & (img_lcz_reclassified <= 10)
            ] = 1
            img_lcz_reclassified[
                (img_lcz_reclassified >= 11) & (img_lcz_reclassified <= 16)
            ] = 2
            img_lcz_reclassified[img_lcz_reclassified == 17] = 3

            # Change Slums class to 4
            img_ref[img_ref > 0] = 4

            # Combine the two labels
            combined_label_array = np.copy(img_lcz_reclassified)
            combined_label_array[img_ref == 4] = 4

            # Get label distribution
            classes, label_dist = np.unique(combined_label_array, return_counts=True)
            class_distribution = {class_num: 0 for class_num in range(5)}

            for class_num, dist in zip(classes, label_dist):
                class_distribution[class_num] = dist / (
                    config["parameters"]["image_size"] ** 2
                )

            # write image tile to dataset
            label_dist_str = [
                (str(int(f * 100)) + "_") for f in list(class_distribution.values())
            ]
            label_dist_str = "".join(label_dist_str)
            image_name = (
                city_name
                + "_x0_"
                + str(ax0)
                + "_x1_"
                + str(ax1)
                + "_dst_"
                + label_dist_str
                + ".tif"
            )

            if not os.path.exists(os.path.join(output_path, "images")):
                os.makedirs(os.path.join(output_path, "images"))
            image_path = os.path.join(output_path, "images", image_name)
            tif.imwrite(image_path, img)

            # Write class distribution and info to dataframe
            class_distribution_record = {
                "city": city_name,
                "image_name": image_path,
                "class_distribution": list(class_distribution.values()),
            }
            class_distribution_records.append(class_distribution_record)

    class_distribution_df = pd.DataFrame(class_distribution_records)

    # save dataframe to CSV
    class_distribution_csv_path = os.path.join(
        output_path, f"{city_name}_class_distribution.csv"
    )
    class_distribution_df.to_csv(class_distribution_csv_path, index=False)


def main(args):
    """Main function."""

    # Read the YAML file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    data_paths = glob(args.data_path + os.sep + "*")

    # Get the aoi rasters
    planet_raster = [f for f in data_paths if "planet" in f.lower()][0]
    lcz_raster = [f for f in data_paths if "lcz" in f.lower()][0]

    # Get the reference raster
    reference_raster = [f for f in data_paths if "reference" in f.lower() and ".tif" in f][0]

    overlap = 5  # Set the overlap value here
    split_raster(
        planet_raster, lcz_raster, reference_raster, args.output_path, overlap, config
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script with two path inputs.")

    parser.add_argument(
        "--config",
        type=str,
        help="Config yaml file.",
        default="/mnt/ushelf_star_th/projects/2023_TDGUP_Dissertation/2023_P3_TDGUP/UncertaintyAwareSlumMapping/config_transfer.yaml",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the first input file.",
        default="/mnt/ushelf_star_th/projects/2023_TDGUP_Dissertation/2023_P3_TDGUP/UncertaintyAwareSlumMapping/data/mumbai", # /.../UncertaintyAwareSlumMapping/data/caracas
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the first input file.",
        default="/mnt/ushelf_star_th/projects/2023_TDGUP_Dissertation/2023_P3_TDGUP/UncertaintyAwareSlumMapping/data/transfer", # /.../UncertaintyAwareSlumMapping/data/pretrain
    )

    args = parser.parse_args()
    main(args)
