import os
import glob
import matplotlib.pyplot as plt


if __name__ == "__main__":
    inputDir = "../Images"
    resultsDir = "../Results"
    outputDir = "../report"

    if not os.path.exists(resultsDir):
        raise FileNotFoundError("Results directory not found. Please run the main.py file first.")
    
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    images_path = sorted(glob.glob(inputDir + "/image_*.jpg"))

    for image_path in images_path:
        image_name = os.path.basename(image_path)
        image_number = image_name.split("_")[1].split(".")[0]

        result_images = glob.glob("{}/result_{}_*.jpg".format(resultsDir, image_number))
        result_images = [result for result in result_images if "mask" not in result]
        
        im1 = plt.imread(image_path)
        im2 = plt.imread(result_images[0])
        im3 = plt.imread(result_images[1])

        if im1.shape[0] > im2.shape[0]:
            reduce_height_image = im2
            reduce_width_image = im3
        else:
            reduce_height_image = im3
            reduce_width_image = im2

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(im1)
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(reduce_width_image)
        ax[1].set_title("widthFac: {:.1f}, heightFac: 1".format(im1.shape[1] / reduce_width_image.shape[1]))
        ax[1].axis("off")

        ax[2].imshow(reduce_height_image)
        ax[2].set_title("widthFac: 1, heightFac: {:.1f}".format(im1.shape[0] / reduce_height_image.shape[0]))
        ax[2].axis("off")

        plt.suptitle(f"Image {image_number}")
        plt.tight_layout()
        plt.savefig(f"{outputDir}/comparison_{image_number}.png")
        plt.close()

    # Extra Credit Part
    sample_id = 4
    sample_image = plt.imread("{}/image_{:02d}.jpg".format(inputDir, sample_id))
    sample_results = glob.glob("{}/result_{:02d}_*.jpg".format(resultsDir, sample_id))
    sample_results_masked = [plt.imread(result) for result in sample_results if "mask" in result]
    sample_results_no_mask = [plt.imread(result) for result in sample_results if "mask" not in result]

    mask = plt.imread("{}/mask_{:02d}.jpg".format(inputDir, sample_id))

    fig, ax = plt.subplots(2, 3, figsize=(15, 5))
    ax[0, 0].imshow(sample_image)
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis("off")

    if sample_results_no_mask[0].shape[0] > sample_results_no_mask[1].shape[0]:
        reduce_width_image = sample_results_no_mask[0]
        reduce_height_image = sample_results_no_mask[1]
    else:
        reduce_width_image = sample_results_no_mask[1]
        reduce_height_image = sample_results_no_mask[0]

    ax[0, 1].imshow(reduce_width_image)
    ax[0, 1].set_title("widthFac: {:.1f}, heightFac: 1".format(sample_image.shape[1] / reduce_width_image.shape[1]))
    ax[0, 1].axis("off")

    ax[0, 2].imshow(reduce_height_image)
    ax[0, 2].set_title("widthFac: 1, heightFac: {:.1f}".format(sample_image.shape[0] / reduce_height_image.shape[0]))
    ax[0, 2].axis("off")

    ax[1, 0].imshow(mask, cmap="gray")
    ax[1, 0].set_title("Mask")
    ax[1, 0].axis("off")

    if sample_results_masked[0].shape[0] > sample_results_masked[1].shape[0]:
        reduce_width_image = sample_results_masked[0]
        reduce_height_image = sample_results_masked[1]
    else:
        reduce_width_image = sample_results_masked[1]
        reduce_height_image = sample_results_masked[0]
    
    ax[1, 1].imshow(reduce_width_image)
    ax[1, 1].set_title("widthFac: {:.1f}, heightFac: 1".format(sample_image.shape[1] / reduce_width_image.shape[1]))
    ax[1, 1].axis("off")

    ax[1, 2].imshow(reduce_height_image)
    ax[1, 2].set_title("widthFac: 1, heightFac: {:.1f}".format(sample_image.shape[0] / reduce_height_image.shape[0]))
    ax[1, 2].axis("off")

    plt.suptitle(f"Image {sample_id}")
    plt.tight_layout()
    plt.savefig("{}/comparison_{:02d}_extra.png".format(outputDir, sample_id))
