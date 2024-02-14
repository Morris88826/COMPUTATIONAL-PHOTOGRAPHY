import os
import glob
import json
import enum
import argparse
import matplotlib.pyplot as plt

class TaskOptions(enum.Enum):
    TASK1 = 0
    TASK2 = 1
    AUTO_CROP = 2
    AUTO_CONTRAST = 3
    AUTO_WHITE_BALANCE = 4
    BETTER_FEATURE = 5
    BETTER_TRANSFORMATION = 6

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RGB Image Alignment')
    parser.add_argument('--task', type=int, default=0, help='Task number')
    parser.add_argument('--input_dir', default='../Images', help='Input image directory')
    parser.add_argument('--result_dir', default='../Results', help='Output image directory')
    parser.add_argument('--out_dir', default='../demo', help='Output demo image directory')

    args = parser.parse_args()

    taskOption = TaskOptions(args.task)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    dpi = 250
    
    if taskOption == TaskOptions.TASK1:
        images_path = [os.path.join(args.result_dir, 'default' , os.path.basename(path)) for path in glob.glob(os.path.join(args.input_dir, '*.jpg'))]
        fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=dpi)
        
        with open(os.path.join(args.result_dir, 'default', 'metadata.json')) as f:
            metadata = json.load(f)

        for i, path in enumerate(images_path):
            basename = os.path.basename(path)
            _metadata = metadata[basename]
            rShift = _metadata['rShift']
            gShift = _metadata['gShift']

            axs[i//2, i%2].imshow(plt.imread(path))
            axs[i//2, i%2].set_title('{}: dR = [{},{}], dG = [{},{}]'.format(basename, rShift[0], rShift[1], gShift[0], gShift[1]))
            axs[i//2, i%2].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(os.path.join(args.out_dir, '{}.png'.format(taskOption.name.lower())))
        plt.close()

    elif taskOption == TaskOptions.TASK2:
        images_path = [os.path.join(args.result_dir, 'default' , os.path.basename(path).replace('.tif', '.jpg')) for path in glob.glob(os.path.join(args.input_dir, '*.tif'))]

        
        
        with open(os.path.join(args.result_dir, 'default', 'metadata.json')) as f:
            metadata = json.load(f)

        # there are 9 images in the result directory, make it into batches of 4
        batched_images_path = [images_path[i:i+4] for i in range(0, len(images_path), 4)]
        
        for i, batch in enumerate(batched_images_path):

            if i == 2:
                plt.figure(figsize=(5, 5), dpi=dpi)
                for j, path in enumerate(batch):
                    basename = os.path.basename(path).replace('.jpg', '.tif')
                    _metadata = metadata[basename]
                    rShift = _metadata['rShift']
                    gShift = _metadata['gShift']

                    plt.imshow(plt.imread(path))
                    plt.axis('off')
                    plt.title('{}: dR = [{},{}], dG = [{},{}]'.format(basename, rShift[0], rShift[1], gShift[0], gShift[1]))
            else:
                fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=dpi)
                for j, path in enumerate(batch):
                    basename = os.path.basename(path).replace('.jpg', '.tif')
                    _metadata = metadata[basename]
                    rShift = _metadata['rShift']
                    gShift = _metadata['gShift']

                    axs[j//2, j%2].imshow(plt.imread(path))
                    axs[j//2, j%2].set_title('{}: dR = [{},{}], dG = [{},{}]'.format(basename, rShift[0], rShift[1], gShift[0], gShift[1]))
                    axs[j//2, j%2].axis('off')
            


            plt.tight_layout()
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.savefig(os.path.join(args.out_dir, '{}_{}.png'.format(taskOption.name.lower(), i+1)))
            plt.close()

    elif taskOption == TaskOptions.AUTO_CROP:
        example_images = ["emir.tif", "village.tif"] 

        fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=dpi)
        for row, example_image in enumerate(example_images):

            without_crop_path = os.path.join(args.result_dir, 'default', example_image.replace('.tif', '.jpg'))
            with_crop_path = os.path.join(args.result_dir, 'auto_crop', example_image.replace('.tif', '.jpg'))

            
            axs[row, 0].imshow(plt.imread(without_crop_path))
            axs[row, 0].set_title('{}: W/O Auto Crop'.format(example_image))

            axs[row, 1].imshow(plt.imread(with_crop_path))
            axs[row, 1].set_title('{}: W/ Auto Crop'.format(example_image))

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(os.path.join(args.out_dir, '{}.png'.format(taskOption.name.lower())))
        plt.close()
    
    elif taskOption == TaskOptions.AUTO_CONTRAST:
        example_images = ["monastery.jpg", "lady.tif"] 

        fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=dpi)
        for row, example_image in enumerate(example_images):

            without_contrast_path = os.path.join(args.result_dir, 'default', example_image.replace('.tif', '.jpg'))
            with_contrast_path = os.path.join(args.result_dir, 'auto_contrast', example_image.replace('.tif', '.jpg'))

            
            axs[row, 0].imshow(plt.imread(without_contrast_path))
            axs[row, 0].set_title('{}: W/O Auto Contrast'.format(example_image))
            axs[row, 0].axis('off')

            axs[row, 1].imshow(plt.imread(with_contrast_path))
            axs[row, 1].set_title('{}: W/ Auto Contrast'.format(example_image))
            axs[row, 1].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(os.path.join(args.out_dir, '{}.png'.format(taskOption.name.lower())))
        plt.close()

    elif taskOption == TaskOptions.AUTO_WHITE_BALANCE:
        example_images = ["cathedral.jpg", "icon.tif"] 

        fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=dpi)
        for row, example_image in enumerate(example_images):

            without_white_balance_path = os.path.join(args.result_dir, 'default', example_image.replace('.tif', '.jpg'))
            with_white_balance_path = os.path.join(args.result_dir, 'auto_white_balance', example_image.replace('.tif', '.jpg'))

            axs[row, 0].imshow(plt.imread(without_white_balance_path))
            axs[row, 0].set_title('{}: W/O Auto White Balance'.format(example_image))
            axs[row, 0].axis('off')

            axs[row, 1].imshow(plt.imread(with_white_balance_path))
            axs[row, 1].set_title('{}: W/ Auto White Balance'.format(example_image))
            axs[row, 1].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(os.path.join(args.out_dir, '{}.png'.format(taskOption.name.lower())))
        plt.close()

    elif taskOption == TaskOptions.BETTER_FEATURE:
        example_image = "emir.tif"

        without_feature_path = os.path.join(args.result_dir, 'default', example_image.replace('.tif', '.jpg'))
        with_feature_path = os.path.join(args.result_dir, 'better_features', example_image.replace('.tif', '.jpg'))

        fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi)
        axs[0].imshow(plt.imread(without_feature_path))
        axs[0].set_title('W/O Better Features')
        axs[0].axis('off')

        axs[1].imshow(plt.imread(with_feature_path))
        axs[1].set_title('W/ Better Features')
        axs[1].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(os.path.join(args.out_dir, '{}.png'.format(taskOption.name.lower())))
        plt.close()