import os
import enum
import argparse
import numpy as np
import matplotlib.pyplot as plt
from hybrid_images import align_images, create_hybrid_image
from pyramid_blending import create_pyramid_blend, Read
from filters import scale_laplacian_image

class TaskOptions(enum.Enum):
    TASK1 = 0
    EXTRA = 1
    TASK2 = 2
    TASK1_ABLATION = 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RGB Image Alignment')
    parser.add_argument('--task', type=int, default=0, help='Task number')
    parser.add_argument('--input_dir', default='../Images', help='Input image directory')
    parser.add_argument('--result_dir', default='../Results', help='Output image directory')

    args = parser.parse_args()

    input_dir = args.input_dir
    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


    taskOption = TaskOptions(args.task)

    if taskOption == TaskOptions.TASK1:
        out_dir = os.path.join(result_dir, "Task1")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        image_pairs = [
            ["Monroe.jpg", "Einstein.jpg"], 
            ["1a_dog.png", "1b_cat.png"], 
            ["2a_plane.png", "2b_bird.png"]]

        matching_pairs = [
                [[466.7762445887447, 388.84767316017303], [636.5070346320348, 379.9921536796536], [161.62121212121215, 192.03246753246748], [237.9199134199135, 194.1969696969697]],
                [[118.05925324675326, 186.44642857142853], [280.68506493506493, 200.60903679653674],[118.05925324675326, 186.44642857142853], [280.68506493506493, 200.60903679653674]],
                [[157.5, 157.5], [180, 180], [157.5, 157.5], [180, 180]]
            ]
        
        sigmas = [[4,4],[3,7],[4,4]]

        params = []
        for i in range(len(image_pairs)):
            # rule of thumb for kernel size: 6*sigma - 1
            kernel_size_1 = 6*sigmas[i][0] - 1 if 6*sigmas[i][0] % 2 == 0 else 6*sigmas[i][0]
            kernel_size_2 = 6*sigmas[i][1] - 1 if 6*sigmas[i][1] % 2 == 0 else 6*sigmas[i][1]
            sigma_1 = sigmas[i][0]
            sigma_2 = sigmas[i][1]
            params.append([kernel_size_1, sigma_1, kernel_size_2, sigma_2])


        num_images = len(image_pairs)
        for i in range(num_images):
            im1_name = image_pairs[i][0]
            im2_name = image_pairs[i][1]

            if not os.path.exists(os.path.join(args.input_dir, image_pairs[i][0])) or not os.path.exists(os.path.join(args.input_dir, image_pairs[i][1])):
                print("Image pair: {} and {} not found".format(image_pairs[i][0], image_pairs[i][1]))
                continue 

            im1 = plt.imread(os.path.join(args.input_dir, image_pairs[i][0]))
            im2 = plt.imread(os.path.join(args.input_dir, image_pairs[i][1]))
            im1_aligned, im2_aligned = align_images(im1, im2, matching_pairs[i])
            
            kernel_size_1 = params[i][0]
            sigma_1 = params[i][1]
            kernel_size_2 = params[i][2]
            sigma_2 = params[i][3]

            im = create_hybrid_image(im1_aligned, im2_aligned, kernel_size_1, sigma_1, kernel_size_2, sigma_2, isGray=True)

            plt.imsave(out_dir +"/"+ im1_name.split('.')[0] + '.jpg', im1.mean(axis=2), cmap='gray')
            plt.imsave(out_dir +"/"+ im2_name.split('.')[0] + '.jpg', im2.mean(axis=2), cmap='gray')
            plt.imsave(out_dir +"/"+ im1_name.split('.')[0] + '_' + im2_name.split('.')[0] + '_Hybrid.jpg', im, cmap='gray')


    elif taskOption == TaskOptions.EXTRA:
        out_dir = os.path.join(result_dir, "Extra")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        image_pair = ["1a_dog.png", "1b_cat.png"]
        matching_pairs = [[118.05925324675326, 186.44642857142853], [280.68506493506493, 200.60903679653674],[118.05925324675326, 186.44642857142853], [280.68506493506493, 200.60903679653674]]
        sigmas = [3,7]

        kernel_size_1 = 6*sigmas[0] - 1 if 6*sigmas[0] % 2 == 0 else 6*sigmas[0]
        kernel_size_2 = 6*sigmas[1] - 1 if 6*sigmas[1] % 2 == 0 else 6*sigmas[1]
        sigma_1 = sigmas[0]
        sigma_2 = sigmas[1]

        if not os.path.exists(os.path.join(args.input_dir, image_pair[0])) or not os.path.exists(os.path.join(args.input_dir, image_pair[1])):
            print("Image pair: {} and {} not found".format(image_pair[0], image_pair[1]))
            exit(0) 

        im1 = plt.imread(os.path.join(args.input_dir, image_pair[0]))
        im2 = plt.imread(os.path.join(args.input_dir, image_pair[1]))

        im1_aligned, im2_aligned = align_images(im1, im2, matching_pairs)

        gray_im1_aligned = im1_aligned.mean(axis=2)
        gray_im2_aligned = im2_aligned.mean(axis=2)

        def plot_result(im1, im2, sigma_1, sigma_2, kernel_size_1, kernel_size_2):
            is_im1_gray = im1.ndim == 2
            is_im2_gray = im2.ndim == 2
            
            if is_im1_gray:
                im1 = np.stack((im1,)*3, axis=-1)
            if is_im2_gray:
                im2 = np.stack((im2,)*3, axis=-1)

            hybrid_image, (im_low, im_high) = create_hybrid_image(im1, im2, kernel_size_1, sigma_1, kernel_size_2, sigma_2, isGray=False, verbose=True)


            im1_title = 'Gray' if is_im1_gray else 'RGB'
            im2_title = 'Gray' if is_im2_gray else 'RGB'

            print("Creating hybrid image for {} im1, {} im2: sigma_1 = {}, sigma_2 = {}".format(im1_title, im2_title, sigma_1, sigma_2))

            fig, ax = plt.subplots(2, 3, figsize=(10, 6))

            if is_im1_gray:
                ax[0,0].imshow(im1, cmap='gray')
            else:
                ax[0,0].imshow(im1)

            ax[0,0].set_title('{} im1'.format(im1_title))
            ax[0,0].axis('off')

            if is_im2_gray:
                ax[0,1].imshow(im2, cmap='gray')
            else:
                ax[0,1].imshow(im2)
            ax[0,1].set_title('{} im2'.format(im2_title))
            ax[0,1].axis('off')

            # remove ax[0,2] for now
            ax[0,2].axis('off')

            if is_im1_gray:
                ax[1,0].imshow(im_low, cmap='gray')
            else:
                ax[1,0].imshow(im_low)
            ax[1,0].set_title('Low-Pass filtered'.format(im1_title))
            ax[1,0].axis('off')

            if is_im2_gray:
                ax[1,1].imshow(scale_laplacian_image(im_high), cmap='gray')
            else:
                ax[1,1].imshow(scale_laplacian_image(im_high))
            ax[1,1].set_title('High-Pass filtered'.format(im2_title))
            ax[1,1].axis('off')

            if is_im1_gray and is_im2_gray:
                ax[1,2].imshow(hybrid_image, cmap='gray')
            else:
                ax[1,2].imshow(hybrid_image)
            ax[1,2].set_title('Hybrid Image'.format(im1_title, im2_title))
            ax[1,2].axis('off')

            plt.imsave(out_dir +"/"+ image_pair[0].split('.')[0] + '_' + image_pair[1].split('.')[0] + '_Hybrid_{}_{}.jpg'.format(im1_title.lower(), im2_title.lower()), hybrid_image)
            
            plt.suptitle('Hybrid Image: im1 (sigma: {}, kernel size:{}), im2 (sigma: {}, kernel size:{})'.format(sigma_1, kernel_size_1, sigma_2, kernel_size_2))
            plt.tight_layout()
            fig.savefig(out_dir +"/"+ image_pair[0].split('.')[0] + '_' + image_pair[1].split('.')[0] + '_Hybrid_{}_{}_cp.jpg'.format(im1_title.lower(), im2_title.lower()))
            plt.close()

            print("=====================================")
        
        plot_result(im1_aligned, im2_aligned, sigma_1, sigma_2, kernel_size_1, kernel_size_2)
        plot_result(gray_im1_aligned, im2_aligned, sigma_1, sigma_2, kernel_size_1, kernel_size_2)
        plot_result(im1_aligned, gray_im2_aligned, sigma_1, sigma_2, kernel_size_1, kernel_size_2)
        plot_result(gray_im1_aligned, gray_im2_aligned, sigma_1, sigma_2, kernel_size_1, kernel_size_2)


    elif taskOption == TaskOptions.TASK2:
        out_dir = os.path.join(result_dir, "Task2") 
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # main area to specify files and display blended image
        indices = [1, 2, 3]
        levels = [6, 6, 8]
        # Read data and clean mask
        for (index, level) in zip(indices, levels):
            
            if not os.path.exists(os.path.join(args.input_dir, "source_{:02d}.jpg".format(index))) or not os.path.exists(os.path.join(args.input_dir, "target_{:02d}.jpg".format(index))) or not os.path.exists(os.path.join(args.input_dir, "mask_{:02d}.jpg".format(index))):
                print("Image pair: source_{:02d}.jpg, target_{:02d}.jpg or mask_{:02d} not found".format(index, index, index))
                continue

            source, maskOriginal, target = Read(str(index).zfill(2), input_dir+'/')
            pyramidOutput = create_pyramid_blend(source, maskOriginal, target, level)
            naiveOutput = create_pyramid_blend(source, maskOriginal, target, 1)
            plt.imsave("{}/pyramid_{}.jpg".format(out_dir, str(index).zfill(2)), pyramidOutput)

            fig, ax = plt.subplots(1, 5, figsize=(24, 8))
            ax[0].imshow(source)
            ax[0].set_title('Source')
            ax[0].axis('off')

            ax[1].imshow(maskOriginal, cmap='gray')
            ax[1].set_title('Mask')
            ax[1].axis('off')

            ax[2].imshow(target)
            ax[2].set_title('Target')
            ax[2].axis('off')

            ax[3].imshow(naiveOutput)
            ax[3].set_title('Naive Blended (no pyramid)')
            ax[3].axis('off')

            ax[4].imshow(pyramidOutput)
            ax[4].set_title('Pyramid Blending')
            ax[4].axis('off')

            plt.tight_layout()
            plt.suptitle('Pyramid Blending: Image {}, num_levels: {}'.format(str(index).zfill(2), level))
            fig.savefig("{}/pyramid_{}_cp.jpg".format(out_dir, str(index).zfill(2)))
            plt.close()



    elif taskOption == TaskOptions.TASK1_ABLATION:
        out_dir = os.path.join(result_dir, "Task1_Ablation")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        image_pair = ["1a_dog.png", "1b_cat.png"]


        matching_pair = [[118.05925324675326, 186.44642857142853], [280.68506493506493, 200.60903679653674],[118.05925324675326, 186.44642857142853], [280.68506493506493, 200.60903679653674]]
        
        if not os.path.exists(os.path.join(args.input_dir, image_pair[0])) or not os.path.exists(os.path.join(args.input_dir, image_pair[1])):
            print("Image pair: {} and {} not found".format(image_pair[0], image_pair[1]))
            exit(0) 
        
        sigmas = [[3,3],[3,9],[3,5],[9,5]]
        params = []
        for i in range(len(sigmas)):
            kernel_size_1 = 6*sigmas[i][0] - 1 if 6*sigmas[i][0] % 2 == 0 else 6*sigmas[i][0]
            kernel_size_2 = 6*sigmas[i][1] - 1 if 6*sigmas[i][1] % 2 == 0 else 6*sigmas[i][1]
            sigma_1 = sigmas[i][0]
            sigma_2 = sigmas[i][1]
            params.append([kernel_size_1, sigma_1, kernel_size_2, sigma_2])
    
        for i in range(len(sigmas)):
            im1_name = image_pair[0]
            im2_name = image_pair[1]
            im1 = plt.imread(os.path.join(args.input_dir, im1_name))
            im2 = plt.imread(os.path.join(args.input_dir, im2_name))
            im1_aligned, im2_aligned = align_images(im1, im2, matching_pair)
            
            kernel_size_1 = params[i][0]
            sigma_1 = params[i][1]
            kernel_size_2 = params[i][2]
            sigma_2 = params[i][3]

            im = create_hybrid_image(im1_aligned, im2_aligned, kernel_size_1, sigma_1, kernel_size_2, sigma_2, isGray=True)

            plt.imsave(out_dir +"/"+ im1_name.split('.')[0] + '_' + im2_name.split('.')[0] + '_Hybrid_{}_{}.jpg'.format(sigma_1, sigma_2), im, cmap='gray')

