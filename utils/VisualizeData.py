import numpy as np
import matplotlib.pyplot as plt
import sys
import os
def visualize_training_samples(original_item,
                               original_class,
                               augmented_item,
                               augmented_class,
                               fold_path,
                               classes_names,
                               global_baseline_transformations):
    if type(global_baseline_transformations) is tuple:
        corresponding_transformation = global_baseline_transformations[original_class]
        global_transformation = str(corresponding_transformation).split('(')[0]
    else:
        global_transformation = str(global_baseline_transformations[0]).split('(')[0]

    if original_class != augmented_class:
        sys.exit('Original class does not match augmented class')

    training_img_sample = original_item
    training_img_sample = training_img_sample.permute(1, 2, 0).numpy()

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    unnormalized_image = (training_img_sample * std) + mean
    unnormalized_image = np.clip(unnormalized_image, 0.0, 1.0)


    augmented_img_sample = augmented_item
    augmented_img_sample = augmented_img_sample.permute(1, 2, 0).numpy()
    unnormalized_augmented_image = (augmented_img_sample * std) + mean
    unnormalized_augmented_image = np.clip(unnormalized_augmented_image, 0.0, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(unnormalized_image)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')  # Hide axes

    # Plot the second image
    axes[0, 1].imshow(training_img_sample)
    axes[0, 1].set_title('Original processed')
    axes[0, 1].axis('off')  # Hide axes

    # Plot the third image
    axes[1, 0].imshow(unnormalized_augmented_image)
    axes[1, 0].set_title(f'Augmented {global_transformation}')
    axes[1, 0].axis('off')  # Hide axes

    # Plot the fourth image
    axes[1, 1].imshow(augmented_img_sample)
    axes[1, 1].set_title('Augmented processed')
    axes[1, 1].axis('off')  # Hide axes
    plt.savefig(os.path.join(fold_path, f'{classes_names[original_class]}.png'))
    plt.clf()
    plt.close()