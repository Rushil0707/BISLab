
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Objective function: Otsu's Thresholding for Image Segmentation
def otsu_variance(threshold, histogram, total_pixels):
    background_weight = np.sum(histogram[:threshold])
    foreground_weight = np.sum(histogram[threshold:])
    if background_weight == 0 or foreground_weight == 0:
        return float('inf')  # Avoid division by zero
    background_mean = np.sum(np.arange(threshold) * histogram[:threshold]) / background_weight
    foreground_mean = np.sum(np.arange(threshold, 256) * histogram[threshold:]) / foreground_weight
    between_class_variance = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
    return -between_class_variance  # Minimize negative of variance

# Grey Wolf Optimizer
def grey_wolf_optimizer(histogram, total_pixels, max_iter=50, population_size=10):
    dim = 1  # Only optimizing threshold
    alpha_pos, beta_pos, delta_pos = None, None, None
    alpha_score, beta_score, delta_score = float('inf'), float('inf'), float('inf')
    wolves = np.random.randint(0, 256, (population_size, dim))

    a = 2  # Control parameter
    for iteration in range(max_iter):
        for i in range(population_size):
            fitness = otsu_variance(wolves[i][0], histogram, total_pixels)
            if fitness < alpha_score:
                alpha_score, beta_score, delta_score = fitness, alpha_score, beta_score
                alpha_pos, beta_pos, delta_pos = wolves[i], alpha_pos, beta_pos
            elif fitness < beta_score:
                beta_score, delta_score = fitness, beta_score
                beta_pos, delta_pos = wolves[i], beta_pos
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = wolves[i]

        # Update positions
        for i in range(population_size):
            for d in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = abs(C1 * alpha_pos[d] - wolves[i][d])
                X1 = alpha_pos[d] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = abs(C2 * beta_pos[d] - wolves[i][d])
                X2 = beta_pos[d] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = abs(C3 * delta_pos[d] - wolves[i][d])
                X3 = delta_pos[d] - A3 * D_delta

                wolves[i][d] = np.clip((X1 + X2 + X3) / 3, 0, 255)

        a -= 2 / max_iter  # Linearly decrease a

    return int(alpha_pos[0])  # Return optimal threshold

# Main function
if __name__ == "__main__":
    # Load and preprocess image
    img = cv2.imread("/content/design_resolution_original.jpg", 0)  # Grayscale image
    histogram, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    total_pixels = img.size

    # Run GWO
    optimal_threshold = grey_wolf_optimizer(histogram, total_pixels)
    print("Optimal Threshold:", optimal_threshold)

    # Apply threshold
    _, segmented_img = cv2.threshold(img, optimal_threshold, 255, cv2.THRESH_BINARY)

    # Display results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(segmented_img, cmap="gray")
    plt.show()

