import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt
from collections import deque

# 1. Compute local minima as markers
def get_local_minima(gray):
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(gray, kernel)
    minima = (gray == eroded)
    return minima.astype(np.uint8)

# 2. Label each connected component (marker)
def label_markers(minima):
    num_labels, markers = cv2.connectedComponents(minima)
    return markers, num_labels

# 3. Watershed from scratch
def watershed_from_scratch(gray, markers):
    h, w = gray.shape
    # Constants
    WATERSHED = -1
    INIT = -2

    # Initialize label and visited map
    label_map = np.full((h, w), INIT, dtype=np.int32)
    label_map[markers > 0] = markers[markers > 0]

    # Priority queue for pixels: (intensity, y, x)
    pq = []

    # Populate queue with boundary of initial markers
    for y in range(h):
        for x in range(w):
            if markers[y, x] > 0:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if markers[ny, nx] == 0 and label_map[ny, nx] == INIT:
                                heapq.heappush(pq, (gray[ny, nx], ny, nx))
                                label_map[ny, nx] = 0  # Mark as in queue

    # Flooding
    while pq:
        intensity, y, x = heapq.heappop(pq)

        neighbor_labels = set()
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    lbl = label_map[ny, nx]
                    if lbl > 0:
                        neighbor_labels.add(lbl)

        if len(neighbor_labels) == 1:
            label_map[y, x] = neighbor_labels.pop()
        elif len(neighbor_labels) > 1:
            label_map[y, x] = WATERSHED

        # Add unvisited neighbors to the queue
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if label_map[ny, nx] == INIT:
                        heapq.heappush(pq, (gray[ny, nx], ny, nx))
                        label_map[ny, nx] = 0  # Mark as in queue

    return label_map

def generate_watershed(iamge_path):
    # Load grayscale image
    image = cv2.imread(iamge_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    minima = get_local_minima(image)
    markers, num_labels = label_markers(minima)
    result = watershed_from_scratch(image, markers)

    # Visualization
    output = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    output[result == -1] = [255, 0, 0]  # Watershed lines in red
    output[result > 0] = [0, 255, 0]    # Segments in green
    output[markers > 0] = [0, 0, 255]   # Original minima in blue
    return image,output
if __name__ == "__main__":
    # Run the process
    # Load grayscale image
    image = cv2.imread("/home/akshat/projects/CSL7360_Project/bird.jpeg", cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    minima = get_local_minima(image)
    markers, num_labels = label_markers(minima)
    result = watershed_from_scratch(image, markers)

    # Visualization
    output = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    output[result == -1] = [255, 0, 0]  # Watershed lines in red
    output[result > 0] = [0, 255, 0]    # Segments in green
    output[markers > 0] = [0, 0, 255]   # Original minima in blue


    # Save the original grayscale and the output image
    cv2.imwrite("original_grayscale.png", image)
    cv2.imwrite("watershed_output.png", output)

    print("Images saved as 'original_grayscale.png' and 'watershed_output.png'")



