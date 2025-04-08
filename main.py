import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import os
from PIL import Image


"""  
This code has been created as a semestral work for subject of Sensoric systems on Czech Technical University, faculty of mechanical engineering.
The code is mainly for demonstrational purposes, isnt really optimalized and definitiely can be done in a cleaner way
For best usage it is recommended to use lower resolution pictures
All used pictures are credited to "The Middlebury Computer Vision Pages" with images to be found at: https://vision.middlebury.edu/stereo/data/scenes2005/FullSize/Laundry/
For those whom this might concern, excuse the code being in english, author is used to do it that way
For contact: Git - jrycho
"""


class StereoDepth():
    """  
    img_left, img_right - np arrays from pictures
    search_block_size:int - size of comparison matrix between pictures
    """
    #loading attributes for good acces
    def __init__(self, img_left, img_right, search_block_size:int):
        self.img_left = img_left
        self.img_right = img_right
        self.search_block_size = search_block_size
        self.search_block_size_floored = search_block_size //2
        self.max_disparity = 16
        self.x_size = np.array(self.img_left.shape[1])
        self.y_size = np.array(self.img_left.shape[0])
        self.disparity_map = np.zeros_like(img_left, dtype=np.float32)
        
    """
    Args:
    left_matrix - matrix from left image
    right_matrix  - matrix from rigt image
    """
    #Function to count quality of matrix selected from right camera with currently compared matrix from right camera using SSD (sum of squares difference) 
    def SSD_counter(self, left_matrix, right_matrix):
        match = np.sum((left_matrix - right_matrix)**2)
        return match

    """  method to find solution
    For every horizontal line
    For every pixel in horizontal line
    selects matrix in the size of search_block (search matrix (can be imagined as convolution principe))
    matrix is sent to right image runner that finds centerpoint of matrix with the least SSD (sum of squares difference)
    returns the best centerpoint of matrix
    calculates the disparity of the pixel
    saves the disparity in the disparity map
    """
    def solve(self):
        for y_left in range(self.search_block_size_floored, self.y_size - self.search_block_size_floored): 
            for x_left in range(self.search_block_size_floored, self.x_size - self.search_block_size_floored):
                
                left_matrix = self.img_left[
                    (y_left - self.search_block_size_floored):(y_left + self.search_block_size_floored),
                      x_left - self.search_block_size_floored:x_left + self.search_block_size_floored]
                
                best_right_position = self.right_image_runner(x_left, y_left, left_matrix)
                disparity = x_left - best_right_position
                self.disparity_map[y_left, x_left] = disparity
    	

    """  
    Args:
    x_left - x coordinate of left image matrix centerpoint
    y_left - y coordinate of left image matrix centerpoint
    left_matrix = matrix to be found in right image
    """
    """
    method of comparisong of left image matrix with right image matrix'es
    compares only points that are within self.max_disparity range to the left, as thats where the object should be found
    (you can test that by looking at an object with left eye and then right eye)
    max_disparity is the estimation of range where obeject can be found, helps with computation speed
    returns the best x value of the right image matrix
    """    
    def right_image_runner(self, x_left, y_left, left_matrix):
        best_match = float('inf')
        best_x = x_left  # fallback in case no match is found

        for d in range(self.max_disparity):
            x_right = x_left - d
            if x_right - self.search_block_size_floored < 0:
                continue

            right_matrix = self.img_right[
                y_left - self.search_block_size_floored : y_left + self.search_block_size_floored,
                x_right - self.search_block_size_floored : x_right + self.search_block_size_floored
            ]

            if right_matrix.shape != left_matrix.shape:
                continue  # skip if block is incomplete (e.g., near edges)

            match = self.SSD_counter(left_matrix, right_matrix)

            if match < best_match:
                best_match = match
                best_x = x_right

        return best_x


            


    """  
    Just creates heatmap of object
    """

    def heatmap_show(self):
        if not self.disparity_map.any():
            self.solve()
        else:
            pass
        plt.imshow(self.disparity_map, cmap='plasma')
        plt.colorbar(label='Disparity')
        plt.title("Stereo Matching Result (Manual SSD)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    """  
    Args:
    ax - list of axis on the plot
    title - title of 
    """
    """  
    method for projecting heatmaps to an axis
    """
    def plot_on_ax(self, ax, title="Stereo Matching Result"):
        if not self.disparity_map.any():
            self.solve()

        im = ax.imshow(self.disparity_to_distance(self.disparity_map, 3740/4, 0.16), cmap='turbo')
        ax.set_title(title)
        ax.axis('off')
        return im 


    """  
    Args:
    disparity_map - disparity map
    focal_length_px - focal length of camera in pixels
    baseline_m - baseline of camera in meters
    """
    """  
    method for recalculation of distance from disparity map
    """
    def disparity_to_distance(self, disparity_map, focal_length_px, baseline_m):
        # Avoid division by zero
        disparity_copy = disparity_map.copy()
        mask = disparity_copy <= 0
        disparity_copy[mask] = 0.0001  # or mask later
        depth_map = (focal_length_px * baseline_m) / disparity_copy
        return ma.masked_array(depth_map, mask)


"""  
import images in numpy array, grayscale
"""
img_left = np.array(Image.open('075view0.png').convert('L')).astype(np.uint8)
img_right = np.array(Image.open('075view1.png').convert('L')).astype(np.uint8)
"""  
objects creation
"""
Test1 = StereoDepth(img_left, img_right, 3)
Test2 = StereoDepth(img_left, img_right, 5)
Test3 = StereoDepth(img_left, img_right, 7)
Test4 = StereoDepth(img_left, img_right, 9)

"""  
figure definition
"""
fig, axs = plt.subplots(2, 2, figsize=(12, 10), 
                        constrained_layout=True, 
                        gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})

# Plot all four StereoDepth outputs
im1 = Test1.plot_on_ax(axs[0, 0], "Image Pair 1")
im2 = Test2.plot_on_ax(axs[0, 1], "Image Pair 2")
im3 = Test3.plot_on_ax(axs[1, 0], "Image Pair 3")
im4 = Test4.plot_on_ax(axs[1, 1], "Image Pair 4")

# Ensure all images use the same color scale
vmin = min(im1.get_array().min(), im2.get_array().min(), im3.get_array().min(), im4.get_array().min())
vmax = max(im1.get_array().max(), im2.get_array().max(), im3.get_array().max(), im4.get_array().max())
for im in [im1, im2, im3, im4]:
    im.set_clim(vmin, vmax)

# Add a single shared colorbar
cbar = fig.colorbar(im1, ax=axs, orientation='vertical', shrink=0.85, label='Distance')

plt.suptitle("Stereo Disparity Maps", fontsize=16)
plt.show()

