
### Supporting code for Computer Vision Assignment 1
### See "Assignment 1.ipynb" for instructions

import math

import numpy as np
from skimage import io

def load(img_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.
    HINT: Converting all pixel values to a range between 0.0 and 1.0
    (i.e. divide by 255) will make your life easier later on!

    Inputs:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, n_channels).
    """
    
    out = None
    # YOUR CODE HERE
    img = io.imread(img_path)
    
    return img/255

def print_stats(image):
    """ Prints the height, width and number of channels in an image.
        
    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels).
        
    Returns: none
                
    """
    
    # YOUR CODE HERE
    print(image)
    
    return None

def crop(image, x1, y1, x2, y2):
    """Crop an image based on the specified bounds. Use array slicing.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        (x1, y1): the coordinator for the top-left point
        (x2, y2): the coordinator for the bottom-right point
        

    Returns:
        out: numpy array of shape(x2 - x1, y2 - y1, 3).
    """

    out = None

    ### YOUR CODE HERE
    # arr = np.array(image)
    
    crop_arr = image[y1:y2, x1:x2]

    return crop_arr
    
def resize(input_image, fx, fy):
    """Resize an image using the nearest neighbor method.
    Not allowed to call the matural function.
    i.e. for each output pixel, use the value of the nearest input pixel after scaling

    Inputs:
        input_image: RGB image stored as an array, with shape
            `(image_height, image_width, 3)`.
        fx (float): the resize scale on the original width.
        fy (float): the resize scale on the original height.

    Returns:
        np.ndarray: Resized image, with shape `(image_height * fy, image_width * fx, 3)`.
    """
    # out = None
        
    height = int(input_image.shape[0] * fy)
    width = int(input_image.shape[1] * fx)
    # channels = int(input_image.shape[2])
    
    out_image = np.zeros((height, width, 3))
    
    for h in range(height):
        for w in range(width):
            i = int( h / fy )
            j = int( w / fx )
            
            out_image[h, w] = input_image[i, j]
            
    return out_image


def change_contrast(image, factor):
    """Change the value of every pixel by following

                        x_n = factor * (x_p - 0.5) + 0.5

    where x_n is the new value and x_p is the original value.
    Assumes pixel values between 0.0 and 1.0 
    If you are using values 0-255, divided by 255.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        factor (float): contrast adjustment

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    new_img = factor * (image - 0.5) + 0.5    

    return new_img.clip(0,1)

def greyscale(input_image):
    """Convert a RGB image to greyscale. 
    A simple method is to take the average of R, G, B at each pixel.
    Or you can look up more sophisticated methods online.
    
    Inputs:
        input_image: RGB image stored as an array, with shape
            `(image_height, image_width, 3)`.

    Returns:
        np.ndarray: Greyscale image, with shape `(image_height, image_width)`.
    """
    out = None
    
    grey_img = np.sum(input_image, axis=2) / 3
    
    return grey_img
    
def binary(grey_img, th):
    """Convert a greyscale image to a binary mask with threshold.
  
                  x_n = 0, if x_p < th
                  x_n = 1, if x_p > th
    
    Inputs:
        input_image: Greyscale image stored as an array, with shape
            `(image_height, image_width)`.
        th (float): The threshold used for binarization, and the value range is 0 to 1
    Returns:
        np.ndarray: Binary mask, with shape `(image_height, image_width)`.
    """
    out = None
    
    with np.nditer(grey_img, op_flags=['readwrite']) as it:
        for x in it:
            if x[...] < th:
                x[...] = 0
            if x[...] > th:
                x[...] = 1
    
    return grey_img


def conv2D(image, kernel):
    """ Convolution of a 2D image with a 2D kernel. 
    Convolution is applied to each pixel in the image.
    Assume values outside image bounds are 0.
    
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    kernel = np.flip(kernel)
    m, n = kernel.shape
    new_m = m // 2
    new_n = n // 2
    mn = m * n
    kernel = np.ravel(kernel)
    
    new_img = np.empty(image.shape)
    
    
    h, w = image.shape
    
    padded = np.zeros((h + m - 1, w + n - 1))
    padded[new_m:new_m+h, new_n:new_n+w] = image

    for x in range(w):
        for y in range(h):
            piece = np.reshape(padded[y:y+m, x:x+n], (mn))
            new_img[y, x] = np.dot(kernel, piece)

    return new_img


def test_conv2D():
    """ A simple test for your 2D convolution function.
        You can modify it as you like to debug your function.
    
    Returns:
        None
    """

    # Test code written by 
    # Simple convolution kernel.
    kernel = np.array(
    [
        [1,0,1],
        [0,0,0],
        [1,0,0]
    ])

    # Create a test image: a white square in the middle
    test_img = np.zeros((9, 9))
    test_img[3:6, 3:6] = 1

    # Run your conv_nested function on the test image
    test_output = conv2D(test_img, kernel)

    # Build the expected output
    expected_output = np.zeros((9, 9))
    expected_output[2:7, 2:7] = 1
    expected_output[5:, 5:] = 0
    expected_output[4, 2:5] = 2
    expected_output[2:5, 4] = 2
    expected_output[4, 4] = 3
    
    # expected_output = np.zeros((6, 6))
    # expected_output[1:6, 1:6] = 1
    # expected_output[4:, 4:] = 0
    # expected_output[3, 1:4] = 2
    # expected_output[1:4, 3] = 2
    # expected_output[3, 3] = 3

    # Test if the output matches expected output
    assert np.max(test_output - expected_output) < 1e-10, "Your solution is not correct."


def conv(image, kernel):
    """Convolution of a RGB or grayscale image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    
    out = None
    
    kernel = np.flip(kernel)
    m, n = kernel.shape
    new_m = m // 2
    new_n = n // 2
    mn = m * n
    kernel = np.ravel(kernel)
    
    new_img = np.empty(image.shape)
    
    
    h, w, channels = image.shape
    
    padded = np.zeros((h + m - 1, w + n - 1, channels))
    padded[new_m:new_m+h, new_n:new_n+w] = image

    for x in range(w):
        for y in range(h):
            piece = np.reshape(padded[y:y+m, x:x+n], (mn, channels))
            new_img[y, x] = np.dot(kernel, piece)
    
    new_img = np.clip(new_img, 0, 1)
    return new_img
    
def gauss2D(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function.
       You should not need to edit it.
       
    Args:
        size: filter height and width
        sigma: std deviation of Gaussian
        
    Returns:
        numpy array of shape (size, size) representing Gaussian filter
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def corr(image, kernel):
    """Cross correlation of a RGB image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    out = None
    ### YOUR CODE HERE
    kernel = np.flip(kernel)
    out = conv2D(image, kernel)
    
    return out