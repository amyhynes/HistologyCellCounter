import matplotlib
import numpy as np
from random import random
from skimage import exposure, io, img_as_float
from skimage.color import rgb2gray
from PIL import Image, ImageEnhance
from skimage.filters import threshold_isodata
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
from scipy import ndimage
from scipy.ndimage import find_objects
from skimage.morphology import label

#Preprocessing pipeline for iba1
def preprocess(img):
    #define original as float
    original = img_as_float(img)

    #denoise with BayesShrink
    sigma = 0.12
    noisy = random_noise(original, seed=1, var=sigma**2)
        # Estimate the average noise standard deviation across color channels.
        # Due to clipping in random_noise, the estimate will be a bit smaller than the specified sigma.
    sigma_est = estimate_sigma(noisy, multichannel=True, average_sigmas=True)
    im_bayes = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True, method='BayesShrink', 
                               mode='soft', rescale_sigma=True)
        # Compute PSNR as an indication of image quality
    psnr_bayes = peak_signal_noise_ratio(original, im_bayes)
    
    #convert to grayscale
    grayscale_bayes = rgb2gray(im_bayes)
    
    #increase contrast and brightness
    gscale = (grayscale_bayes * 255).astype(np.uint8)
    img = Image.fromarray(gscale, mode = 'L')
    brightener = ImageEnhance.Brightness(img)
    bright_bayes_pil = brightener.enhance(1.5)
    contrast = ImageEnhance.Contrast(bright_bayes_pil)
    contrast_bayes = contrast.enhance(2)
    brightener = ImageEnhance.Brightness(contrast_bayes)
    bright_bayes_pil = brightener.enhance(1.2)
    bright_bayes = np.array(bright_bayes_pil)
    
    #apply isodata threshold
    thresh_iso = threshold_isodata(bright_bayes)
    isodata = bright_bayes > thresh_iso
    
    #define new colormap
    colors = [(1,1,1)] + [(random(),random(),random()) for i in range(255)]
    new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)
    
    #label image regions
    label_image = label(isodata,connectivity=2, background=1)
    
    #extract patches
    patches = ndimage.find_objects(label_image)
    
    #convert patches to arrays
    patch_arrs = []

    for i in range(len(patches)):
        patch_arrs.append(np.interp(label_image[patches[i]],
                                    (label_image[patches[i]].min(), label_image[patches[i]].max()), 
                                    (0, 1)))
    
    return label_image, patch_arrs, patches