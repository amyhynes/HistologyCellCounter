import sys
import numpy as np
import pandas as pd
from skimage.measure import regionprops
from skimage.feature import hog
from sklearn.preprocessing import MinMaxScaler

#Initialize MR8 filter bank

#https://gist.github.com/amueller/3129692
##########################################################################
# Maximum Response filterbank from
# http://www.robots.ox.ac.uk/~vgg/research/texclass/filters.html
# based on several edge and bar filters.
# Adapted to Python by Andreas Mueller amueller@ais.uni-bonn.de
# Share and enjoy
#

import cv2
import pickle

from itertools import product, chain

def makeRFSfilters(radius=28, sigmas=[1, 2, 4], n_orientations=6):
    """ Generates filters for RFS filterbank.
    Parameters
    ----------
    radius : int, default 28
        radius of all filters. Size will be 2 * radius + 1
    sigmas : list of floats, default [1, 2, 4]
        define scales on which the filters will be computed
    n_orientations : int
        number of fractions the half-angle will be divided in
    Returns
    -------
    edge : ndarray (len(sigmas), n_orientations, 2*radius+1, 2*radius+1)
        Contains edge filters on different scales and orientations
    bar : ndarray (len(sigmas), n_orientations, 2*radius+1, 2*radius+1)
        Contains bar filters on different scales and orientations
    rot : ndarray (2, 2*radius+1, 2*radius+1)
        contains two rotation invariant filters, Gaussian and Laplacian of
        Gaussian
    """
    def make_gaussian_filter(x, sigma, order=0):
        if order > 2:
            raise ValueError("Only orders up to 2 are supported")
        # compute unnormalized Gaussian response
        response = np.exp(-x ** 2 / (2. * sigma ** 2))
        if order == 1:
            response = -response * x
        elif order == 2:
            response = response * (x ** 2 - sigma ** 2)
        # normalize
        response /= np.abs(response).sum()
        return response

    def makefilter(scale, phasey, pts, sup):
        gx = make_gaussian_filter(pts[0, :], sigma=3 * scale)
        gy = make_gaussian_filter(pts[1, :], sigma=scale, order=phasey)
        temp = gx*gy
        temp = np.reshape(temp,(gx.shape[0],1))
        #f = (gx * gy).reshape(sup, sup)
        f = np.reshape(temp,(int(sup), int(sup)))
        # normalize
        f /= np.abs(f).sum()
        return f

    support = 2 * radius + 1
    x, y = np.mgrid[-radius:radius + 1, radius:-radius - 1:-1]
    orgpts = np.vstack([x.ravel(), y.ravel()])

    rot, edge, bar = [], [], []
    for sigma in sigmas:
        for orient in range(n_orientations):
            # Not 2pi as filters have symmetry
            angle = np.pi * orient / n_orientations
            c, s = np.cos(angle), np.sin(angle)
            rotpts = np.dot(np.array([[c, -s], [s, c]]), orgpts)
            edge.append(makefilter(sigma, 1, rotpts, support))
            bar.append(makefilter(sigma, 2, rotpts, support))

    length = np.sqrt(x ** 2 + y ** 2)
    rot.append(make_gaussian_filter(length, sigma=10))
    rot.append(make_gaussian_filter(length, sigma=10, order=2))

    # reshape rot and edge
    edge = np.asarray(edge)
    edge = edge.reshape(len(sigmas), n_orientations, int(support), int(support))
    bar = np.asarray(bar).reshape(edge.shape)
    rot = np.asarray(rot)[:, np.newaxis, :, :]
    return edge, bar, rot


def apply_filterbank(img, filterbank):
    from scipy.ndimage import convolve
    result = []
    for battery in filterbank:
        for scale in battery:
            response = []
            for filt in scale:
                response.append(convolve(img, filt))
            max_response = np.max(response, axis=0)
            result.append(max_response)
    return result


if __name__ == "__main__":

    sigmas = [1, 2, 4]
    n_sigmas = len(sigmas)
    n_orientations = 6

    edge, bar, rot = makeRFSfilters(sigmas=sigmas,
            n_orientations=n_orientations)

    n = n_sigmas * n_orientations

#Extract shape, texture, and HoG features

def extract_features(label_image, patch_arr):
    #Shape features
    df = pd.DataFrame(columns=['solidity', 'orientation', 'diameter', 'area', 'eccentricity', 'convex area', 'major axis length', 
                 'minor axis length', 'extent'])
    regions = regionprops(label_image)
    for region in regions:
        region_dict = {'solidity': region.solidity,
                       'orientation': region.orientation,
                       'diameter': region.equivalent_diameter,
                       'area': region.area,
                       'eccentricity': region.eccentricity,
                       'convex area': region.convex_area,
                       'major axis length': region.major_axis_length,
                       'minor axis length': region.minor_axis_length,
                       'extent': region.extent}
        df=df.append(region_dict, ignore_index=True)
    
    #Texture features
    filterbank = makeRFSfilters()
    top8_arr = []
    for patch in patch_arr:
        top8_arr.append(apply_filterbank(patch, filterbank))
    df_text = pd.DataFrame(columns=['edge1', 'edge2', 'edge3', 'bar1', 'bar2', 'bar3', 'gauss', 'lap'])
    for top8 in top8_arr:
        top8_dict = {'edge1': np.max(top8[0]),
                     'edge2': np.max(top8[1]),
                     'edge3': np.max(top8[2]),
                     'bar1': np.max(top8[3]),
                     'bar2': np.max(top8[4]),
                     'bar3': np.max(top8[5]),
                     'gauss': np.max(top8[6]),
                     'lap': np.max(top8[7])}
        df_text=df_text.append(top8_dict, ignore_index=True)
    df = df.join(df_text)
    
    #HoG feature
    hog_arr = []
    for patch in patch_arr:
        if (patch.shape >= (4,4)):
            hogpatch = hog(patch, pixels_per_cell=(4, 4), cells_per_block=(1, 1))
            hog_arr.append(np.max(hogpatch))
        else:
            hog_arr.append(0.)
    df['hog'] = hog_arr
    
    #Replace NaN values and scale
    df = df.replace([np.inf], sys.float_info.max)
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    
    columns_to_scale = ['edge1', 'edge2', 'edge3', 'bar1', 'bar2', 'bar3', 'gauss', 'lap', 'hog']
    scaler = MinMaxScaler(feature_range=(0, 1))
    for column in columns_to_scale:
        df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
    
    return df