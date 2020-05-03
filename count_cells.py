import joblib
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from helpers import preprocess_iba1
from helpers import extract_features
from helpers import bbox

exception = """\
This script will count the number of positive Iba1 cells in an image.

Usage:  count_cells.py image_file [--img]
"""

if len(sys.argv) < 2 or  len(sys.argv) > 3:
    print(exception)
    sys.exit(0)

try:
	img = io.imread(sys.argv[1])
except:
	print(exception)
	sys.exit(0)

label_image, patch_arr, patches = preprocess_iba1.preprocess(img)

X_test = extract_features.extract_features(label_image, patch_arr).to_numpy()

loaded_model = joblib.load('helpers/finalized_dt_model.sav')
y_pred = loaded_model.predict(X_test)
result = np.sum(y_pred)
print("Number of Positive Iba1 cells: ", result)

pos_patches = []
i = 0
for pred in y_pred:
	if pred == 1:
		pos_patches.append(patches[i])
	i+=1

if len(sys.argv) > 2 and sys.argv[2] == '--img':
	#generate bbox image
	bbox.bbox_img_generator(img_as_float(img), pos_patches)