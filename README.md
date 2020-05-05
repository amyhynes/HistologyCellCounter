# Histology Cell Counter
This script counts the number of positive Iba1 cells in an image. The decision tree model which powers the script was trained and tested on Iba1 images of mouse brains provided by Stephanie Tullo. 5 sample images are provided here in the data folder. The model performs well in accuracy (77%) and recall (87%). Most positive cells are classified correctly; however the researcher should check the resulting image with the classified cells labelled to remove false positives. More details about how the model was trained and tested can be found in [this repository](https://github.com/amyhynes/ihc-ml).

## Setting up the Software
To get started running the code in this notebook, make sure you are using Python 3 (3.7.4 used to create and test this script) and run the following in the command line: `pip3 install -r requirements.txt`.

## Running the Script
To run the script from the command line, use: `python count_cells.py image_file [--img]`, where image_file is the path to the image to be counted, and the `--img` flag is optional. This repository comes with 4 sample images found in `data/sample_iba1_images/`. An example to run this script is `python count_cells.py data/sample_iba1_images/0.TIF` or `python count_cells.py data/sample_iba1_images/0.TIF --img`

### Output
The output of this script is the number of positive cells as predicted by the decision tree classifier, and if the optional `--img` flag is used, the original image is displayed with the positively classified cells labelled.
