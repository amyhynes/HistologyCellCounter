# Histology Cell Counter
This script counts the number of positive Iba1 cells in an image. The decision tree model which powers the script was trained and tested on Iba1 images of mouse brains provided by Stephanie Tullo. 5 sample images are provided here in the data folder. The model performs well in accuracy (77%) and recall (87%). Most positive cells are classified correctly; however the researcher should check the resulting image with the classified cells labelled to remove false positives. More details about how the model was trained and tested can be found in [this repository](https://github.com/amyhynes/ihc-ml).

## Setting up the Software
To get started running the code in this notebook, make sure you are using Python 3 (3.7.4 used to create and test this script) and run the following in the command line: `pip3 install -r requirements.txt`.

## Running the Script

### Output
