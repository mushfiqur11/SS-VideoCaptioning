# SS-VideoCaptioning
This repository contains the implementational codes of our paper "Semantically Sensible Video Captioning (SSVC)" in Tensorflow 2. The model was trained on Google Cloud Platform with no use of GPU. The process to run the codes are as follows:
* Run the vid2frames.ipynb files to convert the videos of MSVD dataset into successive frames of images where the images will be stored in "framed_dataset" directory and a CSV file will be generated containing the frame names.
* Convert the extracted features from the images as pickle files after passing through the VGG16 architecture and store the pickle files in a folder named "pickle_files".
* Download "glove.6B.100d.txt" as the pretrained embedding layer.
* After having the required files, train the video captioning model by running the train.ipynb notebook. It is to be mentioned that our code provides enough flexibility to change the architecture by changing the options in the train.ipynb notebook and analyze different combinations that are shown in the paper. For example, there are options to change the encoder and decoder type, number of layers of encoder to be used, temporal and max length of the caption and spatial hard pull units(join_seq_out). However, due to lack of GPU and some other computational constraints, we kept some of the parameters constant in our study as described in our work.

Note: Avoid the first two steps, if you use the given pickle files and the CSV files along with our code to save time.
