# About
A project for the Data Mining and Machine Learning course at University of Pisa, implementing a tool to detect fraudulent transactions.

# ReadMe file

# MONEY GUARD
A service implemented for the project of Data Mining and Machine Learning of the Artificial Intelligence and Data Engineering Master Degree at University of Pisa.

# Additional files to download

Since they were too big to be uploaded on Github, here are the links to the files required to run the application:
1. Dataset: https://drive.google.com/file/d/1X2CRd03MF5aj_UiC6tjUcYw3keM6Az4S/view?usp=sharing
2. Random Forest model (optional): https://drive.google.com/file/d/1MASosyvPb0nm8Gp69NkfdNZauRoxTFJV/view?usp=sharing
3. KNN model (not used in the application): https://drive.google.com/file/d/1o32XIOWbGBx-I_u-KSnXr_j5JwIkdumW/view?usp=sharing

Random Forest model is not required, because it can be generated very quickly, but the download is suggested. KNN model is not used in the application, but for the sake of completeness we’ve included all the models we’ve worked with. If you want to have it as well, it is highly recommended to download the model from this link, because generate the model takes hours.
The models have to be inserted in the “models” folder of the application; the dataset file should be included in a “dataSources” folder (that has to be created) in the main directory of the application.

# Application Installation (Windows & Linux)
To run the application, it is required to install some libraries before. First of all, download the most recent version of javabridge from the following link, making sure you download the appropriate version (32 or 64 bit):
https://www.lfd.uci.edu/~gohlke/pythonlibs/#javabridge
From your Anaconda, PyCharm or similar environment, use the following commands from the terminal:
1. pip install numpy
2. pip install pandas
3. pip install C:\where\you\downloaded\it\javabridge-X.Y.Z.whl
	(For Linux, just use pip install javabridge)
4. pip install python-weka-wrapper3
5. pip install matplotlib (optional)
6. Run the script smote.py to install smote

# How to run the application
To run the application, you just have to run the “main.py” module from an Anaconda terminal or Pycharm.
