Here is some of my instructions for my project:

Some important files / folder:

* Folders:
	- Casia2: data folder
	- preprocessed_data:
		+ train.csv: feature vector and label for training ANN to classify tampered / non-tampered patches
		+ mean_and_var_of_trainset: mean and variance vector of training data to normalize test data
	- test_data: ~ 1000 test images for Colab testing only

* .py files
	- dataHandler.py, featureExtractor.py, testing_phase.py, training_phase.py: please read explanations in my report
	- prepare_test_data.py: to select test images from dataset and put them into 'test_data' folder 
	- test.py: for testing python syntax

* other files
	- trained_MLP_sklearn.sav: trained MLP model from sklearn
	- filename_labels.csv: labels for test images in 'test_data' folder. (Format: dict[filename] = 0 or 1)

Colab work link: https://colab.research.google.com/drive/1w85yye8m-HkAJpmIaRH8fdrOKbzgHQHz?usp=sharing

* "test_data" folder can be downloaded at: https://drive.google.com/drive/folders/1kA6X3L07IMtVqcrIvLtNhwsWBOR4SX0A?usp=sharing


