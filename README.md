# Competency-4
Neural network with pytorch
Project Overview
I built a neural network using PyTorch to classify clothing images into 10 categories.
Dataset
I used Fashion-MNIST, which has 60,000 training images and 10,000 test images. I split the training data into 50,000 for training and 10,000 for validation.
Citation: Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747.
License: MIT.
Setup
Install the packages:
python -m pip install torch torchvision matplotlib
Run the code:
python fashion_mnist.py
Model
I used 128 hidden neurons, ReLU activation, CrossEntropyLoss, and the Adam optimizer. I trained the model for 5 epochs.
Experiment
Prediction: I thought increasing the neurons from 128 to 256 would improve performance.
Results: Both models improved, but the modified model finished with lower training and validation loss.
What I Learned: Adding more neurons can help the model learn better, but it doesn't always guarantee better results.
Training Results
Model,Epoch,Train Loss,Validation Loss
Original,1,0.5642654227304459,0.45247245845794676
Original,2,0.41619599262714385,0.41035620646476745
Original,3,0.3736749732875824,0.36480736074447634
Original,4,0.34450921129226686,0.35468867366313933
Original,5,0.32212817761421203,0.37389751851558684
Modified,1,0.5423956701278686,0.4162962657690048
Modified,2,0.39232016700744626,0.3641300819158554
Modified,3,0.35158046857833863,0.36265724830627444
Modified,4,0.32452452964782713,0.3653725571632385
Modified,5,0.3065392286968231,0.31464105706214907
Both models improved throughout training. The original model showed some signs of overfitting near the end.
My results are saved in results.csv and training_results.png.
Next Steps
I would try more epochs or a different learning rate to see if accuracy improves.
AI Assistance
I used ChatGPT to help with fixing errors, and understanding my results.
