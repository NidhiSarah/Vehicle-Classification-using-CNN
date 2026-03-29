# Vehicle-Classification-using-CNN
CNN-based image classification project to identify vehicle types with a Flask web interface.

Objectives
- Build a deep learning model for vehicle classification
- Compare ReLU vs Leaky ReLU activation functions
- Develop a web interface for user interaction
  
Model Details
- Architecture: CNN (Conv2D, MaxPooling, Dense, Dropout)
- Input Size: 224x224 RGB images
- Optimizer: Adam
- Loss Function: Categorical Crossentropy

Approaches:
1. CNN with ReLU
2. CNN with Leaky ReLU (Best Performance)

 Results
- ReLU Accuracy: 88%
- Leaky ReLU Accuracy: 90%
- Better generalization with Leaky ReLU

Dataset
- Source: Kaggle Vehicle Dataset
- Total Images: 15,645
- Classes:
  - City Car
  - Sedan
  - SUV
  - Truck
  - Big Truck
  - Van
  - Multi-purpose vehicle
 
Features
- Image classification using deep learning
- Data augmentation (rotation, zoom, flip)
- Performance evaluation (Accuracy, Precision, Recall, F1-score)
- Flask web interface for predictions


