# Ekonify
A Garbage Classification using Machine Learning

## Problem Statement
According to the World Bank's What a Waste 2.0 report, low-income countries often have recycling rates below 5%, compared to high-income countries where rates exceed 40% [2]. Lagos faces similar challenges due to a growing population and limited waste management infrastructure. Manual sorting is time-consuming, costly, and prone to errors, while public unawareness worsens inefficiencies. While studies like TrashNet [3] and Garbage Type Detection [4] demonstrate the potential of machine learning in waste classification, their reliance on generalised datasets limits their adaptability to specific regions like Lagos. Additionally, these solutions lack integration with public engagement platforms, which are crucial for driving scalable, community-wide impact. How can AI bridge the gap in Lagos’ waste management system and catalyse a greener, more sustainable future for its 20+ million residents?

## Project Overview
Ekonify is a machine learning project designed to classify different types of garbage into 12 categories. The goal is to develop an efficient classification model using various machine learning techniques, optimizing performance through regularization, tuning hyperparameters, and error analysis. This implementation helps in automating waste segregation, promoting recycling, and reducing environmental pollution.

## Dataset Used
The dataset used for this project consists of **15,515 labeled images** belonging to **12 classes**:
- Battery
- Biological Waste
- Brown Glass
- Cardboard
- Clothes
- Green Glass
- Metal
- Paper
- Plastic
- Shoes
- Trash
- White Glass  

Source: [Kaggle - Garbage Classification (12 Classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

## Implementation Details
### 1. Classical Machine Learning Model
- **Algorithm Used**: Logistic Regression
- **Feature Extraction**: RGB pixel values and Histogram of Oriented Gradients (HOG)
- **Hyperparameter Tuning**: Regularization strength (C), Solver choice
- **Performance**: Moderate accuracy, but limited scalability for image data

### 2. Basic Neural Network (No Optimization)
- **Model**: Basic CNN with 3 Conv layers
- **No Optimizations Used**: Default settings, no early stopping, no dropout, and default learning rate
- **Results**: Initial accuracy was **70.88%** on test data, but prone to overfitting

### 3. Optimized Neural Network (Advanced Tuning)
- **Optimizations Applied**:
  - Optimizer: **AdamW**
  - Regularization: **L2 Regularization (0.0005)**
  - Dropout: **0.4**
  - Learning Rate Scheduling: **ReduceLROnPlateau**
  - Early Stopping: **Enabled (Patience = 3)**
  - Data Augmentation: **Applied (rotation, zoom, shift, horizontal flip)**
- **Results**: Validation accuracy improved to **69.96%** with reduced overfitting

## Training Results & Comparison
| **Instance** | **Optimizer** | **Regularizer** | **Epochs** | **Early Stopping** | **Layers** | **Learning Rate** | **Accuracy** | **Loss** | **F1-Score** | **Precision** | **Recall** |
|-------------|--------------|----------------|----------|----------------|---------|----------------|------------|----------|------------|------------|--------|
| Instance 1 | Default SGD | None | 10 | No | 3 Conv Layers | 0.001 | 70.88% | 0.84 | 0.61 | 0.65 | 0.60 |
| Instance 2 | Adam | L2 (0.001) | 20 | No | 3 Conv Layers | 0.0005 | 65.86% | 1.28 | 0.65 | 0.67 | 0.63 |
| Instance 3 | AdamW | L2 (0.0005) | 21 | Yes | 3 Conv Layers + BatchNorm | **0.000045** | **69.96%** | **1.35** | **0.70** | **0.72** | **0.68** |

## Error Analysis & Observations
🔹 **Overfitting Observed in Initial Training**: Basic NN model had high training accuracy but poor validation accuracy. Adding **regularization and dropout** improved generalization.  
🔹 **ReduceLROnPlateau Was Effective**: The learning rate scheduler helped improve validation accuracy after reducing LR from **0.0005 → 0.00015 → 0.000045**.  
🔹 **Optimized CNN Outperformed Logistic Regression**: Classical ML struggled due to lack of feature abstraction.  

## Summary & Conclusion
- **Best Performing Model**: **Optimized CNN with AdamW + Regularization + Dropout (69.96% accuracy).**
- **Key Takeaways**:
  - Early stopping prevented unnecessary training.
  - Learning rate adjustments helped stabilize loss.
  - Data augmentation improved generalization.

## How to Run the Notebook
1️. Clone the repository:
   ```bash
   git clone https://github.com/eadewusic/Ekonify
   cd Ekonify
   ```
2️. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3️. Run the notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
4️. Load the best-trained model:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('saved_models/best_model.keras')
   ```

## Video Presentation
Here's a detailed 5-minute explanation covering model implementation, optimization techniques, and results. [Video Link Here](https://www.youtube.com/@climiradiroberts)


## Final Thoughts
This project successfully demonstrated the impact of **hyperparameter tuning and optimization techniques** in improving machine learning model performance. Future improvements could include **transfer learning** for even better accuracy.
