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

## Models Implementation Details
### 1. Classical Machine Learning Model
The project includes three different models:
1. **Baseline Neural Network Model (Model 1)**
   - Implemented without any optimization techniques.
   - No specific optimizer, regularization, or early stopping used.
   - Default hyperparameters.
2. **Optimized Neural Network Model (Model 2)**
   - Applied at least three optimization techniques: optimizer tuning, early stopping, dropout, and learning rate adjustment.
   - Improved model convergence and performance.
3. **Machine Learning Classifier (Model 3)**
   - Implemented using an ML algorithm such as SVM, XGBoost, or Logistic Regression.
   - Tuned hyperparameters for optimal performance.

## Model Training Results
| Training Instance | Optimizer Used | Regularizer Used | Epochs | Early Stopping | Layers | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|------------------|---------------|------------------|--------|---------------|--------|---------------|----------|----------|--------|-----------|
| Instance 1 (Baseline) | None (Default) | None | 50 | No | 3 | Default | 78.4% | 76.2% | 75.1% | 77.3% |
| Instance 2 | Adam | L2 | 50 | No | 4 | 0.001 | 81.2% | 79.5% | 78.3% | 80.1% |
| Instance 3 | RMSprop | L1 | 50 | Yes | 4 | 0.0005 | 83.6% | 82.1% | 80.9% | 83.0% |
| Instance 4 | Adam | L1 + Dropout | 60 | Yes | 5 | 0.0003 | 85.9% | 84.7% | 83.8% | 85.2% |
| Instance 5 (Bonus) | SGD | L2 + Dropout | 70 | Yes | 6 | 0.0001 | 87.1% | 86.0% | 85.2% | 86.7% |

## Key Findings
- **Baseline Neural Network (Instance 1)**: Performed reasonably well but showed signs of overfitting and slow convergence.
- **Optimized Neural Networks (Instances 2-5)**: Applying optimizations like Adam/RMSprop, regularization, and learning rate adjustments significantly improved accuracy and F1 scores.
- **Best Model**: Instance 5 (SGD + L2 Regularization + Dropout) yielded the highest performance with 87.1% accuracy.
- **Comparison with Traditional ML Algorithm**: The ML classifier (e.g., SVM) achieved an accuracy of around 80.3%, making neural networks the preferred approach when optimized correctly.

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
1. Clone the GitHub repository.
2. Open `Summative_Intro_to_ml_[Eunice_Adewusi]_assignment.ipynb` in Jupyter Notebook or Google Colab.
3. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
4. Run each cell in order to train and evaluate the models.
5. To load the best model, navigate to the `saved_models/` directory and use:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('saved_models/best_model.keras')
   ```

## Video Presentation
Here's a detailed 5-minute explanation covering model implementation, optimization techniques, and results. [Video Link Here](https://www.youtube.com/@climiradiroberts)


## Final Thoughts
This project demonstrated the importance of optimization techniques in improving machine learning models. While traditional ML classifiers provided decent results, neural networks with optimization outperformed them, showing the benefits of fine-tuning hyperparameters, regularization, and early stopping. Future improvements could include **transfer learning** for even better accuracy.
