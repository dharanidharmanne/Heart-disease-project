# Heart Disease Prediction Project

This repository contains the **Heart Disease Prediction Project**, which utilizes machine learning techniques to predict the likelihood of heart disease based on various health parameters.

## Abstract

This project aims to develop a predictive model for heart disease. Early prediction can significantly aid in timely diagnosis and intervention, potentially improving patient outcomes. We explore various machine learning algorithms, focusing on their accuracy and performance in classifying individuals at risk of heart disease. The project involves data exploration, preprocessing, model training, and evaluation.

## Methodology

1.  **Data Collection and Preprocessing**:

      * The project utilizes the `heart.csv` dataset, which contains various medical attributes.
      * Data cleaning, handling missing values, and feature scaling are performed.
      * Categorical features are encoded.

2.  **Exploratory Data Analysis (EDA)**:

      * Visualizations are used to understand the distribution of features and their correlation with heart disease.
      * Key insights into the dataset are extracted.

3.  **Model Selection and Training**:

      * Several machine learning models are explored for classification, including:
          * Logistic Regression
          * K-Nearest Neighbors (KNN)
          * Support Vector Machine (SVM)
          * Decision Tree Classifier
          * Random Forest Classifier
          * XGBoost Classifier
      * The dataset is split into training and testing sets.
      * Models are trained on the training data.

4.  **Model Evaluation**:

      * Models are evaluated using metrics such as:
          * Accuracy
          * Precision
          * Recall
          * F1-score
          * Confusion Matrix
          * ROC Curve and AUC Score
      * Cross-validation techniques are employed to ensure robust evaluation.

## Files in this Repository

  * `heart.csv`: The dataset used for heart disease prediction.
  * `Machine Learning Project.ipynb`: Jupyter Notebook containing the complete code for data loading, preprocessing, model training, and evaluation.
  * `Heart Disease Prediction.pptx`: A presentation summarizing the project.
  * `TEAM 10 _Machine Learning _Final_Report.pdf`: The final report detailing the project.
  * `.ipynb_checkpoints/`: Directory containing checkpoints of Jupyter Notebooks.
      * `Machine Learning Project-checkpoint.ipynb`
      * `Untitled-checkpoint.ipynb`
      * `Untitled1-checkpoint.ipynb`
  * Image files (e.g., `1.png`, `2.png`, `3.png`, `3 (2).png`, `4.png`, `5.png`, `6.png`, `7.png`, `8.png`): These likely represent visualizations or results from the project.

## How to Use

1.  **Clone the repository**:

    ```bash
    git clone <repository-url>
    ```

2.  **Navigate to the project directory**:

    ```bash
    cd heart-disease-project
    ```

3.  **Install dependencies**:
    *(The specific dependencies are likely listed in the Jupyter Notebook or would typically be found in a `requirements.txt` file, which is not provided in the current context. Common dependencies for this type of project include `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`.)*

    You might need to install them manually if a `requirements.txt` is not available:

    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn xgboost
    ```

4.  **Run the Jupyter Notebook**:

    ```bash
    jupyter notebook "Machine Learning Project.ipynb"
    ```

    This will open the notebook in your web browser, where you can execute the cells to reproduce the analysis and model training.

## Contributors

  * Dharanidhar Manne
  * Ravi Prasad Grandhi
  * Sravya Reddy Kaitha
