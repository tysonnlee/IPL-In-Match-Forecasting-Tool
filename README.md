# 🏏 IPL 2nd Innings Chase Success Probability Model

This repository contains a pre-processed dataset and machine learning script that predicts whether a team successfully **chases a target score** using Indian Premier League (IPL) matches across all seasons to 2025.  

The project uses a **Random Forest Classifier** with hyperparameter tuning and is evaluated on data from the **2025 IPL season** as a hold-out test set.  

---

## 📂 Dataset
- Input dataset: **`win_viz_df.xlsx`**
- Contains all historical IPL ball-by-ball data (up to 2025) with pre-processed features pertaining to the context of the match and the target variable (`chased_successfully`).

---

## 🚀 Features
- Load pre-processed historical IPL data  
- Establish a **baseline model** (log loss, accuracy, F1 score)  
- Perform **GridSearchCV** with **Stratified K-Fold cross-validation** focusing on the optimisation of log loss.
- Tune Random Forest hyperparameters (`n_estimators`, `max_depth`, `min_samples_leaf`)  
- Evaluate model on **2025 IPL season test set**  
- Generate:
  - Dataframe with log-loss, accuracy and F1-scores across all grid search permutations
  - Feature importance ranking  
  - ROC Curve with AUC metric  
- Save the trained model with **joblib**

---

## 📂 Repository Structure
├── win_viz_df.xlsx # Input dataset contained in the Git folder
├── ipl_win_viz_modelling.py # Main training and evaluation script
├── results.xlsx # (Generated) CV results exported for analysis
├── rf_model.pkl # (Generated) Saved optimal Random Forest model
├── README.md # Project documentation
└── requirements.txt # Dependencies

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>

    pip install -r requirements.txt
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    joblib
    scipy
    openpyxl

## Usage
1. Place your dataset (win_viz_df.xlsx) in the repo root.
2. Update paths inside model_training.py where marked:
    - df = pd.read_excel('#Enter path to dataset here')
    - results_df.to_excel('#Enter path to output results for analysis')
    - joblib.dump(rf_best_model, '#Enter path to output model/rf_model.pkl'). You can use this for future predictions of in-game win probabilities.
    3. Run the script:
    - ipl_win_viz_modelling.py

## 📝 Further Reading
I’ve written about this project in more detail on my Substack:
👉 Predict the Future: How to use Machine Learning to Forecast T20 Match Outcomes – (https://substack.com/@tysonleeeee/p-170726859)

## 🙌 Acknowledgements

    - IPL ball-by-ball data sourced externally from Kaggle (https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025))
    - scikit-learn, pandas, numpy, seaborn, matplotlib
