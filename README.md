AI Career Recommender System
============================

This project builds a machine learning model that recommends the most suitable career path based on a user's skills, interests, and preferred work style.

Overview
--------
The system uses a Decision Tree Classifier to predict a career category after converting textual inputs into numerical values through Label Encoding.  
The model demonstrates essential ML steps such as preprocessing, feature engineering, model training, model saving, and prediction.

Tech Stack
----------
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Jupyter Notebook

Project Structure
-----------------
AI-Career-Recommender/
- dataset/
  - career_data.csv
- src/
  - career_model.pkl
  - skill_encoder.pkl
  - interest_encoder.pkl
  - style_encoder.pkl
  - career_encoder.pkl
- notebook.ipynb
- requirements.txt
- README.md

Machine Learning Workflow
-------------------------

1. Data Preparation  
   - Custom dataset created  
   - Text columns encoded using Label Encoding  
   - Dataset prepared for classification  

2. Model Training  
   - Model: Decision Tree Classifier  
   - Input features: skills, interests, work style  
   - Output label: career_path  

3. Model Saving  
   The trained model and all encoders are saved using Joblib for later use.

4. Prediction Function  
   A reusable prediction function converts input text into encoded values and returns the recommended career.

Example Prediction
------------------
Input:  
skills = "python"  
interests = "machine learning"  
work_style = "independent"  

Output:  
Machine Learning Engineer

How to Run Locally
------------------

1. Clone the Repository  
   git clone https://github.com/shantanutech7/ai-career-recommender1.git

2. Install Dependencies  
   pip install -r requirements.txt

3. Run the Notebook  
   jupyter notebook  
   Open `notebook.ipynb` and run all cells.

Future Improvements
-------------------
- Build a Streamlit frontend for user interaction  
- Build a FastAPI backend for real-time prediction  
- Expand the dataset with more career options  
- Apply hyperparameter tuning and better ML algorithms  

Author
------
Shantanu Bawane  
AI/ML Engineering Learner  

