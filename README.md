🧠 Job Salary Predictor
A Machine Learning-powered web app that predicts the expected salary for a tech job based on various inputs like experience level, job title, employment type, company size, location, and more. Built with Streamlit for interactive use.

🚀 Features
Predicts salary based on:

1. Job Title
2. Experience Level
3. Employment Type
4. Company Size
5. Company Location
6. Remote Work Ratio
7. Interactive UI built with Streamlit
8. Model trained on real-world salary data
9. Deployed locally or can be deployed to Streamlit Cloud or HuggingFace Spaces

📊 Tech Stack
1. Python 3
2. Pandas, Numpy
3. Scikit-learn
4. Streamlit
5. Joblib for modal serialization

📂 Project Structure
Job_Salary_Predictor/
│
├── data/                  # Dataset files
├── model/                 # Trained ML model (.pkl or .joblib)
├── train_model.py         # Code to clean data, train & save the model
├── app.py                 # Streamlit frontend
├── README.md              # Project description
├── requirements.txt       # Python dependencies
└── .gitignore             # Files and folders to ignore

🖥️ How to Run Locally

1. Clone the Repository:
   git clone https://github.com/shivprasad08/Job_Salary_Predictor.git
   cd Job_Salary_Predictor

2. Install Dependencies:
   pip install -r requirements.txt

3. Run the Streamlit App:
   streamlit run app.py

🧠 Model Training
To retrain the model on updated data:
python train_model.py

