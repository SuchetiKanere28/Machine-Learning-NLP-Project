# 🧠 Automated Job Role Prediction from Resume and Job Description using Traditional Machine Learning Approaches

The objective of this project is to automatically classify job positions with the help of Natural Language Processing (NLP) and Machine Learning (ML) algorithms. The objective is to predict the appropriate category or position of the job based on the resume or job description text.
The concern is important because resume screening and role matching take too much time manually and are subject to human errors. A smart system with a capability to process text and classify roles correctly saves job seekers, recruiters, and HR departments time in hiring.
Two data sets were utilized — one with resumes and the other with categorized updated resumes. A series of ML models were trained and tested after post-text preprocessing and TF-IDF vectorization. In all, the Random Forest Classifier provided the optimal results with an accuracy rate of approximately 93% for Dataset 1 and 90% for Dataset 2.
Experiment results show that standard ML approaches, with the addition of proper text representation, can successfully carry out automated job role prediction with acceptable accuracy and efficiency.

# 📂 Dataset Source
🧾**Dataset 1** – Resume Dataset 

**File Name:** Resume.csv

**Source:** Rescued from Kaggle's open resume classification dataset.

**Size:** Around 4,900 records.

**Columns utilized:** Resume, Category, and ID.

**Description:** Each of the records comprises a text resume and associated job category label like Data Science, HR, Web Development, Testing, Operations, etc.

**Preprocessing Performed:**

* Removal of unwanted digits, punctuation, and characters

* Conversion of text into lower case for standardization

* Missing values and null fields processed

* Only Cases (Resume, Category, ID) are kept

* Text transformation through TF-IDF vectorization for training the model


🧾**Dataset 2** – Updated Resume Dataset 

**File Name:** UpdatedResumeDataSet.csv

**Source:** Extended Resume dataset augmented with more cleaned text samples (from research repositories and Kaggle).

**Size:** Around 5,000 records.

**Columns utilized:** Resume, Category, and ID.

**Preprocessing Performed:**

* Deletion of HTML tags, special characters, and repeated whitespace

* Text normalization and tokenization

* Null or invalid value replacement

* TF-IDF vectorization for text representation

# ⚙️ Methods
🎯 **Problem Approach**

The general goal of this project is to automatically classify job positions from text information — i.e., resumes or job postings — based on Natural Language Processing (NLP) and Machine Learning (ML) methods.

To achieve that, the process was broken down into four primary stages:

<img width="772" height="400" alt="image" src="https://github.com/user-attachments/assets/0c4db54a-d84b-4228-8713-2161191f2be9" />

💡 **Why This Approach Works**

* TF-IDF is able to effectively identify relevance of words between job postings and resumes without favouring highly recurring but less explanatory terms.

* Classic ML models such as Random Forest and Logistic Regression are computationally lightweight and can handle medium-sized datasets and are therefore relevant in real-world recruitment scenarios.

* NLP preprocessing (normalization, tokenization, and cleaning) ensures the model to learn good text patterns and not noise or non-material syntax.

The method keeps precision, interpretability, and efficiency just right and is therefore realistic for HR automation and smart resume filtering platforms.


🔁**Alternative Approaches Considered**


<img width="1056" height="307" alt="image" src="https://github.com/user-attachments/assets/fc8eefe9-a36e-4583-82ad-abe779743846" />

🧭 **Methodology Flow Diagram**

<img width="800" height="1000" alt="image" src="https://github.com/user-attachments/assets/8e3842e4-7c11-4580-b214-58091046bdc9" />



🚀**Steps to Run Code**
* **Clone the Repository**
* **Install Required Dependencies**
  * pip install -r requirements.txt
* **Ensure Model and Dataset Files Are in Place**
  * Place your model and vectorizer files in the project folder:
      * Dataset1.pkl, tfidf_Dataset1.pkl
      * Dataset2.pkl, tfidf_Dataset2.pkl
  * Place your datasets in the same directory:
      * Resume.csv
      * UpdatedResumeDataSet.csv
* **Run the Streamlit Application**
  * streamlit run app.py
* **Use the Application**
  * From the sidebar, choose which dataset you want to use (Dataset 1 or Dataset 2).
  * Enter or paste a job description into the text box.
  * Click on Predict to view the predicted job role.
  * You can also click Evaluate Models to see accuracy metrics for that dataset.

# 📊 Experiments and Results Summary
For comparison of the efficacy of the NLP-based Job Role Classification system, some classic machine learning models were trained and compared on two data sets (Resume.csv and UpdatedResumeDataSet.csv). All models were compared based on uniform preprocessing, TF-IDF feature extraction, and the same data splits to provide a genuine comparison.

⚗️**Experimental Setup**

* Text Preprocessing:
Special characters removed, text converted to lower case, and whitespace regulated.

* Feature Extraction:
TF-IDF Vectorization with max feature size 5000 used.

* Train-Test Split:
80% for training and 20% for testing to test the model.

* Evaluation Metrics:
The performance of all the models was good according to Accuracy, Precision, Recall, and F1-score. All the results indicate that SVM performed best in all cases for both datasets.

📈**Model Comparison**

<img width="800" height="294" alt="image" src="https://github.com/user-attachments/assets/76ca107a-bb4c-45d7-9d7f-93f34c27c71a" />


**Observations**
* SVM was optimal because it could process high-dimensional TF-IDF features efficiently.

* Logistic Regression was average in the trade-off between interpretability and accuracy.

* Naïve Bayes was quicker but sensitive to noise and vocabulary fluctuations.

* Visualizations like confusion matrices and category distribution plots were employed to interpret model predictions and indicate repeated misclassifications.


🖥️**Streamlit Interface**

A Streamlit web app was created exclusively for user-interactive and user-based classification.

Features:
* Live Prediction: The user can input manually or copy-paste resume content or job descriptions to receive live job role predictions.

* Model Performance: Covers on-demand performance metrics like accuracy, precision, recall, and F1-score.

* Dataset Switching: The application features the option of switching between two datasets (Dataset 1 and Dataset 2) to see how different data sources are handled by models.

* Minimal Visualization: The dashboard is mostly about usability and minimalism, with no graphical noise to create room for metric value.

* Ease of Use: Deployment is local through streamlit run app.py, offering a GUI that does not need coding skill.

The integration demonstrates how machine learning models can be shifted from offline trial to an easily embraced real-world solution for recruiters, HR practitioners, and data analysts.

<img width="1919" height="863" alt="image" src="https://github.com/user-attachments/assets/cc6ed30e-1516-43bf-81b4-aea6f23a48e7" />



# 🧩 Conclusion
This project is successfully proven wherein Machine Learning and Natural Language Processing (NLP) can be employed for automatic classification of job titles from text-based resume data. Employing Support Vector Machine (SVM), Naïve Bayes, and Logistic Regression models, we found SVM to work best on both sets with the highest accuracy and balanced precision-recall tradeoff.

The paper also emphasizes the significance of preprocessing data, such as text cleaning and TF-IDF feature engineering, in order to ensure model reliability and accuracy. The Streamlit web interface developed fills the gap between technical deployment and real usability, such that users are able to interact with models easily, evaluate, and get real-time predictions.

In the future, this paradigm can be augmented with deep learning algorithms like BERT or LSTM networks for better contextual analysis, and resume screening and recommendation software for hiring portals.

# 📚 References

* Scikit-learn Documentation: https://scikit-learn.org/

* Streamlit Documentation: https://docs.streamlit.io/

* Dataset 1: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
* Dataset 2: https://www.kaggle.com/code/gauravduttakiit/resume-screening-using-machine-learning/input

* TF-IDF Vectorization Techniques — Stanford NLP Guide

* Python Joblib Library Documentation
