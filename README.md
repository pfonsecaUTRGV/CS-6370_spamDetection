# CS-6370_spamDetection

University of Texas Rio Grande Valley

This is a project for CS-6370 Foundations of Intelligent Security Systems Spring 2025 class. 

The goal of this work is to create a basic app that classify an email as "Spam" or "Not-Spam" base on the text body of the message. In order to achieve this, four machine learning models were trained and tested. The model with the best performance was choose for deployment in the basic app

# Data Base 
The data base used for training the models is the Kaggle, “Spam email Dataset,” www.kaggle.com. And it was taken from: https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset/data

# Instructions
For executing the code its imporatn to dowload the "email.csv" file to the same directory, or edit the path if the case its necessary. 
A virtual environment needs to be created. To do so run the following command in the command prompt

```
#Create a virtual environment
python -m venv virt

#Activate virtual environment
source virt/scripts/activate
```

In the virtual environment install the following packages for training the models: sklearn, numpy, pandas, matplotlib, joblib
```
#sklearn
pip install -U scikit-learn

#pandas
pip install pandas

#numpy
pip install numpy

#matplotlib
python -m pip install -U matplotlib

#joblib
pip install joblib
```

Scripts are runned using python command 
```
#Run machine learning model training and evaluation script
python spamModels.py

#Run ensemble approach training and evaluation script
python ensemble.py
```

To execute the web app install streamlit library

```
pip install streamlit
```

To run the app type the following commnad

```
streamlit run 6370_spam_app.py
```

The app will be automatically open on http://localhost:8501 on the default browser

"spam_detector.plk" & "spam_ensemble.plk" are the files that store the trained models. These files are needed to run the webapp.

