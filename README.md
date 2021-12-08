
### **Summary**

the project objective is to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

### **File Description**
Here's the file structure of the project:
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
```
There are three components for this project.
**1. ETL Pipeline**

process_data.py` a script for data cleaning that loads the  `messages`  and  `categories`  datasets, Cleans the data and stores it in a SQLite database

**2. ML Pipeline**

`train_classifier.py` a machine learning pipeline that:
-   Loads data from the SQLite database and builds a text processing, machine learning pipeline and exports the final model as a pickle file

**3. Flask Web App**
this part will display the results in a Flask web app.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
