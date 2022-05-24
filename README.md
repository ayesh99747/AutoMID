# AutoMID - A novel framework for Automated Computer Aided Diagnosis of Medical Images
### By  - Ayeshmantha Wijegunathileke

## How to setup
1. Clone the project to the required location.
2. Create a python virtual environment.
3. Run the requirements.txt file with python to install the libraries.
4. Use the command python app.py to run the server.

## User Accounts
There are three types of user accounts.
### Radiologist
+ The radiologist can do everything and has access to all the functions of the application. Given below are the available functions -
  + Diagnose Image.
  + Create New Model.
  + Upload New Model.
  + View Model Statistics
  + View Leaderboard.

### Clinician
+ The clinician can do only a few functions of the application. Given below are the available functions -
  + Diagnose image.
  + View Model Statistics.
  + View Leaderboard.
  
### AI Community
+ The AI Community can do only a few functions of the application. Given below are the available functions -
  + Upload New Model.
  + View Model Statistics.
  + View Leaderboard.
  
You can create your own user account with a unique username and password or you can use the existing credentials provided below.

    Username - Radiologist_1
    Password - radio123

    Username - Clinician_1
    Password - clinician123

    Username - Community_1
    Password - community123

    Username - Community_2
    Password - community123

## Help
Once logged in there is a menu item called help. Once help is clicked, you will be able to view a page with step-by-step instructions on how to use the application.

## Setup
### Images
A folder should be created in the static folder to save the uploaded images.
### Datasets
There are two datasets being used. These have not been uploaded to GitHub.
 + Dataset 1 - This dataset contains images of Normal, Covid-19 and Pneumonia Chest X rays. [Link](https://www.kaggle.com/datasets/amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset?select=normal)
 + Dataset 2 - This dataset contains images of Normal, Viral Pneumonia and Bacterial Pneumonia Chest X rays. [Link](https://www.kaggle.com/datasets/inhcngphan/chest-xray)

These datasets need to be saved in static/datasets folder. They have to be divded into train, val and test sets. The ratios are upto you but it is recommended to use .8 train, .1 test and .1 val.

### Dataset Info Json
There is a json file to store the metadata of the datasets. The format to add a new dataset is given below.
```    
  "<Dataset Number>": {
      "Name": "<Dataset Name>",
      "Labels": [
        "<Label 1>",
        "<Label n>"
      ]
    }
```


# AutoMID
