import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField, FileField, RadioField, BooleanField
from flask_wtf.file import FileRequired
from wtforms import StringField, PasswordField
from wtforms.validators import DataRequired

# The following form is used to login, using the username and password.
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember')
    next = SubmitField('Next')


# The following form is used to register using the username, password and user type.
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    user_type = RadioField('User Type', choices = ['Radiologist', 'Clinician', 'AI Community'], validators=[DataRequired()])
    next = SubmitField('Next')

# The following form is used to select the model from a drop down box.
class SelectModelForm(FlaskForm):
    path = "./static/models"
    dir_list = sorted(os.listdir(path))
    select_model = SelectField('Name of the Model',
                               choices=dir_list,
                               validators=[DataRequired()])
    next = SubmitField('Next')

# The following form is used to select the model to use and upload the image to diagnose.
class SelectModelAndUploadImageForm(FlaskForm):
    path = "./static/models"
    dir_list = sorted(os.listdir(path))
    select_model = SelectField('Name of the Model',
                               choices=dir_list,
                               validators=[DataRequired()])
    upload_image = FileField('Please Upload Image to diagnose',
                                     validators=[FileRequired()])
    next = SubmitField('Next')


# The following form is used to upload a new model.
class UploadModelForm(FlaskForm):
    upload_model = FileField('Please Upload the model.',
                                     validators=[FileRequired()])
    path = "./static/datasets"
    dir_list = sorted(os.listdir(path))
    dataset_used = SelectField('Name of the Dataset Used',
                                 choices=dir_list,
                                 validators=[DataRequired()])
    next = SubmitField('Next')

# The following form is used to select the dataset and to enter a new name for the model.
class SelectDatasetAndModelForm(FlaskForm):
    path = "./static/datasets"
    dir_list = sorted(os.listdir(path))
    select_dataset = SelectField('Name of the Dataset',
                                 choices=dir_list,
                                 validators=[DataRequired()])
    model_name = StringField('Model Name', validators=[DataRequired()])
    next = SubmitField('Next')

# THe following form is used to select the dataset to use.
class SelectDatasetForm(FlaskForm):
    path = "./static/datasets"
    dir_list = sorted(os.listdir(path))
    select_dataset = SelectField('Name of the Dataset',
                                 choices=dir_list,
                                 validators=[DataRequired()])
    next = SubmitField('Next')