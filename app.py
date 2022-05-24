import argparse
import fcntl
import json
import os.path
import pty
import select
import shlex
import struct
import subprocess
import termios

from autogluon.vision import ImagePredictor
from flask import Flask, redirect, url_for, render_template, flash, request, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from forms import SelectModelAndUploadImageForm, SelectDatasetAndModelForm, LoginForm, RegistrationForm, \
    SelectDatasetForm, UploadModelForm, SelectModelForm
from flask_wtf.csrf import CSRFProtect

# Version of the application.
__version__ = "1.0.0"

app = Flask(__name__)

# Key generated using the secrets library.
app.config['SECRET_KEY'] = '7bf3493bd030ccff141842bd92f5f08b81e9e1beede6df94'
UPLOAD_FOLDER = './static/' # Upload folder.
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS_IMAGES = {'jpg', 'jpeg', 'png'} # Permitted image extensions.
ALLOWED_EXTENSIONS_MODELS = {'ag'} # Permitted model extensions.

app.config["fd"] = None
app.config["child_pid"] = None
socket_io = SocketIO(app)

# Initializing the database.
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
db = SQLAlchemy(app)

# Initializing the Login Manager
lm = LoginManager(app)
lm.login_view = 'login'

csrf = CSRFProtect(app)

# The following is the users table in the sqlite database.
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(1000))
    user_type = db.Column(db.String(100))

# The following is the models table in the sqlite database.
class Models(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), db.ForeignKey('user.username'), nullable=False)
    model_name = db.Column(db.String(100))
    dataset_used = db.Column(db.String(100))
    train_acc = db.Column(db.String(100))
    val_acc = db.Column(db.String(100))
    total_time = db.Column(db.String(100))
    model_used = db.Column(db.String(100))


# The following method is used when getting the user by id.
@lm.user_loader
def load_user(id):
    return User.query.get(int(id))


# This route is used to display the landing page of the application.
@app.route("/")
def home():
    if current_user.is_authenticated:
        username = current_user.username
        return render_template('index.html', username=username)
    else:
        return render_template('index.html')


# This route is used to display the help page of the application.
@app.route('/view_help', methods=['GET', 'POST'])
def viewHelp():
    return render_template('help.html')


# This route is used to render and handle the login functionality of the application.
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        user = User.query.filter_by(username=username).first()  # Searching for a user with the same username.
        if not user or not check_password_hash(user.password, password):
            flash('The login details provided are incorrect!')
            return redirect(url_for('login'))  # If user doesn't exist or password is wrong they will be redirected.
        login_user(user, remember=remember)  # Logging in the user to the system.
        return redirect(url_for('home'))
    return render_template('login.html', form=form)

# This route is used to render and handle the signup functionality of the application.
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegistrationForm()
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user_type = request.form.get('user_type')
        user = User.query.filter_by(username=username).first()  # Searching for a user with the same username.
        if user:  # If the same username exists, the form displays an error message.
            flash('Username already exists!')
            return redirect(url_for('signup'))
        # Creating a new user object.
        new_user = User(username=username, password=generate_password_hash(password, method='pbkdf2:sha256'),
                        user_type=user_type)
        # Inserting the new user to the database.
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html', form=form)

# This route is used to log the user out of the system.
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# This route is used to select the dataset to use when diagnosing an image.
@app.route("/select_dataset_to_predict", methods=['GET', 'POST'])
@login_required
def selectDatasetToPredict():
    form = SelectDatasetForm()
    if request.method == 'POST':
        session['select_dataset'] = request.form.get('select_dataset')
        return redirect(url_for('selectModel'))
    return render_template('select_dataset_to_predict.html', form=form)

# This route is used to select the model to use and upload the image to diagnose.
@app.route("/select_model", methods=['GET', 'POST'])
@login_required
def selectModel():
    form = SelectModelAndUploadImageForm()
    datasets = Models.query.filter_by(dataset_used=session['select_dataset']).order_by(
        Models.model_name.asc()).all()
    if(not datasets):
        flash('Sorry, no models were found for the selected dataset!')
    if request.method == 'POST':
        session['select_model'] = request.form.get('select_model')
        if form.upload_image.data:
            filename = secure_filename(form.upload_image.data.filename)
            session['upload_image'] = filename
            form.upload_image.data.save(app.config['UPLOAD_FOLDER'] + 'images/' + filename)
        return redirect(url_for('generateDiagnosis'))
    return render_template('select_model_and_image.html', form=form, dir_list=datasets)

# This route is used to produce a diagnosis for the image.
@app.route("/generate_diagnosis", methods=['GET', 'POST'])
@login_required
def generateDiagnosis():
    form = SelectModelAndUploadImageForm()
    datasets = Models.query.filter_by(dataset_used=session['select_dataset']).order_by(
        Models.model_name.asc()).all()
    predictor_loaded = ImagePredictor.load('./static/models/' + session['select_model'])
    image_path = "./static/images/" + session['upload_image']
    output = predictor_loaded.predict(image_path).at[0]
    probability = round(predictor_loaded.predict_proba(image_path)[output][0] * 100, 2)
    with open("./dataset-info.json") as file:
        data = json.load(file)
    for value in data.values():
        if (session['select_dataset'] == value["Name"]):
            labels = value["Labels"]
    return render_template("select_model_and_image.html", form=form,
                           prediction_text=labels[output], probability=probability,
                           filename=session['upload_image'], dir_list=datasets)

# This route is used to upload new models to the system.
@app.route("/upload_model", methods=['GET', 'POST'])
@login_required
def uploadModel():
    form = UploadModelForm()
    if request.method == 'POST':
        dataset_used = request.form.get('dataset_used')
        if form.upload_model.data:
            filename = secure_filename(form.upload_model.data.filename)
            session['uploaded_model'] = filename
            form.upload_model.data.save(app.config['UPLOAD_FOLDER'] + 'models/' + filename)
            predictor_loaded = ImagePredictor.load('./static/models/' + filename)
            model_summary = predictor_loaded.fit_summary()
            train_acc = round(model_summary['train_acc'] * 100, 2)
            val_acc = round(model_summary['valid_acc'] * 100, 2)
            total_time = round(model_summary['total_time'] / 60, 2)
            model_used = model_summary['best_config']['img_cls']['model']
            new_model = Models(username=current_user.username, model_name=filename, dataset_used=dataset_used,
                               train_acc=train_acc,
                               val_acc=val_acc, total_time=total_time, model_used=model_used)
            db.session.add(new_model)
            db.session.commit()
            flash('Model Uploaded Successfully!')
        return render_template('upload_model_page.html', form=form, train_acc=train_acc, val_acc=val_acc,
                               total_time=total_time, model_used=model_used)
    return render_template('upload_model_page.html', form=form)

# This route is used to view the leaderboard page.
@app.route("/view_leaderboard", methods=['GET', 'POST'])
@login_required
def viewLeaderboard():
    form = SelectDatasetForm()
    if request.method == 'POST':
        selected_dataset = request.form.get('select_dataset')
        leaderboard = Models.query.filter_by(dataset_used=selected_dataset).order_by(Models.train_acc.desc()).all()
        return render_template('view_leaderboard.html', form=form, leaderboard=leaderboard)
    return render_template('view_leaderboard.html', form=form)

# This route is used to view the statistics of a model.
@app.route("/view_model_statistics", methods=['GET', 'POST'])
@login_required
def viewModelStatistics():
    form = SelectModelForm()
    models = Models.query.order_by(Models.model_name.asc()).all()
    if request.method == 'POST':
        predictor_loaded = ImagePredictor.load('./static/models/' + request.form.get('select_model'))
        model_summary = predictor_loaded.fit_summary()
        train_acc = round(model_summary['train_acc'] * 100, 2)
        val_acc = round(model_summary['valid_acc'] * 100, 2)
        total_time = round(model_summary['total_time'] / 60, 2)
        model_used = model_summary['best_config']['img_cls']['model']
        return render_template("view_model_statistics.html", form=form,
                               train_acc=train_acc, val_acc=val_acc, total_time=total_time, model_used=model_used,
                               dir_list=models)
    return render_template("view_model_statistics.html", form=form, dir_list=models)

# This function is used to check if the image uploaded is valid.
def allowed_file_images(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_IMAGES

# This route is used to validate if the model uploaded is valid.
def allowed_file_models(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_MODELS

# This route is used to select the dataset to train a new model on.
@app.route("/select_dataset_to_train", methods=['GET', 'POST'])
@login_required
def selectDatasetToTrain():
    if (current_user.user_type == 'Radiologist'):
        form = SelectDatasetAndModelForm()
        if request.method == 'POST':
            session['select_dataset'] = request.form.get('select_dataset')
            session['model_name'] = request.form.get('model_name')
            return redirect(url_for('run'))
        return render_template('select_dataset_to_train.html', form=form)
    else:
        return redirect(url_for('logout'))

# This route is used to run the training of the model.
@app.route("/run", methods=['GET', 'POST'])
@login_required
def run():
    if (current_user.user_type == 'Radiologist'):
        selected_dataset = session['select_dataset']
        model_name = session['model_name']
        return render_template("run_training.html", selected_dataset=selected_dataset,
                               model_name=model_name)
    else:
        return redirect(url_for('logout'))

# Setting the window size of the console
def set_winsize(fd, row, col, xpix=0, ypix=0):
    winsize = struct.pack("HHHH", row, col, xpix, ypix)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)

# Function for forwarding the console output.
def read_and_forward_pty_output():
    max_read_bytes = 1024 * 20
    while True:
        socket_io.sleep(0.01)
        if app.config["fd"]:
            timeout_sec = 0
            (data_ready, _, _) = select.select([app.config["fd"]], [], [], timeout_sec)
            if data_ready:
                output = os.read(app.config["fd"], max_read_bytes).decode()
                socket_io.emit("pty-output", {"output": output}, namespace="/pty")

# Function for console input.
@socket_io.on("pty-input", namespace="/pty")
def pty_input(data):
    if app.config["fd"]:
        os.write(app.config["fd"], data["input"].encode())

# Function for resizing the console.
@socket_io.on("resize", namespace="/pty")
def resize(data):
    if app.config["fd"]:
        set_winsize(app.config["fd"], data["rows"], data["cols"])

# Function for connecting the console to the frontend.
@socket_io.on("connect", namespace="/pty")
def connect():
    if app.config["child_pid"]: # When the child pty is already running
        return

    (child_pid, fd) = pty.fork() # Ensure that the child pid can read.
    if child_pid == 0:  # Child pty process
        # Anything printed here is showcased on the pty
        subprocess.run(app.config["cmd"])
    else:  # Parent pty process
        # This stores the child fd and pid.
        app.config["fd"] = fd
        app.config["child_pid"] = child_pid
        set_winsize(fd, 50, 50) # Setting the size of the window.
        cmd = " ".join(shlex.quote(c) for c in app.config["cmd"])
        print("child pid is", child_pid)
        print(f"Starting background task with command `{cmd}` to continuously read ""and forward pty output to client")
        socket_io.start_background_task(target=read_and_forward_pty_output)
        print("Started pty background task.")

# The following is used to initialize the terminal for the application.
parser = argparse.ArgumentParser(description="Terminal for AutoMID",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# The following are the arguments that can be passed.
parser.add_argument("-p", "--port", default=5000, help="The port which AutoMID runs on.") # Port number
parser.add_argument("--host", default="127.0.0.1", help="The host for AutoMID.", ) # Host address
parser.add_argument("--debug", action="store_true", help="Allow the user to debug AutoMID server.") # enable/disable debugging
parser.add_argument("--version", action="store_true", help="Print the version.") # Printing the app version.
parser.add_argument("--command", default="bash", help="The command to run.") # Selecting the command to run.
parser.add_argument("--cmd-args", default="",
                    help="The arguments to add to the command (i.e. --cmd-args='arg1 arg2 --flag').", )
args = parser.parse_args() # Parsing the arguments.
if args.version:
    print(__version__)
    exit(0)
print(f"Serving AutoMID on http://127.0.0.1:{args.port}")
app.config["cmd"] = [args.command] + shlex.split(args.cmd_args)
socket_io.run(app, debug=args.debug, port=args.port, host=args.host)
