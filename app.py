from os import read
from flask import Flask, request, render_template, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask import Flask, request, render_template
import json
import pandas as pd
import numpy as np
import neattext.functions as nfx
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from dashboard import getvaluecounts, getlevelcount, getsubjectsperlevel, yearwiseprofit


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = 'mysecretkey'  # Change to a secure key

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# 🔹 User Model (For Authentication)

class Progress(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    status = db.Column(db.String(50), nullable=False)
    time_spent = db.Column(db.Integer, nullable=True)  # Time spent on the course in hours

class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(500), nullable=False)
    recommended = db.Column(db.Boolean, default=False)

class Badge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(200), nullable=False)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("User already exists. Please log in.", "danger")
            return redirect(url_for('login'))

        new_user = User(email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash("Login successful!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials. Please try again.", "danger")

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))
def getcosinemat(df):

    countvect = CountVectorizer()
    cvmat = countvect.fit_transform(df['Clean_title'])
    return cvmat

# getting the title which doesn't contain stopwords and all which we removed with the help of nfx


def getcleantitle(df):

    df['Clean_title'] = df['course_title'].apply(nfx.remove_stopwords)

    df['Clean_title'] = df['Clean_title'].apply(nfx.remove_special_characters)

    return df


def cosinesimmat(cv_mat):

    return cosine_similarity(cv_mat)


def readdata():

    df = pd.read_csv('UdemyCleanedTitle.csv')
    return df

# this is the main recommendation logic for a particular title which is choosen


def recommend_course(df, title, cosine_mat, numrec):

    course_index = pd.Series(
        df.index, index=df['course_title']).drop_duplicates()

    index = course_index[title]

    scores = list(enumerate(cosine_mat[index]))

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    selected_course_index = [i[0] for i in sorted_scores[1:]]

    selected_course_score = [i[1] for i in sorted_scores[1:]]

    rec_df = df.iloc[selected_course_index]

    rec_df['Similarity_Score'] = selected_course_score

    final_recommended_courses = rec_df[[
        'course_title', 'Similarity_Score', 'url', 'price', 'num_subscribers']]

    return final_recommended_courses.head(numrec)

# this will be called when a part of the title is used,not the complete title!


def searchterm(term, df):
    result_df = df[df['course_title'].str.contains(term)]
    top6 = result_df.sort_values(by='num_subscribers', ascending=False).head(6)
    return top6


# extract features from the recommended dataframe
@app.route('/')
def home():
    courses = Course.query.all()
    return render_template('index.html', courses=courses)
def extractfeatures(recdf):
    course_id=list(recdf['course_id'])
    course_url = list(recdf['url'])
    course_title = list(recdf['course_title'])
    course_price = list(recdf['price'])

    return course_url, course_title, course_price


@app.route('/dashboard', methods=['GET', 'POST'])
def hello_world():

    if request.method == 'POST':

        my_dict = request.form
        titlename = my_dict['course']
        print(titlename)
        try:
            df = readdata()
            df = getcleantitle(df)
            cvmat = getcosinemat(df)

            num_rec = 6
            cosine_mat = cosinesimmat(cvmat)

            recdf = recommend_course(df, titlename,
                                     cosine_mat, num_rec)

            course_url, course_title, course_price = extractfeatures(recdf)

            # print(len(extractimages(course_url[1])))

            dictmap = dict(zip(course_title, course_url))

            if len(dictmap) != 0:
                return render_template('dashboard.html', coursemap=dictmap, coursename=titlename, showtitle=True)

            else:
                return render_template('dashboard.html', showerror=True, coursename=titlename)

        except:

            resultdf = searchterm(titlename, df)
            if resultdf.shape[0] > 6:
                resultdf = resultdf.head(6)
                course_url, course_title, course_price = extractfeatures(
                    resultdf)
                coursemap = dict(zip(course_title, course_url))
                if len(coursemap) != 0:
                    return render_template('dashboard.html', coursemap=coursemap, coursename=titlename, showtitle=True)

                else:
                    return render_template('dashboard.html', showerror=True, coursename=titlename)

            else:
                course_url, course_title, course_price = extractfeatures(
                    resultdf)
                coursemap = dict(zip(course_title, course_url))
                if len(coursemap) != 0:
                    return render_template('dashboard.html', coursemap=coursemap, coursename=titlename, showtitle=True)

                else:
                    return render_template('dashboard.html', showerror=True, coursename=titlename)

    return render_template('dashboard.html')

@app.route('/progress/<int:course_id>/<status>', methods=['POST'])
@login_required
def update_progress(course_id, status):
    progress = Progress.query.filter_by(user_id=current_user.id, course_id=course_id).first()
    if not progress:
        progress = Progress(user_id=current_user.id, course_id=course_id, status=status)
        db.session.add(progress)
    else:
        progress.status = status
    db.session.commit()
    return jsonify({'message': 'Progress updated successfully'})

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    user_progress = Progress.query.filter_by(user_id=current_user.id).all()
    user_badges = Badge.query.filter_by(user_id=current_user.id).all()
    recommended_courses = Course.query.filter_by(recommended=True).all()
    
    completed_courses = Progress.query.filter_by(user_id=current_user.id, status='Completed').count()
    time_spent = sum([course.time_spent for course in user_progress if course.status == 'Completed'])
    
    return render_template('dashboard.html', 
                           progress=user_progress, 
                           recommended_courses=recommended_courses,
                           user_badges=user_badges,
                           completed_courses=completed_courses,
                           time_spent=time_spent)
if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, render_template, request, redirect, url_for, session, jsonify
# from flask_sqlalchemy import SQLAlchemy
# from flask_bcrypt import Bcrypt
# from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

# app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///courses.db'
# app.config['SECRET_KEY'] = 'your_secret_key'
# db = SQLAlchemy(app)
# bcrypt = Bcrypt(app)
# login_manager = LoginManager(app)
# login_manager.login_view = 'login'

# class User(UserMixin, db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(150), unique=True, nullable=False)
#     password = db.Column(db.String(150), nullable=False)
#     progress = db.relationship('Progress', backref='user', lazy=True)

# class Course(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     title = db.Column(db.String(200), nullable=False)
#     description = db.Column(db.Text, nullable=False)

# class Progress(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
#     status = db.Column(db.String(20), default='Not Started')  # Not Started, In Progress, Completed

# @login_manager.user_loader
# def load_user(user_id):
#     return User.query.get(int(user_id))

# @app.route('/')
# def home():
#     courses = Course.query.all()
#     return render_template('index.html', courses=courses)

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         user = User.query.filter_by(username=username).first()
#         if user and bcrypt.check_password_hash(user.password, password):
#             login_user(user)
#             return redirect(url_for('dashboard'))
#     return render_template('login.html')

# @app.route('/logout')
# @login_required
# def logout():
#     logout_user()
#     return redirect(url_for('home'))

# @app.route('/dashboard')
# @login_required
# def dashboard():
#     user_progress = Progress.query.filter_by(user_id=current_user.id).all()
#     return render_template('dashboard.html', progress=user_progress)

# @app.route('/progress/<int:course_id>/<status>', methods=['POST'])
# @login_required
# def update_progress(course_id, status):
#     progress = Progress.query.filter_by(user_id=current_user.id, course_id=course_id).first()
#     if not progress:
#         progress = Progress(user_id=current_user.id, course_id=course_id, status=status)
#         db.session.add(progress)
#     else:
#         progress.status = status
#     db.session.commit()
#     return jsonify({'message': 'Progress updated successfully'})

# if __name__ == '__main__':
#     db.create_all()
#     app.run(debug=True)
