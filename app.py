# import matplotlib
# matplotlib.use('Agg')  # Use the Agg backend for Matplotlib

# from flask import Flask, render_template, request, redirect, url_for, session, flash
# import os
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import matplotlib.pyplot as plt
# import numpy as np
# import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import firebase_admin
# from firebase_admin import credentials, firestore
# import pyrebase
# import openai


# nltk.download('vader_lexicon')

# app = Flask(__name__)
# app.secret_key = "your_secret_key_here"

# # OpenAI API key
# openai.api_key = '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'

# # Call OpenAI API
# def get_writing_suggestions(text, emotion, limit=5):
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": f"Given the text:\n\n{text}\n\nand the emotion: {emotion}, provide up to {limit} writing improvement suggestions:"}
#         ],
#         max_tokens=200,
#         temperature=0.7
#     )
#     suggestions = response.choices[0].message['content'].strip()
#     # Split suggestions by new line and limit them
#     suggestions_list = suggestions.split('\n')
#     limited_suggestions = suggestions_list[:limit]
#     return "\n".join(limited_suggestions)


# # Initialize Firebase Admin SDK
# cred = credentials.Certificate('serviceAccountKey.json')
# firebase_admin.initialize_app(cred)

# # Initialize Firestore
# db = firestore.client()

# # Firebase configuration for Pyrebase
# firebase_config = {
#     "apiKey": "AIzaSyDckIo2fmSxD6uCCn4V-5GEXzdiP6Y1n6Q",
#     "authDomain": "penman-4a01b.firebaseapp.com",
#     "databaseURL": "https://penman-4a01b.firebaseio.com",
#     "projectId": "penman-4a01b",
#     "storageBucket": "penman-4a01b.appspot.com",
#     "messagingSenderId": "891859428295",
#     "appId": "1:891859428295:web:d8beb4c69da686c2482fe2",
#     "measurementId": "G-X10M4QTF2V"
# }

# firebase = pyrebase.initialize_app(firebase_config)
# auth = firebase.auth()

# # Ensure upload folder exists
# UPLOAD_FOLDER = 'uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Load the model and tokenizer
# model_path = 'emotion_model'
# tokenizer_path = 'emotion_model'
# model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=6)
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# # Define emotion labels
# emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# # Check if using GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# def detect_emotions_bert(text):
#     input_encoded = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    
#     with torch.no_grad():
#         outputs = model(**input_encoded)
    
#     logits = outputs.logits
#     probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
    
#     # Return probabilities for all emotions
#     return dict(zip(emotion_labels, probabilities))

# def detect_sentiment_vader(text):
#     analyzer = SentimentIntensityAnalyzer()
#     sentiment_scores = analyzer.polarity_scores(text)
#     if sentiment_scores['compound'] >= 0.05:
#         sentiment = "Positive"
#     elif sentiment_scores['compound'] <= -0.05:
#         sentiment = "Negative"
#     else:
#         sentiment = "Neutral"
#     return sentiment

# def plot_emotions(emotions):
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     emotions_labels = list(emotions.keys())
#     intensities = list(emotions.values())
    
#     x = np.arange(len(emotions_labels))
#     width = 0.3
    
#     bars = ax.bar(x, intensities, width, color='skyblue')
    
#     ax.set_xlabel('Emotions')
#     ax.set_ylabel('Intensity (0-1)')
#     ax.set_title('Emotion Analysis')
#     ax.set_xticks(x)
#     ax.set_xticklabels(emotions_labels, rotation=45)
#     ax.legend(['Intensity'])
    
#     # Add values on top of the bars
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate('{}'.format(round(height * 100, 2)),
#                     xy=(bar.get_x() + bar.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
    
#     graph_filename = os.path.join('static', 'emotion_analysis.png')
#     plt.savefig(graph_filename)
#     plt.close()
    
#     return graph_filename

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/home')
# def home():
#     if 'user' in session:
#         return render_template('home.html', username=session['user']['username'])
#     else:
#         flash("Please log in to access the home page.", "danger")
#         return redirect(url_for('login'))

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form['email']
#         password = request.form['password']
        
#         try:
#             user = auth.sign_in_with_email_and_password(email, password)
#             uid = user['localId']
            
#             # Fetch the username from Firestore
#             user_doc = db.collection('users').document(uid).get()
#             username = user_doc.to_dict().get('username')
            
#             session['user'] = {'email': email, 'username': username}
#             flash("Login successful!", "success")
#             return redirect(url_for('home'))
#         except Exception as e:
#             flash(f"Login failed: {str(e)}", "danger")
#             return render_template('login.html')
#     return render_template('login.html')

# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         username = request.form['username']
#         email = request.form['email']
#         password = request.form['password']
        
#         try:
#             user = auth.create_user_with_email_and_password(email, password)
#             uid = user['localId']
            
#             # Save the username in Firestore
#             db.collection('users').document(uid).set({
#                 'username': username,
#                 'email': email
#             })
            
#             flash("Registration successful! Please log in.", "success")
#             return redirect(url_for('login'))
#         except Exception as e:
#             flash(f"Registration failed: {str(e)}", "danger")
#             return render_template('register.html')
#     return render_template('register.html')


# @app.route('/logout')
# def logout():
#     session.pop('user', None)
#     return redirect(url_for('index'))

# # upload route
# @app.route('/upload', methods=['GET', 'POST'])
# def upload():
#     if 'user' not in session:
#         return redirect(url_for('login'))
    
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#             file.save(filename)
#             with open(filename, 'r', encoding='utf-8') as f:
#                 text = f.read()
#             emotions = detect_emotions_bert(text)
#             sentiment = detect_sentiment_vader(text)
#             graph_filename = plot_emotions(emotions)
#             highest_emotion = max(emotions, key=emotions.get)
#             highest_emotion_percentage = emotions[highest_emotion] * 100
            
#             # Get writing improvement suggestions
#             suggestions = get_writing_suggestions(text, highest_emotion)
            
#             # Store suggestions and original text in session
#             session['suggestions'] = suggestions
#             session['original_text'] = text
#             session['highest_emotion'] = highest_emotion
            
#             return render_template('result.html', 
#                                    emotions=emotions, 
#                                    graph_filename=graph_filename, 
#                                    highest_emotion=highest_emotion, 
#                                    highest_emotion_percentage=highest_emotion_percentage,
#                                    sentiment=sentiment)
    
#     return render_template('upload.html', username=session['user']['username'])

# # suggestions route
# @app.route('/suggestions', methods=['GET', 'POST'])
# def suggestions():
#     if 'suggestions' in session:
#         if request.method == 'POST':
#             return render_template('suggestions.html', 
#                                    suggestions=[s for s in session['suggestions'].split('\n') if s.strip()], 
#                                    original_text=session['original_text'], 
#                                    highest_emotion=session['highest_emotion'])
#         return render_template('suggestions.html', 
#                                suggestions=[s for s in session['suggestions'].split('\n') if s.strip()], 
#                                original_text=session['original_text'], 
#                                highest_emotion=session['highest_emotion'])
#     else:
#         return redirect(url_for('upload'))

# #dashboard route
# @app.route('/dashboard')
# def dashboard():
#     if 'user' in session:
#         return render_template('dashboard.html', username=session['user']['username'])
#     else:
#         return redirect(url_for('login'))


# if __name__ == '__main__':
#     app.run(debug=True)


import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for Matplotlib

from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import firebase_admin
from firebase_admin import credentials, firestore
import pyrebase
import openai
import uuid


nltk.download('vader_lexicon')

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# OpenAI API key
openai.api_key = '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'

# Call OpenAI API
def get_writing_suggestions(text, emotion, limit=5):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Given the text:\n\n{text}\n\nand the emotion: {emotion}, provide up to {limit} writing improvement suggestions:"}
        ],
        max_tokens=200,
        temperature=0.7
    )
    suggestions = response.choices[0].message['content'].strip()
    # Split suggestions by new line and limit them
    suggestions_list = suggestions.split('\n')
    limited_suggestions = suggestions_list[:limit]
    return "\n".join(limited_suggestions)


# Initialize Firebase Admin SDK
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()

# Firebase configuration for Pyrebase
firebase_config = {
    "apiKey": "AIzaSyDckIo2fmSxD6uCCn4V-5GEXzdiP6Y1n6Q",
    "authDomain": "penman-4a01b.firebaseapp.com",
    "databaseURL": "https://penman-4a01b.firebaseio.com",
    "projectId": "penman-4a01b",
    "storageBucket": "penman-4a01b.appspot.com",
    "messagingSenderId": "891859428295",
    "appId": "1:891859428295:web:d8beb4c69da686c2482fe2",
    "measurementId": "G-X10M4QTF2V"
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model and tokenizer
model_path = 'emotion_model'
tokenizer_path = 'emotion_model'
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=6)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Define emotion labels
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Check if using GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# emotion dtection function
def detect_emotions_bert(text):
    input_encoded = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**input_encoded)
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
    
    # Return probabilities for all emotions
    return dict(zip(emotion_labels, probabilities))

# detect sentiment
def detect_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    if sentiment_scores['compound'] >= 0.05:
        sentiment = "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment


# plot graph
def plot_emotions(emotions):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    emotions_labels = list(emotions.keys())
    intensities = list(emotions.values())
    
    x = np.arange(len(emotions_labels))
    width = 0.3
    
    bars = ax.bar(x, intensities, width, color='skyblue')
    
    ax.set_xlabel('Emotions')
    ax.set_ylabel('Intensity (0-1)')
    ax.set_title('Emotion Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(emotions_labels, rotation=45)
    ax.legend(['Intensity'])
    
    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(round(height * 100, 2)),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    graph_filename = os.path.join('static', 'emotion_analysis.png')
    plt.savefig(graph_filename)
    plt.close()
    
    return graph_filename

# routes
# index route
@app.route('/')
def index():
    return render_template('index.html')

# Home route
@app.route('/home')
def home():
    if 'user' in session:
        return render_template('home.html', username=session['user']['username'])
    else:
        flash("Please log in to access the home page.", "danger")
        return redirect(url_for('login'))
    
# Login route    
@app.route('/login', methods=['GET', 'POST'])
def login():
    error_message = None  # Initialize error message

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            user = auth.sign_in_with_email_and_password(email, password)
            uid = user['localId']
            
            # Fetch the username from Firestore
            user_doc = db.collection('users').document(uid).get()
            username = user_doc.to_dict().get('username')
            
            session['user'] = {'email': email, 'username': username, 'uid': uid}
            return redirect(url_for('home'))
        except Exception as e:
            error_message = "Login failed: Invalid email or password."

    return render_template('login.html', error_message=error_message)


# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    error_message = None

    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if username is already taken
        users_ref = db.collection('users')
        username_query = users_ref.where('username', '==', username).get()
        
        if username_query:
            error_message = "Username is already taken."
        elif len(password) < 6:  # Check for password length (add more rules as needed)
            error_message = "Password must be at least 6 characters long."
        else:
            try:
                user = auth.create_user_with_email_and_password(email, password)
                uid = user['localId']
                
                # Save the username and email in Firestore
                db.collection('users').document(uid).set({
                    'username': username,
                    'email': email
                })
                
                flash("Registration successful! Please log in.", "success")
                return redirect(url_for('login'))
            except Exception as e:
                if "EMAIL_EXISTS" in str(e):
                    error_message = "Email is already registered."
                elif "INVALID_EMAIL" in str(e):
                    error_message = "Invalid email address."
                else:
                    error_message = f"Registration failed: {str(e)}"
    
    return render_template('register.html', error_message=error_message)


# Logout Route
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

# upload route
# @app.route('/upload', methods=['GET', 'POST'])
# def upload():
#     if 'user' not in session:
#         return redirect(url_for('login'))
    
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             filename = file.filename  # Get the original file name
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(file_path)
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 text = f.read()
#             emotions = detect_emotions_bert(text)
#             sentiment = detect_sentiment_vader(text)
#             graph_filename = plot_emotions(emotions)
#             highest_emotion = max(emotions, key=emotions.get)
#             highest_emotion_percentage = emotions[highest_emotion] * 100
            
#             # Convert numpy.float32 to Python float
#             emotions = {k: float(v) if isinstance(v, np.float32) else v for k, v in emotions.items()}
#             highest_emotion_percentage = float(highest_emotion_percentage) if isinstance(highest_emotion_percentage, np.float32) else highest_emotion_percentage
            
#             # Get writing improvement suggestions
#             suggestions = get_writing_suggestions(text, highest_emotion)
            
#             # Store suggestions and original text in session
#             session['suggestions'] = suggestions
#             session['original_text'] = text
#             session['highest_emotion'] = highest_emotion
            
#             # Save to Firestore
#             try:
#                 uid = session['user']['uid']
#                 upload_id = str(uuid.uuid4())  # Generate a unique ID for the upload
#                 print(f"Saving upload to Firestore for user {uid} with upload ID {upload_id}")  # Debugging
#                 db.collection('users').document(uid).collection('uploads').document(upload_id).set({
#                     'filename': filename,  # Save the original file name
#                     'text': text,
#                     'emotions': emotions,
#                     'highest_emotion': highest_emotion,
#                     'highest_emotion_percentage': highest_emotion_percentage,
#                     'suggestions': suggestions
#                 })
#                 print("Upload saved successfully!")  # Debugging
#             except Exception as e:
#                 print(f"Error: {str(e)}")  # Debugging
#                 flash(f"Failed to save upload to Firestore: {str(e)}", "danger")
            
#             return render_template('result.html', 
#                                    emotions=emotions, 
#                                    graph_filename=graph_filename, 
#                                    highest_emotion=highest_emotion, 
#                                    highest_emotion_percentage=highest_emotion_percentage,
#                                    sentiment=sentiment)
    
#     return render_template('upload.html', username=session['user']['username'])

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))

    allowed_extensions = {'txt'}  # Allow only text files

    if request.method == 'POST':
        file = request.files.get('file')  # Use .get() to handle cases where 'file' might not be in request.files

        if file:
            filename = file.filename
            file_ext = filename.split('.')[-1].lower()

            # Check if the file has an allowed extension
            if file_ext not in allowed_extensions:
                error_message = "Invalid file type. Please upload a .txt file."
                return render_template('upload.html', username=session['user']['username'], error_message=error_message)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process the file as before
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            emotions = detect_emotions_bert(text)
            sentiment = detect_sentiment_vader(text)
            graph_filename = plot_emotions(emotions)
            highest_emotion = max(emotions, key=emotions.get)
            highest_emotion_percentage = emotions[highest_emotion] * 100

            # Convert emotions and highest_emotion_percentage to float if they are np.float32
            emotions = {k: float(v) if isinstance(v, np.float32) else v for k, v in emotions.items()}
            highest_emotion_percentage = float(highest_emotion_percentage) if isinstance(highest_emotion_percentage, np.float32) else highest_emotion_percentage

            suggestions = get_writing_suggestions(text, highest_emotion)

            session['suggestions'] = suggestions
            session['original_text'] = text
            session['highest_emotion'] = highest_emotion

            try:
                uid = session['user']['uid']
                upload_id = str(uuid.uuid4())
                db.collection('users').document(uid).collection('uploads').document(upload_id).set({
                    'filename': filename,
                    'text': text,
                    'emotions': emotions,
                    'highest_emotion': highest_emotion,
                    'highest_emotion_percentage': highest_emotion_percentage,
                    'suggestions': suggestions
                })
            except Exception as e:
                error_message = f"Failed to save upload to Firestore: {str(e)}"
                return render_template('upload.html', username=session['user']['username'], error_message=error_message)

            return render_template('result.html',
                                   emotions=emotions,
                                   graph_filename=graph_filename,
                                   highest_emotion=highest_emotion,
                                   highest_emotion_percentage=highest_emotion_percentage,
                                   sentiment=sentiment)

    return render_template('upload.html', username=session['user']['username'])


# suggestions route
@app.route('/suggestions', methods=['GET', 'POST'])
def suggestions():
    if 'suggestions' in session:
        if request.method == 'POST':
            return render_template('suggestions.html', 
                                   suggestions=[s for s in session['suggestions'].split('\n') if s.strip()], 
                                   original_text=session['original_text'], 
                                   highest_emotion=session['highest_emotion'])
        return render_template('suggestions.html', 
                               suggestions=[s for s in session['suggestions'].split('\n') if s.strip()], 
                               original_text=session['original_text'], 
                               highest_emotion=session['highest_emotion'])
    else:
        return redirect(url_for('upload'))
    
# #dashboard route
# @app.route('/dashboard')
# def dashboard():
#     if 'user' in session:
#         return render_template('dashboard.html', username=session['user']['username'])
#     else:
#         return redirect(url_for('login'))

#dashboard route
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    uid = session['user']['uid']
    user_files = []

    try:
        # Fetch files from Firestore
        uploads_ref = db.collection('users').document(uid).collection('uploads')
        docs = uploads_ref.stream()
        for doc in docs:
            file_data = doc.to_dict()
            file_data['id'] = doc.id
            file_data['name'] = doc.id  # Assuming the filename is stored as the document ID
            user_files.append(file_data)
    except Exception as e:
        print(f"Error fetching files: {str(e)}")
        flash("Could not retrieve files.", "danger")
    
    return render_template('dashboard.html', username=session['user']['username'], uploaded_files=user_files)


#view files route
@app.route('/view_file/<file_id>', methods=['GET'])
def view_file(file_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    try:
        uid = session['user']['uid']
        file_doc = db.collection('users').document(uid).collection('uploads').document(file_id).get()
        if file_doc.exists:
            file_data = file_doc.to_dict()
            return render_template('file_details.html', file=file_data)
        else:
            flash("File not found.", "danger")
            return redirect(url_for('dashboard'))
    except Exception as e:
        print(f"Error: {str(e)}")  # Debugging
        flash(f"Failed to retrieve file: {str(e)}", "danger")
        return redirect(url_for('dashboard'))


#delete files route
@app.route('/delete_file/<file_id>', methods=['POST'])
def delete_file(file_id):
    if 'user' not in session:
        return redirect(url_for('login'))

    uid = session['user']['uid']
    try:
        # Delete the file from Firestore
        file_ref = db.collection('users').document(uid).collection('uploads').document(file_id)
        file_ref.delete()
        flash("File deleted successfully.", "success")
    except Exception as e:
        print(f"Error deleting file: {str(e)}")
        flash("Could not delete file.", "danger")
    
    return redirect(url_for('dashboard'))
if __name__ == '__main__':
    app.run(debug=True)
