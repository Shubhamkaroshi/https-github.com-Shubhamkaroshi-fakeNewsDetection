from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time

app = Flask(__name__)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_models = {
    'checkbox_a': ('Passive-Aggressive', pickle.load(open('model1.pkl', 'rb'))),
    'checkbox_b': ('Logistic Regression', pickle.load(open('model2.pkl', 'rb'))),
    'checkbox_c': ('Decision Tree', pickle.load(open('model3.pkl', 'rb'))),
    'checkbox_d': ('Gradient Boosting', pickle.load(open('model4.pkl', 'rb'))),
    'checkbox_e': ('Random Forest', pickle.load(open('model5.pkl', 'rb'))),
}
dataframe = pd.read_csv('news.csv')
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

progress = 0


def fake_news_det(news, model, model_name, total_models):
    global progress
    progress_increment = 100 / (total_models * 10)

    xv_train = vectorizer.fit_transform(x_train)
    xv_test = vectorizer.transform(x_test)
    input_data = [news]
    vectorized_input_data = vectorizer.transform(input_data)

    for _ in tqdm(range(10)):
        time.sleep(0.1)
        progress += progress_increment

    prediction = model.predict(vectorized_input_data)[0]
    accuracy = accuracy_score(y_test, model.predict(xv_test))

    return prediction, model_name, accuracy


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global progress
    progress = 0
    message = request.form['message']
    selected_models = [key for key in request.form if key.startswith('checkbox')]
    total_models = len(selected_models)
    results = []

    for key in selected_models:
        prediction, model_name, accuracy = fake_news_det(message, loaded_models[key][1], loaded_models[key][0],
                                                         total_models)
        results.append({'model': model_name, 'prediction': prediction, 'accuracy': accuracy})

    return render_template('results.html', results=results)


@app.route('/progress')
def get_progress():
    global progress
    return jsonify({'progress': progress})


if __name__ == '__main__':
    app.run(debug=True)
