from flask import Flask, render_template,request
import re
from nltk.stem import WordNetLemmatizer
import pickle

wo = WordNetLemmatizer()

app = Flask(__name__)

def preprocess(data):
    #preprocess
    a = re.sub('[^a-zA-Z]',' ',data)
    a = a.lower()
    a = a.split()
    a = [wo.lemmatize(word) for word in a ]
    a = ' '.join(a)
    return a


tfidf_vectorizer = pickle.load(open('vectorizer.pkl','rb'))
model =  pickle.load(open('prediction.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods= ['POST'])
def predict():
    msg = request.form['mood_pred']
    a = preprocess(msg)

    # example_counts = vectorizer.transform( [a] )
    # prediction = mnb.predict( example_counts )
    # prediction[0]

    result = model.predict(tfidf_vectorizer.transform([a]))[0]
    return render_template('index.html',pred = "You are--{}".format(result))



if __name__ == '__main__':
    app.run(debug=True)