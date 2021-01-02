
# Importing the libraries
import spacy
import requests
import tensorflow as tf
from flask import Flask, jsonify, request, render_template
from model import data_utils_2
from model import data_utils_1
from model import data_preprocessing
import importlib
from model import seq2seq_wrapper
import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning)


importlib.reload(seq2seq_wrapper)
nlp = spacy.load("en_core_web_sm")


app = Flask(__name__)
# graph = tf.get_default_graph()
# Importing the dataset
metadata, idx_q, idx_a = data_preprocessing.load_data(PATH='./model/')

# Splitting the dataset into the Training set and the Test set
(trainX, trainY), (testX, testY), (validX,
                                   validY) = data_utils_1.split_dataset(idx_q, idx_a)

# Embedding
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 16
vocab_twit = metadata['idx2w']
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024
idx2w, w2idx, limit = data_utils_2.get_metadata()


print('Preparing model............')

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                                yseq_len=yseq_len,
                                xvocab_size=xvocab_size,
                                yvocab_size=yvocab_size,
                                ckpt_path='./model/weights',
                                emb_dim=emb_dim,
                                num_layers=3)


session = model.restore_last_session()

# Getting the ChatBot predicted answer


def respond(question):
    encoded_question = data_utils_2.encode(question, w2idx, limit['maxq'])
    answer = model.predict(session, encoded_question)
    return data_utils_2.decode(answer[0], idx2w)

# Setting up the chat
# while True :
#   question = input("You: ")
#   answer = respond(question)
#   print ("ChatBot: "+answer)


def weather_forecast(url):
    res = requests.get(url).json()
    return res


@app.route('/', methods=['GET'])
def home():
    return render_template("home.html")


@app.route('/call', methods=['POST'])
def get():
    ques = request.form['question']
    if 'weather' in ques.lower():
        doc = nlp(ques)
        for ent in doc.ents:
            if ent.label_ == "GPE":
                try:
                    URL = "http://api.weatherapi.com/v1/current.json?key="+os.environ.get('API_WEATHER_KEY')+"&q=" + ent.text
                    res = weather_forecast(URL)
                    text = "Currently " + ent.text + " is " + \
                        res['current']['condition']['text'] + ' and temperature is ' + \
                        str(res['current']['temp_c']) + " degree "
                    return text
                except Exception as e:
                    print(e)
                    return "Sorry! I'm not able tell that."
            else:
                return "Sorry! I have no answer for this."
    else:
        answer = respond(ques)
        return answer


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
