import os
import keras
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

from keras.models import load_model
from keras.backend import set_session
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
print("Loading model")
global sess
sess = tf.Session()
set_session(sess)
global model
model = load_model('my_Odia37_model.h5')
global graph
graph = tf.get_default_graph()

@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')

@app.route('/prediction/<filename>')
def prediction(filename):
    #Step 1
    my_image = plt.imread(os.path.join('uploads', filename))
    #Step 2
    my_image_re = resize(my_image, (64,64,3))
    
    #Step 3
    with graph.as_default():
      set_session(sess)
      probabilities = model.predict(np.array( [my_image_re,] ))[0,:]
      print(probabilities)
#Step 4
      number_to_class = ['ଅ','ଆ','ଇ','ଔ','କ','କ୍ଷ','ଖ','ଚ','ଛ','ଜ','ଝ','ଞ','ଟ','ଠ','ଡ','ଥ','ଦ','ଧ','ନ','ପ','ଫ','ବ','ଭ','ମ','ଯ','ୟ','ର','ଲ','ଳ','ୱ','ଶ','ଷ','ସ','ହ']
      index = np.argsort(probabilities)
      predictions = {
        "class1":number_to_class[index[36]],
        "class2":number_to_class[index[35]],
        "class3":number_to_class[index[34]],
        "prob1":probabilities[index[36]],
        "prob2":probabilities[index[35]],
        "prob3":probabilities[index[34]],
      }
#Step 5
    return render_template('predict.html', predictions=predictions)

app.run(host='0.0.0.0', port=80)