

from flask import Flask,render_template,url_for,request
from sklearn.externals import joblib
import pandas as pd
import pickle
import numpy as np

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/predict',methods=['POST'])
def predict():
	#load model
	model=open('data/project_web_model.pickle','rb')
	model1=joblib.load(model)

	if request.method=='POST':
		bedrooms=request.form['bedrooms']
		bathrooms=request.form['bathrooms']
		floors=request.form['floors']
		a=[bedrooms,bathrooms,floors]
		data=np.array(a).astype(float).reshape(1,-1)
		my_pred=model1.predict(data)
		b=int(my_pred[0])
		print(b)
		return render_template('results.html',prediction=b)

        #return render_template('results.html',prediction=my_pred)


if __name__=='__main__':
    app.run(debug=True)