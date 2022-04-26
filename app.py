# -*- coding: utf-8 -*-  
from flask import Flask,request, render_template
#from requests import request
import model as m 
#import test as m 

app = Flask(__name__)


@app.route("/", methods = ["GET","POST"])
def tags():
    result = None
    if request.method == "POST":
        article = request.form['article']
        result = m.predict_tags(article)
        
        print(result)
    return render_template("index.html",result=result )



if __name__ == "__main__":
    app.run(debug=True)

    