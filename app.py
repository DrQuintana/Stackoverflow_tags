from flask import Flask,request, render_template
#from requests import request
import model as m 
#import test as m 

app = Flask(__name__)


@app.route("/", methods = ["GET","POST"])
def tags():
    if request.method == "POST":
        article = request.form['article']
        results = m.predict_tags(article)
        print(results)
    return render_template("index.html",results = results)



if __name__ == "__main__":
    app.run(debug=True)