from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('pipe.pkl', 'rb'))

# Define the teams and venues
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders',
         'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Chandigarh', 'Jaipur', 'Chennai',
          'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur',
          'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru']

@app.route("/")
def home():
    return render_template('home.html', teams=teams, cities=cities)

@app.route("/predict", methods=['POST'])
def predict():
    battingteam = request.form['battingteam']
    bowlingteam = request.form['bowlingteam']
    city = request.form['city']
    target = int(request.form['target'])
    score = int(request.form['score'])
    overs = int(request.form['overs'])
    wickets = int(request.form['wickets'])

    runs_left = target - score
    balls = 120 - 6 * overs
    wickets = 10 - wickets
    currentrunrate = score / overs
    requiredrunrate = (runs_left * 6) / balls

    data = pd.DataFrame({'batting_team': [battingteam], 'bowling_team': [bowlingteam], 'city': [city],
                         'runs_left': [runs_left], 'balls_left': [balls], 'wickets': [wickets],
                         'total_runs_x': [target], 'cur_run_rate': [currentrunrate],
                         'req_run_rate': [requiredrunrate]})

    result = model.predict_proba(data)
    lossprob = result[0][0]
    winprob = result[0][1]

    winprob_percent = "{:.2f}%".format(winprob * 100)
    lossprob_percent = "{:.2f}%".format(lossprob * 100)

    winmsg = f"The probability of {battingteam} winning is {winprob_percent}"
    lossmsg = f"The probability of {bowlingteam} winning is {lossprob_percent}"

    return render_template('home.html', teams=teams, cities=cities, m1=winmsg, m2=lossmsg)

if __name__ == '__main__':
    app.run(debug=True)
