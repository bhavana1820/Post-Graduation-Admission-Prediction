from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load the dataset
data = pd.read_csv("Admission_Predict_Custom.csv")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract user inputs
    GRE_Score = int(request.form['GRE Score'])
    TOEFL_Score = int(request.form['TOEFL Score'])
    CGPA = float(request.form['CGPA'])

    # Prepare the input data
    user_input = [[GRE_Score, TOEFL_Score, CGPA]]
    scaled_input = scaler.transform(user_input)

    # Predict admission chance
    prediction = model.predict(scaled_input)[0]

    # Filter universities and courses based on user inputs
    if prediction == 1:
        eligible_universities = data[
            (data['GRE Score'] <= GRE_Score + 5) &
            (data['TOEFL Score'] <= TOEFL_Score + 5) &
            (data['CGPA'] <= CGPA + 0.5)
        ][['University Name', 'Course Name']]
        result = eligible_universities.to_html(index=False, classes='table table-striped')
    else:
        low_chance_universities = data[
            (data['GRE Score'] > GRE_Score - 10) &
            (data['TOEFL Score'] > TOEFL_Score - 10) &
            (data['CGPA'] > CGPA - 0.5)
        ][['University Name', 'Course Name']]
        result = " "
        result += low_chance_universities.to_html(index=False, classes='table table-striped')
        # result = "Unfortunately, you do not have a high chance of admission."

    # return render_template('index.html', result=result)

    return render_template(
    'index.html',
    result=result,
    gre_score=GRE_Score,
    toefl_score=TOEFL_Score,
    cgpa=CGPA
)


if __name__ == "__main__":
    app.run(debug=True)