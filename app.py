from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# --------------------------------------------------
# Load trained Linear Regression model
# --------------------------------------------------
try:
    with open("house_price_model.pkl", "rb") as file:
        model = pickle.load(file)

    print("Model loaded successfully.")

    # Display model features
    if hasattr(model, "feature_names_in_"):
        print("Model features:")
        print(list(model.feature_names_in_))

except Exception as e:
    print("Error loading model:", e)
    model = None


# --------------------------------------------------
# Home Page
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# House Price Prediction
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:

        if model is None:
            return """
            <h2>Model could not be loaded.</h2>
            <a href="/">Go Back</a>
            """

        # --------------------------------------------------
        # Get values from HTML form
        # --------------------------------------------------

        bedrooms = float(request.form["bedrooms"])
        bathrooms = float(request.form["bathrooms"])
        sqft_living = float(request.form["sqft_living"])
        sqft_lot = float(request.form["sqft_lot"])
        floors = float(request.form["floors"])
        waterfront = float(request.form["waterfront"])
        view = float(request.form["view"])
        condition = float(request.form["condition"])

        sqft_above = float(request.form["sqft_above"])
        sqft_basement = float(request.form["sqft_basement"])
        yr_built = float(request.form["yr_built"])
        yr_renovated = float(request.form["yr_renovated"])

        city = request.form["city"]

        year = float(request.form["year"])
        month = float(request.form["month"])
        day = float(request.form["day"])


        # --------------------------------------------------
        # Create basic input DataFrame
        # NOTE:
        # grade is NOT included because the saved model
        # was not trained with grade.
        # --------------------------------------------------

        input_data = pd.DataFrame({
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "sqft_living": [sqft_living],
            "sqft_lot": [sqft_lot],
            "floors": [floors],
            "waterfront": [waterfront],
            "view": [view],
            "condition": [condition],
            "sqft_above": [sqft_above],
            "sqft_basement": [sqft_basement],
            "yr_built": [yr_built],
            "yr_renovated": [yr_renovated],
            "year": [year],
            "month": [month],
            "day": [day]
        })


        # --------------------------------------------------
        # One-Hot Encoding for CITY
        # --------------------------------------------------

        # Get all features expected by the trained model
        if hasattr(model, "feature_names_in_"):

            expected_features = list(model.feature_names_in_)

        else:
            return """
            <h2>Error</h2>
            <p>The saved model does not contain feature names.</p>
            <a href="/">Go Back</a>
            """


        # --------------------------------------------------
        # Create all city columns expected by model
        # --------------------------------------------------

        city_columns = [
            column
            for column in expected_features
            if column.startswith("city_")
        ]

        # Create city dummy columns
        for column in city_columns:
            input_data[column] = 0


        # Create the correct city column
        city_column = "city_" + city

        if city_column in input_data.columns:
            input_data[city_column] = 1


        # --------------------------------------------------
        # Add any missing features
        # --------------------------------------------------

        for feature in expected_features:

            if feature not in input_data.columns:
                input_data[feature] = 0


        # --------------------------------------------------
        # Remove extra features
        # --------------------------------------------------

        input_data = input_data[
            [feature for feature in expected_features]
        ]


        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        prediction = model.predict(input_data)

        predicted_price = float(prediction[0])


        # --------------------------------------------------
        # Result Page
        # --------------------------------------------------

        return f"""
        <!DOCTYPE html>

        <html lang="en">

        <head>

            <meta charset="UTF-8">

            <meta name="viewport"
                  content="width=device-width, initial-scale=1.0">

            <title>House Prediction Result</title>

            <style>

                * {{
                    box-sizing: border-box;
                    margin: 0;
                    padding: 0;
                    font-family: Arial, sans-serif;
                }}

                body {{
                    min-height: 100vh;

                    background:
                    linear-gradient(
                        135deg,
                        #dbeafe,
                        #f3e8ff
                    );

                    display: flex;
                    justify-content: center;
                    align-items: center;
                }}

                .result {{
                    width: 90%;
                    max-width: 550px;

                    background: white;

                    padding: 45px;

                    border-radius: 20px;

                    text-align: center;

                    box-shadow:
                    0 10px 30px
                    rgba(0, 0, 0, 0.15);
                }}

                h1 {{
                    color: #1e3a8a;
                    margin-bottom: 25px;
                }}

                h2 {{
                    color: #15803d;
                    margin-bottom: 30px;
                    font-size: 25px;
                }}

                a {{
                    display: inline-block;

                    padding: 13px 25px;

                    background: #2563eb;

                    color: white;

                    text-decoration: none;

                    border-radius: 8px;

                    font-weight: bold;
                }}

                a:hover {{
                    background: #1d4ed8;
                }}

            </style>

        </head>

        <body>

            <div class="result">

                <h1>House Price Prediction</h1>

                <h2>
                    Predicted Price:
                    ₹{predicted_price:,.2f}
                </h2>

                <a href="/">
                    Predict Another House
                </a>

            </div>

        </body>

        </html>
        """


    except Exception as e:

        return f"""
        <!DOCTYPE html>

        <html>

        <head>
            <title>Prediction Error</title>
        </head>

        <body>

            <h2>Error occurred during prediction</h2>

            <p>{str(e)}</p>

            <br>

            <a href="/">Go Back</a>

        </body>

        </html>
        """


# --------------------------------------------------
# Run Flask Application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)