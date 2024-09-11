from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract form data
            Kms_Driven = int(request.form['Kms_Driven'])
            Year = int(request.form['Year'])
            Fuel_Type = request.form['Fuel_Type']
            Seller_Type = request.form['Seller_Type']
            Transmission = request.form['Transmission']
            Owner = request.form['Owner']

            # Preprocess form data
            Kms_Driven2 = np.log(Kms_Driven + 1)  # Log transformation for kms driven
            no_year = 2020 - Year  # Calculate age of the car

            # One-hot encode fuel type
            fuel_Diesel = fuel_Electric = fuel_LPG = fuel_Petrol = 0
            if Fuel_Type == 'Diesel':
                fuel_Diesel = 1
            elif Fuel_Type == 'Electric':
                fuel_Electric = 1
            elif Fuel_Type == 'LPG':
                fuel_LPG = 1
            elif Fuel_Type == 'Petrol':
                fuel_Petrol = 1

            # One-hot encode seller type
            seller_type_Individual = seller_type_Trustmark_Dealer = 0
            if Seller_Type == 'Individual':
                seller_type_Individual = 1
            elif Seller_Type == 'Trustmark Dealer':
                seller_type_Trustmark_Dealer = 1

            # Encode transmission
            transmission_Manual = 1 if Transmission == 'Manual' else 0

            # One-hot encode owner type
            owner_First = owner_Second = owner_Third = owner_Fourth_Above = owner_Test_Drive = 0
            if Owner == 'First Owner':
                owner_First = 1
            elif Owner == 'Second Owner':
                owner_Second = 1
            elif Owner == 'Third Owner':
                owner_Third = 1
            elif Owner == 'Fourth & Above Owner':
                owner_Fourth_Above = 1
            elif Owner == 'Test Drive Car':
                owner_Test_Drive = 1

            # Prepare the input array for prediction
            features = np.array([[Kms_Driven2, no_year,
                                  fuel_Diesel, fuel_Electric, fuel_LPG, fuel_Petrol,
                                  seller_type_Individual, seller_type_Trustmark_Dealer,
                                  transmission_Manual,
                                  owner_Fourth_Above, owner_Second, owner_Test_Drive, owner_Third]])

            # Make prediction
            prediction = model.predict(features)
            output = round(prediction[0], 2)

            if output < 0:
                return render_template('index.html', prediction_texts="Sorry, you cannot sell this car")
            else:
                return render_template('index.html', prediction_text="You Can Sell The Car at ₹{}".format(output))

        except Exception as e:
            return render_template('index.html', prediction_texts=f"Error: {str(e)}")

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
