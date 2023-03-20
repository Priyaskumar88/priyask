
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler
import pickle


model = pickle.load(open('model.pkl', 'rb'))
minmax = pickle.load(open('minmax.pkl', 'rb'))

# Creating our app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
      
        Tenure = float(request.form['Tenure'])
        CityTier= int(request.form['CityTier'])
        WarehouseToHome = float(request.form.get('WarehouseToHome', False))
        HourSpendOnApp = float(request.form['HourSpendOnApp'])
        NumberOfDeviceRegistered = int(request.form['NumberOfDeviceRegistered'])
        SatisfactionScore  = float(request.form.get('SatisfactionScore', False))
        NumberOfAddress = int(request.form['NumberOfAddress'])
        Complain = int(request.form['Complain'])
        OrderAmountHikeFromlastYear  = float(request.form['OrderAmountHikeFromlastYear'])
        CouponUsed = float(request.form['CouponUsed'])
        OrderCount = float(request.form['OrderCount'])
        DaySinceLastOrder = float(request.form['DaySinceLastOrder'])
        CashbackAmount  = float(request.form['CashbackAmount'])

        PreferredLoginDevice_Phone = request.form['PreferredLoginDevice']

        if PreferredLoginDevice_Phone == 'Phone':
            PreferredLoginDevice_Phone = 1
            PreferredLoginDevice_Computer= 0
            
        else:
            PreferredLoginDevice_Phone = 0
            PreferredLoginDevice_Computer= 1
        
        MaritalStatus_Married = request.form['MaritalStatus']

        if(MaritalStatus_Married == 'Married'):
            MaritalStatus_Married  = 1
            MaritalStatus_Single = 0
            MaritalStatus_Divorced= 0

        elif(MaritalStatus_Married == 'Single'):
            MaritalStatus_Married  = 0
            MaritalStatus_Single = 1
            MaritalStatus_Divorced= 0

        else:
            MaritalStatus_Married  = 0
            MaritalStatus_Single = 0
            MaritalStatus_Divorced= 1

        
        Gender_Male = request.form['Gender']
        if(Gender_Male == 'Male'):
            Gender_Male = 1
            Gender_Female = 0

        else:
            Gender_Male = 0
            Gender_Female = 1

        PreferredPaymentMode_COD = request.form['PreferredPaymentMode']
        if(PreferredPaymentMode_COD == 'COD'):
            PreferredPaymentMode_COD  = 1
            PreferredPaymentMode_DebitCard = 0  
            PreferredPaymentMode_Ewallet = 0       
            PreferredPaymentMode_UPI = 0 
            PreferredPaymentMode_CC = 0

        elif(PreferredPaymentMode_COD == 'Debit Card'):
            PreferredPaymentMode_COD  = 0
            PreferredPaymentMode_DebitCard = 1  
            PreferredPaymentMode_Ewallet = 0       
            PreferredPaymentMode_UPI = 0 
            PreferredPaymentMode_CC = 0

        elif(PreferredPaymentMode_COD == 'E wallet'):
            PreferredPaymentMode_COD  = 0
            PreferredPaymentMode_DebitCard = 0 
            PreferredPaymentMode_Ewallet = 1      
            PreferredPaymentMode_UPI = 0 
            PreferredPaymentMode_CC = 0

        elif(PreferredPaymentMode_COD == 'CC'):
            PreferredPaymentMode_COD  = 0
            PreferredPaymentMode_DebitCard = 0 
            PreferredPaymentMode_Ewallet = 0      
            PreferredPaymentMode_UPI = 0
            PreferredPaymentMode_CC = 1
       
        else:
            PreferredPaymentMode_COD  = 0
            PreferredPaymentMode_DebitCard = 0 
            PreferredPaymentMode_Ewallet = 0    
            PreferredPaymentMode_UPI = 1
            PreferredPaymentMode_CC = 0
        
        PreferedOrderCat_Grocery = request.form['PreferedOrderCat']
        if(PreferedOrderCat_Grocery == 'Grocery'):
            PreferedOrderCat_Grocery  = 1
            PreferedOrderCat_Laptop = 0  
            PreferedOrderCat_Mobile = 0       
            PreferedOrderCat_Others= 0 
            PreferedOrderCat_Fashion= 0 

        elif(PreferedOrderCat_Grocery == 'Laptop'):
            PreferedOrderCat_Grocery  = 0
            PreferedOrderCat_Laptop = 1
            PreferedOrderCat_Mobile = 0       
            PreferedOrderCat_Others= 0 
            PreferedOrderCat_Fashion= 0 
        

        elif(PreferedOrderCat_Grocery == 'Mobile'):
            PreferedOrderCat_Grocery  = 0
            PreferedOrderCat_Laptop = 0
            PreferedOrderCat_Mobile = 1      
            PreferedOrderCat_Others= 0 
            PreferedOrderCat_Fashion= 0 
        
        elif(PreferedOrderCat_Grocery == 'Others'):
            PreferedOrderCat_Grocery  = 0
            PreferedOrderCat_Laptop = 0  
            PreferedOrderCat_Mobile = 0       
            PreferedOrderCat_Others= 1
            PreferedOrderCat_Fashion= 0 
        
        else:
            PreferedOrderCat_Grocery = 0
            PreferedOrderCat_Laptop=0  
            PreferedOrderCat_Mobile = 0       
            PreferedOrderCat_Others= 0
            PreferedOrderCat_Fashion= 1
        

         # Preprocess the input data using the scaler
        scaled_data=minmax.transform([[
            Tenure,
            CityTier,
            WarehouseToHome,
            HourSpendOnApp,
            NumberOfDeviceRegistered,
            SatisfactionScore,
            NumberOfAddress,
            Complain,
            OrderAmountHikeFromlastYear,
            CouponUsed,
            OrderCount,
            DaySinceLastOrder,
            CashbackAmount,
            PreferredLoginDevice_Phone,
            Gender_Male,
            MaritalStatus_Married,
            MaritalStatus_Single,
            PreferredPaymentMode_COD,
            PreferredPaymentMode_DebitCard,
            PreferredPaymentMode_Ewallet,
            PreferredPaymentMode_UPI,
            PreferedOrderCat_Grocery, 
            PreferedOrderCat_Laptop,
            PreferedOrderCat_Mobile, 
            PreferedOrderCat_Others]])
        

            # Make a prediction using the model
        prediction = model.predict(scaled_data)


        output = prediction[0]
        if output == 1:
                     return render_template('result.html',prediction_text="The Customer will leave")
        else:
                     return render_template('result.html',prediction_text="The Customer will not leave")
        

if __name__=="__main__":
    app.run(debug=False,host='0.0.0.0')


     
 
