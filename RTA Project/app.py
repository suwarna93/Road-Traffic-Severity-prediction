
 
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from prediction import get_prediction, ordinal_encoder

st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚧", layout="wide")
                   

pickle_in = open('model.pkl', 'rb') 
model = pickle.load(pickle_in)


Accident_Severity = ('serious_injury','slight_injury','fatal_injury') 

#creating option list for dropdown menu   
           

options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']

options_Age_band_of_driver = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']

options_Sex_of_driver = ["Male","Female"]

options_Educational_level =["Graduate","Undergraduate"]

options_Driving_experience = ["1-5","5-10","10-20"]

options_Type_of_vehicle =['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
       'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
       'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi']

options_Owner_of_vehicle =['Owner','Governmental','Organization']

options_Service_year_of_vehicle =['Above 10yr','5-10yrs','1-2yr','2-5yrs','Unknown']

options_Area_accident_occured = ['Office areas', 'Residential areas', 'Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', 'Hospital areas', 'Market areas',
       'Rural village areas','Rural village areasOffice areas',
       'Recreational areas']

options_Lanes_or_Medians =['Two-way (divided with broken lines road marking)', 'Undivided Two way','other', 'Double carriageway (median)', 'One way',
       'Two-way (divided with solid lines road marking)']

Road_allignment = ['1','2','3','4','5','6','7','8','9']

Types_of_Junction = ['1','2','3','4','5','6','7','8','9']

options_Road_surface_type =['1','2','3','4','5','6','7','8','9','10']

Road_surface_conditions =['1','2','3','4','5','6']

Light_conditions =['1','2','3','4','5','6']

Weather_conditions =['1','2','3','4','5','6','7','8','9']

Type_of_collision =['1','2','3','4','5','6','7','8','9','10']

Number_of_vehicles_involed =['1','2','3','4','5','6','7','8','9','10']

Number_of_casualties =['1','2','3','4','5','6','7','8','9','10']

options_Vehicle_movement =['Going straight','U-Turn','Moving Backward','Turnover','Waiting to go']

options_Casualty_class =['Driver or rider','Pedestrian','Pedestrian','Passenger']

options_Sex_of_casualty =['Male','Female']

options_Age_band_of_casualty =['31-50','18-30','Under 18']

Casualty_severity =['1','2','3','4','5','6','7','8','9','10']

options_Pedestrian_movement =['Moving Backward','Overtaking','Changing lane to the left','Overloading','Other','No priority to vehicle','No priority to pedestrian','No distancing','No priority to vehicle']

options_Cause_of_accident =['No distancing', 'Changing lane to the right',
       'Changing lane to the left', 'Driving carelessly','No priority to vehicle', 'Moving Backward','No priority to pedestrian','Overtaking',
       'Driving under the influence of drugs', 'Driving to the left','Getting off the vehicle improperly', 'Driving at high speed','Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
       'Improper parking']


features = ['Day_of_week', 'Age_band_of_driver', 'Sex_of_driver','Educational_level', 'Driving_experience','Type_of_vehicle','Owner_of_vehicle', 'Service_year_of_vehicle', 'Area_accident_occured',
       'Lanes_or_Medians', 'Road_allignment', 'Types_of_Junction','Road_surface_type', 'Road_surface_conditions', 'Light_conditions', 'Weather_conditions', 'Type_of_collision',
      'Number_of_vehicles_involved','Number_of_casualties','Vehicle_movement','Casualty_class','Sex_of_casualty','Age_band_of_casualty','Casualty_severity','Pedestrian_movement',
       'Cause_of_accident', 'hour', 'minute']


 

st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():

    with st.form('prediction_form'):

         st.header("Enter the input for following features:")


         hour = st.slider("Pickup Hour:", 0, 23, value=0, format="%d")
         minute = st.slider('minute', 0, 59, value=0, format="%d")

         Day_of_week = st.selectbox ('Select Day of the Week: ', options=options_day)
        

         Age_band_of_driver= st.selectbox("Select Driver Age: ", options=options_age)

         Sex_of_driver =  st.selectbox('Sex_of_driver:',options=options_Sex_of_driver )
         Educational_level = st.selectbox('Educational_level:',options=options_Educational_level)
         Driving_experience = st.selectbox('Driving_experience:',options=options_Driving_experience)
         Type_of_vehicle = st.selectbox('Type_of_vehicle:',options=options_Type_of_vehicle)
         Owner_of_vehicle = st.selectbox('Owner_of_vehicle:',options=options_Owner_of_vehicle)
         Service_year_of_vehicle = st.selectbox('Service_year_of_vehicle:',options=options_Service_year_of_vehicle)
         Area_accident_occured = st.selectbox('Area_accident_occured:',options=options_Area_accident_occured)
         Lanes_or_Medians = st.selectbox('Lanes_or_Medians:',options=options_Lanes_or_Medians)
         Road_allignment = st.slider('Road_allignmen', 1, 10, value=0, format="%d")
         Types_of_Junction = st.slider('Types_of_Junction', 1, 10, value=0, format="%d")
         Road_surface_type = st.slider('Road_surface_type', 1, 10, value=0, format="%d")
         Road_surface_conditions = st.slider('Road_surface_conditions ', 1, 7, value=0, format="%d")
         Light_conditions = st.slider('Light_conditions', 1, 7, value=0, format="%d")
         Weather_conditions = st.slider('Weather_conditions', 1, 10, value=0, format="%d")
         Type_of_collision = st.slider('Type_of_collision', 1, 10, value=0, format="%d")
         Number_of_vehicles_involed = st.slider('Number_of_vehicles_involed', 1, 10, value=0, format="%d")
         Number_of_casualties = st.slider('Number_of_casualties', 1, 10, value=0, format="%d")
         Vehicle_movement = st.selectbox('vehicle_movement:',options=options_Vehicle_movement)
         Casualty_class = st.selectbox('casulty_class:',options=options_Casualty_class)
         Sex_of_casualty = st.selectbox('sex_of_casualty:',options=options_Sex_of_casualty)
         Age_band_of_casualty = st.selectbox('age_band_of_casualty:',options=options_Age_band_of_casualty)

         Casualty_severity = st.slider('casualty_severity', 1, 10, value=0, format="%d")
         Pedestrian_movement = st.selectbox('pedestrian_movement:',options=options_Pedestrian_movement)
         Cause_of_accident = st.selectbox('cause_of_accident:',options=options_Cause_of_accident)




         submit_values = st.form_submit_button("Predict")

    if submit_values:

     Day_of_week = ordinal_encoder(Day_of_week,options_day) 
     Age_band_of_driver =  ordinal_encoder(Age_band_of_driver,options_age )
     Sex_of_driver =  ordinal_encoder(Sex_of_driver,options_Sex_of_driver)
     Educational_level = ordinal_encoder(Educational_level,options_Educational_level)
     Driving_experience = ordinal_encoder(Driving_experience,options_Driving_experience)
     Type_of_vehicle = ordinal_encoder(Type_of_vehicle,options_Type_of_vehicle)
     Owner_of_vehicle = ordinal_encoder(Owner_of_vehicle,options_Owner_of_vehicle)
     Service_year_of_vehicle = ordinal_encoder(Service_year_of_vehicle,options_Service_year_of_vehicle)
     Area_accident_occured  = ordinal_encoder(Area_accident_occured,options_Area_accident_occured)
     Lanes_or_Medians = ordinal_encoder(Lanes_or_Medians,options_Lanes_or_Medians)
     Vehicle_movement = ordinal_encoder(Vehicle_movement,options_Vehicle_movement)
     Casualty_class = ordinal_encoder(Casualty_class,options_Casualty_class)
     Sex_of_casualty = ordinal_encoder(Sex_of_casualty,options_Sex_of_casualty)
     Age_band_of_casualty = ordinal_encoder(Age_band_of_casualty,options_Age_band_of_casualty)
     Pedestrian_movement = ordinal_encoder(Pedestrian_movement,options_Pedestrian_movement)
     Cause_of_accident = ordinal_encoder(Cause_of_accident,options_Cause_of_accident)




   

              
    data = np.array([Day_of_week,Age_band_of_driver, Sex_of_driver,Educational_level, Driving_experience,Type_of_vehicle,
    Owner_of_vehicle, Service_year_of_vehicle, Area_accident_occured,Lanes_or_Medians, Road_allignment, Types_of_Junction,Road_surface_type, Road_surface_conditions, Light_conditions,
    Weather_conditions, Type_of_collision,Number_of_vehicles_involed,Number_of_casualties,Vehicle_movement, Casualty_class,Sex_of_casualty,
    Age_band_of_casualty,Casualty_severity,Pedestrian_movement,Cause_of_accident, hour, minute]).reshape(1,-1)


    if Accident_Severity == 0:
       value = 'seriour_injury' 
    elif Accident_Severity == 1:
       value = 'slight_injury'
    else:
       value = 'fatal_injury' 



    pred = get_prediction(data=data, model=model)

    st.write(f"The predicted severity is:  {pred[0]}")

         


     
     
     
if __name__ == '__main__':
  main()      

       

 



        


























