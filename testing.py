import pandas as pd
import telecom_model
test_df = pd.read_csv("Tele-com churn\current_data.csv")
sample_size = 1
hello =telecom_model.input_run(test_df.sample(n= sample_size,random_state=2))
hd = pd.DataFrame({"Customer_status" : hello})
hd['Customer_status']=hd['Customer_status'].map({0:"Loyal",1 :"Churn"})
print(hd)