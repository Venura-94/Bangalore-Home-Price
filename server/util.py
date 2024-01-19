import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,total_sqft,bhk,bath):

    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns)) # Numnber of Zeros will be same as the length of the data columns
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0: # Because we are using dummy one-hot encoding.
        x[loc_index] = 1

    return round(__model.predict([x])[0],2) # we access the zeroth element since this has only one predicted value according to the inputs

def get_location_names():
    return __locations

def load_saved_artifacts():
    print("load saved artifacts....start")
    global __data_columns
    global __locations
    
    with open("./server/artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    global __model
    with open('./server/artifacts/house_price_prediction_mdl.pkl', 'rb') as f:
        __model = pickle.load(f)
    print("load saved artifacts....done")

if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st phase jp nagar',1000, 3, 3))
