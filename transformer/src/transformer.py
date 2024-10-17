import kserve
import argparse
from kserve import InferInput, InferRequest, ModelServer, model_server
from typing import Dict, Union
import logging
import joblib
import pandas as pd
import numpy as np


scaler_filename = "scaler.save"
scaler = joblib.load(scaler_filename)

col_types = {'Gender':"str",
             'Married':"str",
             'Dependents':"str",
             'Education':"str",
             'Self_Employed':"str",
             'ApplicantIncome':"float",
             'CoapplicantIncome':"float",
             'LoanAmount':"float",
             'Loan_Amount_Term':"float",
             'Credit_History':"bool",
             'Property_Area':"str",
             'Loan_Status':"str"}

def transform(data):
    df = pd.DataFrame.from_dict(data, orient='index').transpose()
    
    # Drop Unecessary Variables
    if 'Loan_ID' in df.columns:
        df = df.drop(['Loan_ID'], axis=1)
    
    # Set Variable types
    df = df.astype(dtype = {key: value for key, value in col_types.items() if key != 'Loan_Status'})
    df = df.replace("nan", None)
    
    # One-hot Encoding of Categorical Variables
    df['Gender'] = df['Gender'].replace({'Female': False, 'Male': True})
    df['Married'] = df['Married'].replace({'No': False, 'Yes': True})
    df['Education'] = df['Education'].replace({'Not Graduate': False, 'Graduate': True})
    df['Self_Employed'] = df['Self_Employed'].replace({'No': False, 'Yes': True})
    df['Dependents_0'] = df['Dependents'].apply(lambda x: True if x == '0' else False)
    df['Dependents_1'] = df['Dependents'].apply(lambda x: True if x == '1' else False)
    df['Dependents_2'] = df['Dependents'].apply(lambda x: True if x == '2' else False)
    df['Dependents_3+'] = df['Dependents'].apply(lambda x: True if x == '3+' else False)
    df = df.drop('Dependents', axis=1)
    df['Property_Area_Rural'] = df['Property_Area'].apply(lambda x: True if x == 'Rural' else False)
    df['Property_Area_Semiurban'] = df['Property_Area'].apply(lambda x: True if x == 'Semiurban' else False)
    df['Property_Area_Urban'] = df['Property_Area'].apply(lambda x: True if x == 'Urban' else False)
    df = df.drop('Property_Area', axis=1)
    
    # Data Normalization Over Data Set
    df = scaler.transform(df)

    return df.astype(np.float32)

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)
# for REST predictor the preprocess handler converts to input dict to the v1 REST protocol dict
class ImageTransformer(kserve.Model):
    def __init__(self, name: str, predictor_host: str, protocol: str, predictor_use_ssl: bool):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.protocol = protocol
        self.use_ssl = predictor_use_ssl
        self.ready = True

    def preprocess(self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None) -> Dict:
        input_tensors = transform(payload.inputs[0].data[0])
        input_tensors = np.asarray(input_tensors, dtype="float32")
        print(input_tensors)
        shape = [1, 16]
        name = "X"
        infer_inputs = [InferInput(name=name, datatype='FP32', shape=shape,
                                   data=input_tensors)]
        infer_request = InferRequest(model_name=self.name, infer_inputs=infer_inputs)
        print(infer_request.to_rest())
        return infer_request

    def postprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        return inputs
    
parser = argparse.ArgumentParser(parents=[model_server.parser])
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    print(args)
    model = ImageTransformer(args.model_name, predictor_host=args.predictor_host,
                             protocol=args.protocol, predictor_use_ssl=args.predictor_use_ssl)
    ModelServer().start([model])