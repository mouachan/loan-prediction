# Model deployment with a custom transfomer for data pre/post processing

This folder deploys a sklearn model server on RHOAI with the loan-prediction model. The model server embeds a transfomer to transform a readable json into tensors. The Kserve inference service can be customized (for instance to set scaling for both predictor and transformer). Here is a pointer to the [Kserve documentation](https://kserve.github.io/website/master/reference/api/#serving.kserve.io/v1beta1.ComponentExtensionSpec).

## Prerequisites
- RHOAI with Kserve component deployed (tested with the serverless deployment method)
- Adapt the `data-connection.yaml` with the credentials of your object storage or ensure:
    - A minio instance is deployed in the `minio` namespace
    - The model `../models/loan-prediction.joblib` is located at `s3://models/joblib/loan-prediction.joblib`

## Source and build

The transformer python script is located under `src/`. The Containerfile is located under `container/`. 

## Deploy 

```
oc new-project loan-prediction
oc apply -f manifests/

➜  transformer git:(main) ✗ oc get isvc,servingruntime,pod

NAME                                                      URL                                                                                            READY   PREV   LATEST   PREVROLLEDOUTREVISION   LATESTREADYREVISION                    AGE
inferenceservice.serving.kserve.io/kserve-sklearnserver   https://kserve-sklearnserver-loan-prediction.apps.cluster-bsrhx.bsrhx.sandbox286.opentlc.com   True           100                              kserve-sklearnserver-predictor-00001   34s

NAME                                                    DISABLED   MODELTYPE   CONTAINERS         AGE
servingruntime.serving.kserve.io/kserve-sklearnserver              sklearn     kserve-container   34s

NAME                                                                  READY   STATUS    RESTARTS   AGE
pod/kserve-sklearnserver-predictor-00001-deployment-57f896d545fmssd   3/3     Running   0          32s
pod/kserve-sklearnserver-transformer-00001-deployment-6466bdd68rj88   3/3     Running   0          32s
```

## Test

**Results**: Inference to the model endpoint (which routes to the transformer component)

``` shell
# Get the model endpoint
MODEL=model
MODEL_URL=$(oc get isvc kserve-sklearnserver  -ojsonpath='{.status.url}')
TRANSFORMER_PAYLOAD=./test/transformer-payload.json
cat $TRANSFORMER_PAYLOAD
curl -X POST -H "Content-Type: application/json" -d @$TRANSFORMER_PAYLOAD $MODEL_URL/v2/models/$MODEL/infer
```

*Output*: 

```
MODEL=model
MODEL_URL=$(oc get isvc kserve-sklearnserver  -ojsonpath='{.status.url}')
TRANSFORMER_PAYLOAD=./test/transformer-payload.json
cat $TRANSFORMER_PAYLOAD
{ 
    "inputs": [
        { 
            "name": "X", 
            "shape": [1, 16], 
            "datatype": "FP32", 
            "data": [{
                "Loan_ID": "LP001002",
                "Gender": "Male",
                "Married": "No",
                "Dependents": "0",
                "Education": "Graduate",
                "Self_Employed": "No",
                "ApplicantIncome": "5849",
                "CoapplicantIncome": "0.0",
                "LoanAmount": "120",
                "Loan_Amount_Term": "360.0",
                "Credit_History": "1.0",
                "Property_Area": "Urban"
            }]
        }
    ]
}
curl -X POST -H "Content-Type: application/json" -d @$TRANSFORMER_PAYLOAD $MODEL_URL/v2/models/$MODEL/infer
{
  "model_name": "model",
  "model_version": null,
  "id": "409bc53b-c3bf-488c-a364-b88858cc755b",
  "parameters": null,
  "outputs": [
    {
      "name": "output-0",
      "shape": [
        1
      ],
      "datatype": "BOOL",
      "parameters": null,
      "data": [
        true
      ]
    }
  ]
}
```

**Withtout transformer**: Inference to the predictor endpoint (which does not routes to the transformer component)

``` shell
# Get the model endpoint
MODEL=model
PREDICTOR_URL=$(oc get isvc kserve-sklearnserver  -ojsonpath='{.status.components.predictor.url}')
ORIGINAL_PAYLOAD=./test/original-payload.json
cat $ORIGINAL_PAYLOAD
curl -X POST -H "Content-Type: application/json" -d @$ORIGINAL_PAYLOAD $PREDICTOR_URL/v2/models/$MODEL/infer
```

*Output:*
```
PREDICTOR_URL=$(oc get isvc kserve-sklearnserver  -ojsonpath='{.status.components.predictor.url}')
ORIGINAL_PAYLOAD=./test/original-payload.json
cat $ORIGINAL_PAYLOAD
{ 
    "inputs": [
        { 
            "name": "X", 
            "shape": [1, 16], 
            "datatype": "FP32", 
            "data": [1.0, 0.0, 1.0, 0.0, 48.99017333984375, 0.0, 7.978983402252197, 0.7435897588729858, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] 
        }
    ]
}
curl -X POST -H "Content-Type: application/json" -d @$ORIGINAL_PAYLOAD $PREDICTOR_URL/v2/models/$MODEL/infer
{
  "model_name": "model",
  "model_version": null,
  "id": "0237775d-3987-4944-b89d-d3f42802a438",
  "parameters": null,
  "outputs": [
    {
      "name": "output-0",
      "shape": [
        1
      ],
      "datatype": "BOOL",
      "parameters": null,
      "data": [
        true
      ]
    }
  ]
}
```