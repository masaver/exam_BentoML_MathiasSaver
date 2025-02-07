# Examen BentoML

First, we must set the working enviroment and create a python enviroment by executing the following lines
1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt `

Next, we download the data from aws-s3 by executing:
`wget -O ./data/raw/admissions_test.csv https://assets-datascientest.s3.eu-west-1.amazonaws.com/MLOPS/bentoml/admission.csv`

The headers of this table were also manually modified (see below) to have a nicer format:
`**serial_no,gre_score,toefl_score,university_ranking,sop,lor,cgpa,research,chance_of_admit**`

With that, we are ready to train the model ( Random Forest regressor with GridSearchCV ), save it to the BentoML store and create the bento service. For this, we execute the following two lines:
* `python ./src/prepare_data.py`
* `python ./src/train_model.py `

The code for the bento service is located in `./src/service.py`, and the respective bento file is located in `./bentofile.yaml`

Now, we build and containerize the bento by executing:
* `bentoml build`
* `bentoml containerize rf_reg_service:latest`
* `docker tag rf_reg_service:latest masaver/rf_reg_service:2.0.0`
* `docker push masaver/rf_reg_service:2.0.0`

The docker image can also be pulled from Docker Hub by running: `docker pull masaver/rf_reg_service:2.0.0`

To start the service under localhost:3000, and run it inside a detached screen session, you can execute:
**`screen -dmS rf_service docker run -p 3000:3000 masaver/rf_reg_service:2.0.0`**

Next, to run the respective unit tests the show the API is working properly, you can execute: **`pytest ./src/test_service.py -v`**

Finally, when the service is no longer needed you can stop it ( kill the respective screen session ) by running: **`screen -S {session_name} -X quit`**