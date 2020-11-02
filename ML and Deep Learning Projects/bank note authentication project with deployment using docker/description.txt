This project is related to bank note authentication. This is a end-to-end project deployed using docker. This project is carried out following the below steps.

Step1: Make a model
	-- This process is performed using datasets from https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data. Appropriate model was used in order to get maximum accuracy. In this case, RandomForestClassifier was used.
	-- Refer to bank_note_authentication_model.ipynb for the step

Step2: Save the model using pickle library as .pkl file.
	-- Refer to bank_note_authenticate.pkl for the step

Step3: Using the model to create website using Flask and flassger
	-- This step is used for web development in order to use the model that was created in Step2. There are three user-defined functions created in this step. The first function named welcome() is used to welcome the user, the second function named predict_note_authentication() is used to manually enter the values of features and obtain the target value, the third function named predict_note_file() is used to post the values of features saved in testfile.csv in order to obtain the target values.
	-- The site was created using Flask library and Swagger api was used to design the website. Swagger api handles the front-end development part so, we need not worry about the HTML, CSS and Javascript.
	-- Refer to bank_note_authentication.py for the step

Step4: Deployment of the website using docker
	-- Finally, the website was deployed using docker. For that, docker was installed at first and then Dockerfile along with requirements.txt were created to make a docker container containing all the libraries, dependencies and requirements.
	-- Refer to Dockerfile and requirements.txt for the step

Step5: Building a docker and running it
	-- To build the docker, use the following cmd:
		docker build -t money_api .
	-- To run the docker, use the following cmd:
		docker run -p 8000:8000 money_api

In this way, the model was created and deployed successfully using docker. The above steps are brief description of the project and following these steps any ML projects can be deployed using docker.

Source site for dataset: https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data
Reference help to create the project: https://www.youtube.com/playlist?list=PLZoTAELRMXVNKtpy0U_Mx9N26w8n0hIbs