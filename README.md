# phdata-challenge.

Repository created for the phData Machine Learning Engineer (MLE) challenge.

# Structure of the repository.

This repository follows the structure outlined below:

```
├── conda_environment.yml
├── create_model.py
├── data
│   ├── future_unseen_examples.csv
│   ├── kc_house_data.csv
│   └── zipcode_demographics.csv
├── Dockerfile
├── LICENSE
├── Makefile
├── model
│   ├── metrics.json
│   ├── model_features.json
│   ├── model.pkl
│   ├── regression_residuals.png
│   └── regression_scatterplot.png
├── notebooks
│   └── exploration.ipynb
├── presentation
├── README.md
├── ruff.toml
├── send_request_housing.py
├── src
│   └── app.py
└── tests
    └── test_requests.py
```

# How to use this solution.

## Building the docker image.

To build the Docker image, run the command below in your terminal. If you have Docker Desktop installed, you'll be able to monitor the build process through its interface.

```
make build-image-locally
```

## Running the container

After building the image,you can start the container using the following command. Here you can also monitor the build process through Docker Desktop interface.

```
make run-image-locally
```

Once the container is up and running, you can send POST requests to test the application.

## Testing the App

To test the application, there is two differents ways you can execute the script in the root and send some requests tothe API:

```
python3 send_request_housing.py
```

This script performs two POST requests one to each endpoint of the API to validate the application’s behavior.

Other way is to use the spcript inside the folder tests/ running the script as below from the root of the project.

```
pytest tests
```

This script is going to run two simple testing functions validating the responses from de API.

## Next Steps

This application can certainly be improved and extended into a more robust solution. Some potential next steps include:

- MLflow Component.
- Implementing a CI/CD pipeline with integration and unit tests.
- Migrate solution to the Cloud.
