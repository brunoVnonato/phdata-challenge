build-image-locally:
	docker build -t phdata-challange .
	
run-image-locally:	
	docker run -d --name phdata-challange-container --env-file .env -p 8080:8080 phdata-challange

update-conda-environment:
	conda env update --file conda_environment.yml --prune

code-quality:

housingapp-test:
	python3 tests/test_housingapp.py


