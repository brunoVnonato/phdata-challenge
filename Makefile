build-image-locally:
	docker build -t phdata-challange .
	
run-image-locally:	
	docker run -d --name phdata-challange-container -v /home/brunovn/Projects/phdata-challenge/model:/app/model --env-file .env -p 8080:8080 phdata-challange

update-conda-environment:
	conda env update --file conda_environment.yml --prune

housingapp-test:
	python3 tests/test_housingapp.pydicke


