# The very most basicest

.PHONY: build docker-build clean deploy

init:
	echo "Init"

test:
	pytest

perftest: test
	echo "Perftest"

doc:
	echo "Doc"

build: docker-build

docker-build:
	docker build -t flask-sample-one:latest -f Dockerfile.alpine .

docker-python:
	docker build -t python374 -f Dockerfile.ubuntu .

docker-python-bash:
	docker run -it python374 bash

clean:
	yes | docker system prune
	-docker rmi flask-sample-one

superclean:

deploy:
	docker run -d -p 8000:8000 flask-sample-one

Thesis.pdf:
	Rscript -e "rmarkdown::render('Thesis.rmd')"
