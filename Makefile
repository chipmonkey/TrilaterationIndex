# The very most basicest

.PHONY: build docker-build clean deploy

init:
	echo "Init"

venv: venv/bin/activate
	. venv/bin/activate

venv/bin/activate: monkeynn/requirements.txt
	test -d venv || python3 -m venv venv

test:
	pytest

perftest: test
	echo "Perftest"

doc: Thesis/Thesis.pdf
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

Thesis/Thesis.pdf: Thesis/*.rmd
	Rscript -e "rmarkdown::render('Thesis/Thesis.rmd',output_file='Thesis.pdf')"

readdoc:
	evince /home/chipmonkey/repos/TrilaterationIndex/Thesis/Thesis.pdf

# Apt packages required for python and R dependencies
apt:
	apt install libudunits2-dev libgdal-dev
