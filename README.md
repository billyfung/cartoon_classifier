# Basic cartoon image classifier
Resnet34 trained model from Fastai on cartoon data.

Cartoons included so far:
- simpsons
- peanuts
- dragonball
- family guy
- one piece

Backend api built using Fastapi

## Setup
Using the docker image tiangolo/uvicorn-gunicorn-fastapi:python3.7

build the image
```
docker build . -t myapp 
```

run the app
`docker run -d -p 80:80 -v $(pwd):/app myapp /start-reload.sh`

visit 127.0.0.1

## Improvements
More data, longer training. This was more a proof of concept to see how quickly I could get it up and into production. `flex` option in google app engine is not ideal for prototyping and expensive.