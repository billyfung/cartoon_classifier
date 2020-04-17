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

## Deploying 
The easy option is to deploy to google app engine, with the flex environment. The downside is that this is also the expensive option

Google cloud run is much cheaper, you pay for what you use

1. Set up project and Google Cloud cli
2. Store and build image in Container Registry
```
gcloud builds submit --tag gcr.io/funsies-274500/cartoon
```
3. Deploy to Cloud Run
```
gcloud run deploy --image gcr.io/funsies-274500/cartoon --platform managed --memory 1Gi
```
