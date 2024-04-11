# Grab these secrets from the environment variables we set via "source .env"
AWS_ACCESS_KEY_ID := $(aws_access_key)
AWS_SECRET_ACCESS_KEY := $(aws_secret_key)
IMAGE_NAME := rcb
CONTAINER_NAME := rcb
LOCAL_ABS_PATH := /Users/deanorenstein/Documents/academic/coding_stuff/research-chatbot/

# Build the container, setting the ARGs (build time), which also sets the ENV variables (run time)
build:
	docker build --build-arg AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} --build-arg AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} -t ${IMAGE_NAME}:latest .

# Run the container, using a bind mount for the code, so we dont need to rebuild image when making code changes
run:
	docker run -d --rm --name ${CONTAINER_NAME} -p 8000:8000 -v ${LOCAL_ABS_PATH}:/app ${IMAGE_NAME}

stop:
	docker stop ${CONTAINER_NAME}

# Check that an environment variable is set
check:
	@echo "API key is ${AWS_ACCESS_KEY_ID}" && echo "Secret key is ${AWS_SECRET_ACCESS_KEY}"