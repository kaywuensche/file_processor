# File Processor

API for file processing. It's a collection of methods for handling various file types:
- Pdf
- Excel
- Word
- Powerpoint
- Video
- Image
- Text

**Goals**:
- Extract text from files for further usage in natural language processing, like natural entity recognition or text classification
- Extract images from files for further usage in computer vision, like [optical character recognition](https://github.com/kaywuensche/inference_optical_character_recognition), [image classification](https://github.com/BMW-InnovationLab/BMW-Classification-Inference-GPU-CPU) or [object detection](https://github.com/BMW-InnovationLab/BMW-YOLOv4-Inference-API-CPU)
- General methods for image preparation, like resizing or augmentation
- General methods for text preparation, like language classification or chunking

## Prerequisites:
- docker
- docker-compose

### Check for prerequisites
**To check if docker-ce is installed:**

```docker --version```

**To check if docker-compose is installed:**

```docker-compose --version```

### Install prerequisites
**Ubuntu**

To install [Docker](https://docs.docker.com/engine/install/ubuntu/) and [Docker Compose](https://docs.docker.com/compose/install/) on Ubuntu, please follow the link.

**Windows 10**

To install Docker on [Windows](https://docs.docker.com/docker-for-windows/install/), please follow the link.

**P.S: For Windows users, open the Docker Desktop menu by clicking the Docker Icon in the Notifications area. Select Settings, and then Advanced tab to adjust the resources available to Docker Engine.**

## Build The Docker Image
In order to build the project run the following command from the project's root directory:

```sudo docker-compose up --build --remove-orphans```

## API Endpoints
To see all the available endpoints, open your favorite browser and navigate to:

```http://<machine_IP>:5011/docs```

<img width="373" alt="Bildschirmfoto 2021-05-21 um 09 19 25" src="https://user-images.githubusercontent.com/58667455/119098001-aaf20780-ba15-11eb-88aa-898306c11ba2.png">

Some of the endpoint will response with a pdf or a zip file. Fast API can't display these files, but you can use a tool like Postman with the additional functionality of saving the response for test purposes:

<img width="947" alt="Bildschirmfoto 2021-05-20 um 16 57 26" src="https://user-images.githubusercontent.com/58667455/119001731-8867dc00-b98c-11eb-92d2-475d1ff690c2.png">

