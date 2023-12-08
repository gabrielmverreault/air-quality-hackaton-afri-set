# Afri-SET 

This project was created as part of the Air Quality Hackathon Challenge

**Challenge 5**
Afri-SET evaluates low-cost air quality sensors in West Africa for use in under-served areas. They want to create a manufacturer-agnostic database to store data from different sensors, which currently involves bespoke solutions for each manufacturer.

They envision providing a data management solution for any stakeholder who wishes to have their sensors hosted on our platform. Additionally, they aim to report corrected data from low-cost sensors, which require information beyond specific pollutants. They need a database that can easily find ancillary data and apply corrections.

Visit the Afri SET website for additional information on the non-profit organization: https://afriset.org/


![Logo](logo-camaraderie-afriset.png)

## Pre-requisites
This project leverages Amazon Bedrock, to access state-of-the-art Anthropic Claude 2.1 model. 
- AWS Account with Amazon Bedrock enabled and Claude 2.1
- At the time of publishing, Claude 2.1 is only available in `us-west-2` and `us-east-1`

## Configure your environment

To manually create a virtualenv

```bash
python3 -m venv .venv
```

After the init process completes and the virtualenv is created, you can use the following
step to activate your virtualenv.

```bash
source .venv/bin/activate
```

Create a `.env` file in the root directory and add values for the following

```
AWS_REGION="YourRegion" #example us-east-1
AWS_PROFILE="airquality" #from ~/.aws/config
S3_BUCKET="TargetBucketName"
```

Install the required packages

`pip install -r requirements.txt`


## Go!

Define the file path, or point to an S3 filepath, and run the notebook!
