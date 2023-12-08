# Afri-SET 

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

Installed the required packages

`pip install -r requirements.txt`


## Go!

Define the file path, or point to an S3 filepath, and run the notebook!
