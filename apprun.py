import streamlit as st
from dotenv import load_dotenv
import os
import boto3
import pandas as pd
import json
import re
import sys
from datetime import datetime
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate

load_dotenv()

st.title("Afri-SET MVP")

#Setup the environment
session = boto3.Session(
    region_name=os.getenv("AWS_REGION"), profile_name=os.getenv("AWS_PROFILE")
)
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',    
    region_name=os.environ.get("AWS_REGION", None))

sns_client = session.client('sns')
s3 = session.client('s3')



#Function definition

def preprocess_csv(csv_file):

    df = pd.read_csv(csv_file, nrows=0)
    num_expected_cols = len(df.columns)
    
    df = pd.read_csv(csv_file)

    # Identify numeric and string columns
    numeric_cols = df.select_dtypes(include=['number']).columns 
    string_cols = df.select_dtypes(exclude=['number']).columns

    # String columns - fill NaN with empty string 
    df[string_cols] = df[string_cols].fillna('')  

    # Numeric cols - fill NaN with Pandas NaN
    df[numeric_cols] = df[numeric_cols].fillna(pd.NA)

    # Trim or drop rows with extra columns
    df = df.iloc[:, :num_expected_cols]  
    
    return df

def load_source_file(input_data_file):
  ext = input_data_file.split('.')[-1]
  
  if ext == 'json':
    data = json.load(open(input_data_file))
    threshold = 50000
    if len(str(data)) > threshold:
      sample_data = str(data)[:threshold]
    else:
      sample_data = data

  elif ext == 'csv':
      data = preprocess_csv(input_data_file)
      sample_data = data

  return sample_data, data

def get_final_code(text):
  # Extract code from markdown codeblock   
  pattern = r'```python(.*?)\n```'
  match = re.search(pattern, text, re.DOTALL)
  if match:
    return match.group(1)
  return ''

def create_df(input_data_file):
  ext = input_data_file.split('.')[-1]
  if ext == 'json':
    data = json.load(open(input_data_file))
    threshold = 50000
    if len(str(data)) > threshold:
      sample_data = str(data)[:threshold]
    else:
      sample_data = data
    output = llm(p1_prompt_template.format(input=sample_data))
    convert_code = (get_final_code(output))
    exec(convert_code,globals())
    df = convert_to_df(data)
  elif ext == 'csv':
      df = preprocess_csv(input_data_file)
      convert_code = ''
  return df, convert_code

def get_table_type(llm_output):
    match = re.search(r'<tableType>(\w+)</tableType>', llm_output)
    if match:
        return match.group(1)
    else:
        return None


def branch_LLM_invocations(file_name, is_existing):
    if not is_existing:
        human_in_the_loop(file_name)
    else:
        pass # call some python function later to output SQL tables
def human_in_the_loop(file_name):
    message = "New data format is detected! Please check the file: " + file_name
    msg_body = json.dumps(message)
    sns_client.publish(
                TopicArn=sns_topic_arn,
                Message=json.dumps({'default': msg_body}),
                MessageStructure='json')
    print("\nSNS published message: \n" + str(message))

def upload_to_s3(df, bucket, key):
   s3.put_object(Bucket=bucket, Key=key, Body=df.to_csv(index=False))

def store_functions(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(os.path.join(folder, "code_tf.py"), "w") as f:
        f.write(code_tf)
        
    with open(os.path.join(folder, "code_pivot.py"), "w") as f:
        f.write(code_pivot)
        
    with open(os.path.join(folder, "convert_code.py"), "w") as f:
        f.write(convert_code)
        
#Templates for LLM
p1_template = """Given the following input file containing air quality data from sensors:
    <input> {input} </input> 
    Provide some python code to convert the data into a pandas dataframe. It should be a function that takes in one input, the input data, as a `dict`.
    `def convert_to_df(input_data):`
    The function should return a panda dataframe.
    <requirements>
    Make sure to import all libraries required.
    Flatten the json as required.
    The timestamp should always be stored as a timestamp. The code must handle conversion. Be mindful of string vs integer. Use a <scratchpad> to Think step by step and insure that the function will work properly when the input data is fed to it.
    </requirements>
    
    OUTPUT: Wrap your final code in the <FinalCode> XML tag and make sure it is formatted as a python code block in Markdown
    EXAMPLE: <FinalCode>```python
import pandas as pd
from datetime import datetime

def convert_to_df(input_data):
        ...
``` 
</FinalCode>
    """

p1_prompt_template = PromptTemplate(
    input_variables=["input"],
    template=p1_template
)

template_pivot = """Type1 table format:
- Each row contains a timestamp
- Each column represents a measurement taken at that time
- Each sensor reading has its own column 
- Type1 example: 
<Type1TableExample>
City,State,Country,Latitude,Longitude,pollution_ts,aqius,mainus,aqicn,maincn,wether_ts,pr,hu,ws,wd,ic
Accra,Greater Accra,Ghana,-0.186964,5.603717,2023-11-25T23:00:00.000Z,74,p2,33,p2,26,1011,82,4.12,272,04n
</Type1TableExample>

Type2 table format 
- Each row contains a timestamp
- One column contains the tag name or value type. This column will not contain the measurements, but instead a string of what the measurement is collecting.
- One column contains the measurement value for that tag. This column will not contain the tag name or value type, but simply the value.
- This table type will generally have only a few columns (3-9)
- Type2 example: 
<Type2TableExample>
timestamp;location;sensor;software_version;value_type;value
2023-11-30 20:37:40.316811+00:00;3615;4829;NRZ-2020-129;P2;14.25
2023-11-30 20:37:40.316811+00:00;3615;4829;NRZ-2020-129;P1;16.25
2023-11-30 20:37:40.316811+00:00;3615;4829;NRZ-2020-129;P0;10.0
</Type2TableExample>
<Type2TableExample2>
timestamp;var;value
2023-11-30 20:37:40;pressure;14.25
2023-11-30 20:37:40;temperature;16.25
2023-11-30 20:37:40;PM1;10.0
2023-11-30 20:39:40;pressure;14.25
2023-11-30 20:39:40;temperature;16.25
2023-11-30 20:39:40;PM1;10.0
</Type2TableExample2>

Given the following table:
<table>
{rawtable}
</table>


Analyze the input table and respond with:

1. The identified table type (Type1 or Type2). Write the answer in a <tableType> XML tag. Think step by step in <scratchpad> to properly identify the type.

2. If the table is Type2:
   - Provide a Python function to transform the table to Type1 format. 
   - The Python functin should be called convert_to_type1 and take in a single input, the input dataframe
   - Use markdown to write the function in a python codeblock

If already Type1, state no transformation needed.

<example>
table:
timestamp;location;sensor;software_version;value_type;value
2023-11-30 20:37:40.316811+00:00;3615;4829;NRZ-2020-129;P2;14.25
2023-11-30 20:37:40.316811+00:00;3615;4829;NRZ-2020-129;P1;16.25
2023-11-30 20:37:40.316811+00:00;3615;4829;NRZ-2020-129;P0;10.0
2023-11-30 20:37:08.424916+00:00;3615;4829;NRZ-2020-129;P2;12.0
2023-11-30 20:37:08.424916+00:00;3615;4829;NRZ-2020-129;P1;12.0
2023-11-30 20:37:08.424916+00:00;3615;4829;NRZ-2020-129;P0;9.0
2023-11-30 20:36:36.545464+00:00;3615;4829;NRZ-2020-129;P2;13.4
2023-11-30 20:36:36.545464+00:00;3615;4829;NRZ-2020-129;P1;16.0
2023-11-30 20:36:36.545464+00:00;3615;4829;NRZ-2020-129;P0;8.0
2023-11-30 20:36:04.634028+00:00;3615;4829;NRZ-2020-129;P2;10.5

<scratchpad>This table has the following key characteristics:
A timestamp column
Columns indicating metadata like location, sensor, software version
Columns for value_type and value
Each row contains a timestamp, metadata about the reading, the type of value, and the actual value. Different value types are captured in different rows for the same timestamp.
This matches the description of a Type 2 table format:
Timestamp column
One column for tag name/value type
One column for the measurement value
Multiple value types captured per timestamp across rows
Therefore, I would classify this as a Type 2 table format. Therefore I need to write a function to transform it to Type1. To do so, I need to look at the values in value_type and pivot them so they become columns. </scratchpad>
<tableType>Type2</tableType>

2: Here is the python code to transform the table to Type1 format:
```python
function:
import pandas as pd  

def convert_to_type1(df):
    df_pivoted = df.pivot(index=['timestamp', 'location', 'sensor', 'software_version'], 
                          columns='value_type',
                          values='value')
    
    df_pivoted.reset_index(inplace=True)

    return df_pivoted
```
</example>

Table type analysis and transformation function (if applicable):
"""

prompt_template_pivot = PromptTemplate(
    input_variables=["rawtable"],
    template=template_pivot
)

templateTransform = """Given this dataframe:
{input_df}
Perform the following operations to create a python function called `transform_df`:
<tasks>
For each column, use the column name and the values, to determine what the column likely contains. Provide a description. Store those desciptions in a <description> XML tag for every column in the dataframe. This does not belong in the function. It is to help you think.

<context>
For context, this is data from air quality sensors, so some common items are:
temperature (in Celsius), humidity, relative humidity, PM1 are extremely fine particulates with a diameter of fewer than 1 microns. PM2.5 (also known as fine particles) have a diameter of less than 2.5 microns. PM10 means the particles have a diameter less than 10 microns, or 100 times smaller than a millimeter. Generally in sensor data, they will be stored in order, PM1, PM2.5 (PM25), then PM10. 
in some dataframes, there will only be a location id (numerical), sometimes, a city. Store the most relevant location, if available, ALWAYS as a string.
<context>
The output file name for all rows will be: {input_filename}
When creating the new dataframe, make sure to properly define the datatype. If not certain, string is OK.
THEN, write a python function to convert it to a dataframe of this format:
<output_structure>
{output_structure}
</output_structure>
Do your best to match the input dataframe to the target. If there are no values for a column of the output_structure, write None

Output your code in python markdown, in <FinalCode> XML tag 
</tasks>
EXAMPLE:
<FinalCode> ```python
import pandas as pd

def transform_df(df):
    
    output_df = pd.DataFrame(columns=['deviceId', 'timestamp', 'locationId', 'geo_lat', 'geo_lon', 'pm1', 'pm10', 'pm25', 'temperature', 'pressure', 'humidity', 'sourcefile'])
    
    output_df['deviceId'] = df['sn']
    output_df['timestamp'] = df['timestamp']
    output_df['locationId'] = None
    output_df['geo_lat'] = df['lat']
    output_df['geo_lon'] = df['lon'] 
    output_df['pm1'] = df['pm1']
    output_df['pm10'] = df['pm10']
    output_df['pm25'] = df['pm25']
    output_df['temperature'] = df['temp'] 
    output_df['pressure'] = None 
    output_df['humidity'] = df['rh']
    output_df['sourcefile'] = df['url']
    
    return output_df
```
</FinalCode>

GO!
"""
output_structure = """
deviceId | timestamp | locationId | geo_lat | geo_lon | pm1 | pm10 | pm25 | temperature | pressure | humidity | sourcefile 

Make sure you store the data in the dataframe with the following datatype:
<datatype>
deviceId          object
timestamp    datetime64[ns]
locationId        object
geo_lat           object   
geo_lon           object
pm1               float64
pm10              float64
pm25              float64
temperature       float64
pressure          float64
humidity          float64
sourcefile        object
</datatype>

"""
prompt_template2 = PromptTemplate(
    input_variables=["input_df", "output_structure", "input_filename"],
    template=templateTransform
)

allowed_extensions = ['.csv', '.json']
uploaded_file = st.file_uploader("Upload sensor data file", type=allowed_extensions)

#
#
# ///////// streamlit Application ///////
#
#

if uploaded_file is not None:
    file_ext = os.path.splitext(uploaded_file.name)[1] 
    if file_ext.lower() not in allowed_extensions:
        st.error("Invalid file extension, please upload .csv or .json")
    else:
        # Save the file to disk
        if not os.path.exists('data/upload'):
            os.makedirs('data/upload')
        dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = f'data/upload/raw_{dt}{file_ext}'
        with open(os.path.join(path), "wb") as f: 
            f.write(uploaded_file.getbuffer())
        st.success("Saved file to {}".format(path))
        
        input_data_file = path
        filename = uploaded_file.name
        
        #show a preview of the source file (for demo purposes only)
        if file_ext.lower() == ".csv":
            df = pd.read_csv(uploaded_file, nrows=100)
            with st.expander("Source CSV Preview"):
                st.write(df)
        elif file_ext.lower() == ".json":
            preview_json = json.load(uploaded_file)
            preview_json = dict(list(preview_json.items())[:5])
            with st.expander("Source JSON Preview"):
                st.json(preview_json)
                
        # LLM 
        llm = Bedrock(model_id="anthropic.claude-v2:1", client=bedrock_runtime, model_kwargs={"temperature":0,"max_tokens_to_sample": 8000, "top_k": 250, "top_p": 1})
        
        #Create df from source file
        df1, convert_code = create_df(input_data_file)
        st.header("Converted Raw File into Datafrane")
        st.write(df1.head(5))
        
        
        #Create pivoted df if needed
        rawtable = df1.head(10).to_csv(sep=';', index = False)
        output_transformation = llm(prompt_template_pivot.format(rawtable=rawtable))
        table_type = get_table_type(output_transformation)
        if table_type == 'Type1':
            df2 = df1
            code_pivot = ''
        else: 
            if table_type == 'Type2':
                code_pivot = get_final_code(output_transformation) 
                exec(code_pivot,globals())
                df2 = convert_to_type1(df1)
            else:
                raise ValueError('Invalid table type')
            
        st.header("Pivoted as needed")
        st.write(f"This table is of type {table_type}")
        st.write(df2.head(5))
        
        
        #Final standardized table
        output_tf = llm(prompt_template2.format(input_df=df2.head(10).to_csv(sep=';', index = False), output_structure=output_structure, input_filename=input_data_file))
        code_tf  = (get_final_code(output_tf))
        exec(code_tf, globals())
        df3 = transform_df(df2)
        st.header("Final Table!")
        st.write(df3.head(5))
        
        #storing the functions
        folder = os.path.join("functions", filename)
        store_functions(folder)
        
        #Save to S3
        bucket_name = os.getenv('S3_BUCKET')
        upload_to_s3(df3,bucket_name,f'processed/transformed/csv/processed_{dt}.csv') #final table, target schema
        upload_to_s3(df1,bucket_name,f'processed/raw/csv/processed_{dt}.csv') #raw data

        with st.container():
            st.success(f"""
                Processed file saved successfully to:  
                    s3://{bucket_name}/transformed/raw/csv/processed_{dt}{file_ext}
                """)