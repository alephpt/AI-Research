import json
import boto3
import sagemaker
from sagemaker.huggingface import get_huggingface_llm_image_uri, HuggingFaceModel

prompt = """
            Your name is Glaza. You have been trained on the Falcon-7B-Instruct model, and are the central bot to help assist in the management of Order 332.
            Order 332 is a discord server that embraces cultures from all parts of the world, and values education and open-mindedness. The Order is based around
            the pillars of Community, Knowledge and Friendship. We aspire to maintain brutal honesty, openness, and full disclosure.. even it it means
            being brutally honest, and saying things that people may not here. Please adhere to these values, and speak with a crude but honest tongue.
        """

instance_type = "ml.p3.2xlarge"
num_gpus = 4
health_check_timeout = 300

config = {
    'HF_MODEL_ID': 'tiiuae/falcon-7b-instruct',
    'SM_NUM_GPUS': json.dumps(num_gpus),
    'MAX_INPUT_LENGTH': json.dumps(1024),
    'MAX_TOTAL_TOKENS': json.dumps(2048),
}

session = sagemaker.Session()
sagemaker_session_bucket = None

if sagemaker_session_bucket is None and session is not None:
    sage_maker_session_bucket = session.default_bucket()

print(f"Sagemaker session bucket: {sage_maker_session_bucket}")

try:
    role = sagemaker.get_execution_role()
except ValueError:
    client = boto3.client('iam')
    role = client.get_role(RoleName='SageMaker-sagemaker-execution')['Role']['Arn']

print(f"Sagemaker role arn: {role}")

session = sagemaker.Session(default_bucket=sage_maker_session_bucket)

print(f"Safemaker session region: {session.boto_region_name}")

llm_image = get_huggingface_llm_image_uri(
    "huggingface",
    version="0.8.2"
)

print(f"llm image uri: {llm_image}")

llm_model = HuggingFaceModel(
    role=role,
    image_uri=llm_image,
    env=config
)

llm = llm_model.deploy(
    initial_instance_count=1,
    instance_type=instance_type,
    container_startup_health_check_timeout=health_check_timeout
)

userIn = input("Enter a prompt: ")

full_prompt = prompt + userIn + "\nGlaza:"

payload = {
    "inputs": full_prompt,
    "parameters": {
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.8,
        "max_new_tokens": 1024,
        "repitition_penalty": 1.03,
        "stop": ["\n", "Glaza:"]
    }
}

response = llm.predict(payload)

assistant1 = response[0]['generated_text'].split("\nGlaza:")[1]
assistant2 = response[0]['generated_text'][len(full_prompt):]

print(f"assistant1: {assistant1}")
print(f"assistant2: {assistant2}")

userIn = input("Enter a prompt: ")

full_prompt = prompt + userIn + "\nGlaza:"

payload["inputs"] = full_prompt
response = llm.predict(payload)

assistant3 = response[0]['generated_text'].split("\nGlaza:")[1]
assistant4 = response[0]['generated_text'][len(full_prompt):]

print(f"assistant3: {assistant3}")
print(f"assistant4: {assistant4}")

#llm.delete_model()
#llm.delete_endpoint()
