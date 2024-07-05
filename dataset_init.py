from roboflow import Roboflow
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Access the API_KEY environment variable
API_KEY = os.getenv('API_KEY')


rf = Roboflow(api_key=API_KEY)
project = rf.workspace("roboflow-100").project("chess-pieces-mjzgj")
version = project.version(2)
dataset = version.download("yolov8")
