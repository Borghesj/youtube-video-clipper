import os

try:
    import cv2
    FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
except ImportError:
    print("Warning: OpenCV (cv2) is not installed. Face detection will not be available.")
    FACE_CASCADE_PATH = None

# Paths
IMAGEMAGICK_BINARY = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"

# Project directory (where main.py is located)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Output directory (within the project directory)
OUTPUT_DIRECTORY = os.path.join(PROJECT_DIR, "output")

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)

# AI Model Paths
SENTIMENT_MODEL_PATH = os.path.join(PROJECT_DIR, 'models', 'sentiment_model.h5')

# OpenAI API Key (you should set this securely, preferably as an environment variable)
OPENAI_API_KEY = "your_openai_api_key_here"
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"
