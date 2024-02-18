
import sys
import os

script_path = os.path.abspath(__file__)  # 获取当前脚本的绝对路径
parent_dir = os.path.dirname(os.path.dirname(script_path))  # 获取父目录的路径
sys.path.append(parent_dir)
from modular.utils import download_data
# Download pizza, steak, sushi images from GitHub
image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                           destination="pizza_steak_sushi")
print(image_path)