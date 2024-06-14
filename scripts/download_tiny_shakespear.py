import os
import urllib.request

os.makedirs("../dataset", exist_ok=True)

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
file_path = os.path.join("../dataset", "tiny_shakesprear.txt")

urllib.request.urlretrieve(url, file_path)
