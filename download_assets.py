import os
import sys
import gdown

print("Starting download of weights...")
weights_url = "https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i"
os.makedirs("weights", exist_ok=True)
gdown.download_folder(weights_url, output="weights", quiet=False, use_cookies=False)

print("Starting download of demo data...")
demo_url = "https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP"
os.makedirs("demo_data", exist_ok=True)
gdown.download_folder(demo_url, output="demo_data", quiet=False, use_cookies=False)

print("Downloads finished!")
