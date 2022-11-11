import requests
from pathlib import Path
import os
import shutil

import zipfile
import py7zr

import argparse


data_path = Path("data/")

if not data_path.is_dir():
    data_path.mkdir(parents=True, exist_ok=True)

def download_sketchy():
    sketchy_path = data_path / "sketchy"

    if not sketchy_path.is_dir():
        sketchy_path.mkdir(parents=True, exist_ok=True)
    else:
        check = input("Do you want to redownload the sketchy_dataset? [y/n] ")
        if check != 'y': return


    
    with open(data_path / "sketchy.7z", "wb") as f:
        print("Downloading main sketchy")
        request = requests.get("https://drive.google.com/u/0/uc?id=1z4--ToTXYb0-2cLuUWPYM5m7ST7Ob3Ck&export=download&confirm=t&uuid=629d846a-8320-449a-b8c7-6b23fc35cac5&at=AHV7M3fyxJYRWSigUdjNPukj6p98:1668164254721")
        print("Downloaded")
        f.write(request.content)
        print("...")

    with py7zr.SevenZipFile(data_path / "sketchy.7z", "r") as zip_ref:
        print("Unzipping main sketchy")
        zip_ref.extractall(sketchy_path)

    os.remove(data_path / "sketchy.7z")

    #other rendering options available but probably not needed
    original_sketch_path = sketchy_path / "256x256/sketch/tx_000000000000"
    target_sketch_path = sketchy_path / "sketches_png"
    shutil.move(original_sketch_path, target_sketch_path)

    original_sketch_path = sketchy_path / "256x256/photo/tx_000000000000"
    target_sketch_path = sketchy_path / "photos"
    shutil.move(original_sketch_path, target_sketch_path)

    print("Removing unneeded sketches")
    shutil.rmtree(sketchy_path / "256x256")
    #os.remove(sketchy_path / "256x256") does not work for directory
    

    with open(data_path / "sketchy_info.7z", "wb") as f:
        print("Downloading sketchy info")
        request = requests.get("https://drive.google.com/u/0/uc?id=1x8n7qaMg1z2SC-1sT5yjIMmMIr0UcBVW&export=download")
        f.write(request.content)

    with py7zr.SevenZipFile(data_path / "sketchy_info.7z", "r") as zip_ref:
        print("Unzipping sketchy info")
        zip_ref.extractall(sketchy_path)

    os.remove(data_path / "sketchy_info.7z")
    

    with open(data_path / "sketchy_svg.7z", "wb") as f:
        print("Downloading sketchy svg")
        request = requests.get("https://drive.google.com/u/0/uc?id=1Qr8HhjRuGqgDONHigGszyHG_awCstivo&export=download&confirm=t&uuid=d055343a-e1ee-4aee-8eee-6dd692c3dddb&at=AHV7M3d8DVQr2oPlUVrP-UsFL-cd:1668167067619")
        print("Downloaded")
        f.write(request.content)
        print("...")

    with py7zr.SevenZipFile(data_path / "sketchy_svg.7z", "r") as zip_ref:
        print("Unzipping sketchy svg")
        zip_ref.extractall(sketchy_path)
        print("...")

    os.remove(data_path / "sketchy_svg.7z")
    os.rename(sketchy_path / "sketches", sketchy_path / "sketches_svg")

    print("Finished downloading the Sketchy Dataset")




# command line tool

msg = "depending on the option corresponding data will be downloaded"

parser = argparse.ArgumentParser(description=msg)

#may be add TU-Berlin sketches only later
parser.add_argument("--sketchy_download", action="store_true", help="downloads sketchy dataset with info and svg")

args = parser.parse_args()

if args.sketchy_download:
    download_sketchy()