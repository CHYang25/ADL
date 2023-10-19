#! /bin/bash

# model data download
gdown https://drive.google.com/drive/folders/1PlltWQWvU9QSWSdEuj09I3LYOOGDZ6VD -O ./model/multiple_choice_dir --folder
gdown https://drive.google.com/drive/folders/1ugOlxEGIrCqOuYdGla6xFvMsnvLS4PuI -O ./model/extractive_dir --folder

# data download
gdown https://drive.google.com/drive/folders/1vBKarytyQUjwgFSPXg2s_cGbnyl7Gyla -O ./data/ --folder