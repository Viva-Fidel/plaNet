import cv2
import sys
import os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home_page():
    return render_template('home.html')