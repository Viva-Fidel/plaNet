import cv2
import numpy as np
import streamlit as st
import webcolors

# NeuralWeb
PlantNet = cv2.dnn.readNet('configuration/PlaNet_weights.weights', 'configuration/PlaNet_config.cfg')
with open('configuration/labels.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = PlantNet.getLayerNames()
output_layers = [layer_names[i - 1] for i in PlantNet.getUnconnectedOutLayers()]
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Colors
lower_blue_v1 = np.array([105, 50, 50])
upper_blue_v1 = np.array([130, 255, 255])

lower_blue_v2 = np.array([60, 20, 0])
upper_blue_v2 = np.array([150, 255, 255])

lower_green = np.array([30, 50, 0])
upper_green = np.array([70, 255, 255])

lower_red = np.array([1, 0, 0])
upper_red = np.array([80, 255, 255])

lower_blue_v3 = np.array([90, 220, 0])
upper_blue_v3 = np.array([100, 255, 255])

#if mean_hue_value < 5:
#    plant_color = "Red"
#elif mean_hue_value < 22:
#    plant_color = "Orange"
#elif mean_hue_value < 33:
#    plant_color = "Yellow"
#elif mean_hue_value < 78:
#    plant_color = "Green"
#elif mean_hue_value < 131:
#    plant_color = "Blue"
#elif mean_hue_value < 170:
#    plant_color = "Violet"
#else:
#    plant_color = "Undefined"