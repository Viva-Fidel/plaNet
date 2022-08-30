from main import *

#NeuralWeb
PlantNet = cv2.dnn.readNet('PlaNet_weights.weights', 'PlaNet_config.cfg')
classes = []
with open ('plants.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = PlantNet.getLayerNames()
output_layers = [layer_names[i-1] for i in PlantNet.getUnconnectedOutLayers()]

#Colors
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

