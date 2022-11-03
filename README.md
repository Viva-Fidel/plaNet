# plaNet
## Introduction
The programme can calculate leaves and convex hull area of plants. Also it can calculate width and legth of the root.

## Usage
To use the app please open the [website](https://viva-fidel-planet-main-n4sviq.streamlitapp.com/)  
Accpeted formats:
* jpg
* png 
* jpeg

#### Basic plant detection
Basic plant detection uses standart OpenCV algorithm. The plant should be on a white background. Anywhere near the plant should be allocated blue square with 1cm x 1cm dimensions. 

#### Advanced plant detection
Advanced plant detection uses standart pretrained [Yolov4](https://github.com/AlexeyAB/darknet). The plant should be on a white background. Anywhere near the plant should be allocated blue square with 1cm x 1cm dimensions. 

#### Plant root detection
Plant root detection uses standart OpenCV algorithm. The root should be on a red background. Anywhere near the root should be allocated blue square with 1cm x 1cm dimensions. 

## Downloading data
All the data is saved in CSV format, processed photos are saved in zip. To download the data press "Download data as CSV" or "Download ZIP file with results" 
For plant detection next information is saved:  
* File_name  
* Color  
* Leaves_area  
* Convex hull area  

For plant root detection next information is saved:  
* File_name  
* Root width  
* Root lenght  
