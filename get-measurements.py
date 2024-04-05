"""get-measurements.py

    This Python script is a modification of the version found in the following URL:
    https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html

    Author: Juan Carlos Chávez Villarreal
    Organisation: Universidad de Monterrey
    Contact: juan.chavezv@udem.edu
    First created: 2024-04-05

    EXAMPLE OF USAGE
    python get-measurements.py
    -c 0
    --z 50
    -j calibration_data.json

    python get-measurements.py -c 0 --z 50 -j calibration_data.json

"""
import numpy as np
import cv2
import glob
import os
import argparse
import sys
import textwrap
import json
import platform
from numpy.typing import NDArray
from typing import List, Tuple

points = [] #Tupla que almacena los puntos del dibujo
drawing = False #True si ya se completó el dibujo

def user_arguments() -> argparse.ArgumentParser:
    """
    Parse arguments.

    Returns:
        argparse.ArgumentParser: Object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(prog='HW8 - 3D camera measurement', 
                                    description='Calculate dimensions of user provided geometries.', 
                                    epilog='JCCV - 2024')
    parser.add_argument('--camera_index',
                        '-c', 
                        type=int, 
                        required=True,
                        help="Index for desired camera ")
    parser.add_argument('--z',
                        type=float,
                        required=True,
                        help="Distance between camera and object")
    parser.add_argument('--input_calibration_parameters',
                        '-j',
                        type=str,
                        required=True,
                        help='JSON file with calibration parameters')
    args = parser.parse_args()
    return args

def load_calibration_parameters_from_json_file(
        args:argparse.ArgumentParser):
    """
    Load camera calibration parameters from a JSON file.

    Args:
        args: Parsed command-line arguments.

    Returns:
        camera_matrix: Camera matrix.
        distortion_coefficients: Distortion coefficients.

    This function may raise a warning if the JSON file 
    does not exist. In such a case, the program finishes.
    """

    # Check if JSON file exists
    json_filename = args.input_calibration_parameters
    check_file = os.path.isfile(json_filename)

    # If JSON file exists, load the calibration parameters
    if check_file:
        f = open(json_filename)
        json_data = json.load(f)
        f.close()
        
        camera_matrix = np.array(json_data['camera_matrix'])
        distortion_coefficients = np.array(json_data['distortion_coefficients'])
        return camera_matrix, distortion_coefficients
    
    # Otherwise, the program finishes
    else:
        print(f"The file {json_filename} does not exist!")
        sys.exit(-1)

def initialize_camera(args):
    """
    Initialize the camera.

    Args:
        args: Parsed command line arguments.

    Returns:
        cv.VideoCapture: Initialized camera object.
    """
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("Error al inicializar la cámara")
        return None
    return cap

def mouse_callback(event,x,y,flags,param):
    """
    Callback function for mouse events.

    Args:
        event: Type of mouse event.
        x: X coordinate of the mouse event.
        y: Y coordinate of the mouse event.
        flags: Flags indicating the mouse event details.
        param: Additional parameters.
    Return:
        None
    
    """
    global points
    global drawing

    if event == cv2.EVENT_RBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_ALTKEY:
            if points:
                points.clear()
        else:
            if points:
                points.pop()
    elif event == cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))
    elif event == cv2.EVENT_MBUTTONDOWN:
        points.append(points[0])
        drawing = True

def undistort_images(
        frame, 
        mtx:NDArray, 
        dist:NDArray, 
        )->NDArray:
    """
    Undistort images using camera calibration parameters and return 
    the undistorted images.

    Args:
        frame: distorted frame.
        mtx: Camera matrix.
        dist: Distortion coefficients.

    Returns:
        undistorted image 
    """

    # Get size
    h,  w = frame.shape[:2]

    # Get optimal new camera
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    # Undistort image
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # Crop image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

def compute_line_segments(points: List[Tuple[int,int]]):
    """
    """

    line_length = [] #matriz donde se guardarán las distancias medidas
    for i in range(1, len(points)):
        x1, y1 = points[i-1] #punto punto anterior
        x2, y2 = points[i] # punto nuevo
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) #distancia entre dos puntos
        line_length.append(length) #agrega dato a matriz
    return line_length

def compute_perimeter(points: List[Tuple[int, int]], z: float, mtx: np.ndarray, height:int, width:int):
    distance = []
    perimeter = 0.0

    Cx = mtx[0,2]*width/1280
    Cy = mtx[1,2]*height/720
    
    fx = mtx[0,0]*width/1280
    fy = mtx[1,1]*height/720

    for i in range(1, len(points)):
        x1, y1 = points[i-1]
        x2, y2 = points[i]
        
        # Convertir de píxeles a coordenadas
        X1 = (x1 - Cx) * z / fx
        Y1 = (y1 - Cy) * z / fy
        X2 = (x2 - Cx) * z / fx
        Y2 = (y2 - Cy) * z / fy
        
        # Calcular distancia entre puntos
        dist = np.sqrt((X2 - X1)**2 + (Y2 - Y1)**2 )
        distance.append(dist)

        perimeter += dist
    return distance, perimeter

def pipeline():
    global drawing
    
    #datos de usuario
    args = user_arguments()

    #iniciar camara
    cam = initialize_camera(args)

    #Abrir pantalla y activar mouse
    cv2.namedWindow('Live Camera View')
    cv2.setMouseCallback('Live Camera View',mouse_callback)

    #cargar datos de .json
    mtx, dist = load_calibration_parameters_from_json_file(args)

    while True:
        ret, frame = cam.read() #ret es una variablo booleana para saber si hay video
        if not ret:
            print("Error: no hay señal")
            break
        
        h,w = frame.shape[:2]

        if drawing:
            #calcular distancias y perimetros
            distance,perimeter = compute_perimeter(points, args.z,mtx,h,w)

            # distancias a dos puntos decimales en orden de registro
            text = "Distancias (puntos seleccionados):\n"
            for i, dist in enumerate(distance, start=1):
                if i < len(points):
                    text += f"Punto {i}-{i+1}: {dist:.2f} cm\n"
                else:
                    text += f"Punto {i}-{1}: {dist:.2f} cm\n"

            text += f"\nPerímetro total: {perimeter:.2f} cm"

            text += "\nMedidas ordenadas de mayor a menor:\n"
            sorted_distance = sorted(distance, reverse=True)
            for i, dist in enumerate(sorted_distance, start=1):
                index = distance.index(dist)
                if index == len(points) - 1:
                    text += f"Punto {len(points)}-{1}: {dist:.2f} cm\n"
                else:
                    text += f"Punto {index+1}-{index+2}: {dist:.2f} cm\n"

            print(text)
            drawing = False  #reiniciar estado del dibujo 

        # Dibujar las líneas entre los puntos seleccionados
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], (0, 255, 0), 1)

        # Dibujar los puntos seleccionados
        for point in points:
            cv2.circle(frame, point, 3, (0, 255, 0), -1)

        cv2.imshow('Live Camera View', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    pipeline()            