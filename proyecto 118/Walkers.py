import cv2
import numpy as np

# Crea nuestro body classifier
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Inicializa video capture para el archivo de video
cap = cv2.VideoCapture('walking.avi')

# Inicia el bucle cuando el video se haya cargado correctamente
while True:
    
    # Lee el primer cuadro
    ret, frame = cap.read()
    #frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pasa el cuadro a nuestro body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
    
    # Extrae los cuadros delimitadores de los cuerpos identificados
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Pedestrians', frame)

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()