import torch
import cv2


# El modelo de la inteligencia artificial está cargado en el Github "ncotti/yolov5"
#model = torch.hub.load('ncotti/yolov5', 'yolov5s', trust_repo=True)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

im = 'https://ultralytics.com/images/zidane.jpg'

results = model(im)
print("Hola")

#results.show()  // Muestra la imagen con recuadros
results.render()
cv2.imshow("test", results.ims[0])


#print (results.crop(save=True))
cv2.waitKey()
print("Adios")

#print(results.pandas().xyxy[0])