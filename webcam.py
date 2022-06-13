import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if (cap.isOpened()== False):
    print("Error opening video stream or file")
 

def initialize(pbtxt = 'graph.pbtxt', 
               model = "output/export_saved_model/frozen_inference_graph.pb"):
    
    # Define global variables
    global net, classes
 
    # ReadNet function takes both files and intitialize the network
    net = cv2.dnn.readNetFromTensorflow(model, pbtxt);
 
    # Define Class Labels
    classes = {1: "with_mask", 2: "without_mask", 3: "mask_weared_incorrect"}


def detect_object(img, returndata=False, conf = 0.5):
    
    # Get the rows, cols of Image 
    rows, cols, channels = img.shape
 
    # This is where we pre-process the image, Resize the image and Swap Image Channels
    # We're converting BGR channels to RGB since OpenCV reads in BGR and our model was trained on RGB images
    blob = cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True)
    
    # Set the blob as input to the network
    net.setInput(blob)
 
    # Runs a forward pass, this is where the model predicts on the image.
    networkOutputs = net.forward()
 

 
    # Loop over the output results
    for detection in networkOutputs[0,0]:
        
        # Get the score for each detection
        score = float(detection[2])
        
        # IF the class score is bigger than our threshold
        if score > conf:
            
            # Get the index of the class i.e. 1 or 2
            class_index = int(detection[1])
            
            # Use the Class index to get the class name i.e. Jerry or tom
            class_name = classes[class_index]
            
            # Get the bounding box coordinates.
            # Note: the returned coordinates are relative e.g. they are in 0-1 range.
            # Se we multiply them by rows and cols to get the real coordinates.
            x1 = int(detection[3] * cols)
            y1 = int(detection[4] * rows)
            x2 = int(detection[5] * cols)
            y2 = int(detection[6] * rows)
            
            # Show the class name and the confidence
            text = "{},  {:.2f}% ".format(class_name, score*100)
            cv2.putText(img, text, (x1, y2+ 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                        (255,0,255), 2)
            
            # Draw the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (125, 255, 51), thickness = 2)
            
    cv2.imshow("img",img)

    cv2.waitKey(1)


            


    
if __name__== "__main__":
    initialize()

    while(cap.isOpened()):

        ret, img = cap.read()

        detect_object(img)



cap.release()

cv2.destroyAllWindows()
 
