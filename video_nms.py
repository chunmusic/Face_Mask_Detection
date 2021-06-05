import cv2
import numpy as np

cap = cv2.VideoCapture("mask1.mp4")

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


def detect_object(img, returndata=False, conf = 0.0):
    
    # Get the rows, cols of Image 
    rows, cols, channels = img.shape
 
    # This is where we pre-process the image, Resize the image and Swap Image Channels
    # We're converting BGR channels to RGB since OpenCV reads in BGR and our model was trained on RGB images
    blob = cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True)
    
    # Set the blob as input to the network
    net.setInput(blob)
 
    # Runs a forward pass, this is where the model predicts on the image.
    networkOutputs = net.forward()
 

    class_ids_list = []
    boxes_list = []
    confidences_list = []
    
    # Loop over the output results
    for detection in networkOutputs[0,0]:
        
        # Get the score for each detection
        score = float(detection[2])
        
        # IF the class score is bigger than our threshold
        if score > conf:

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

            class_ids_list.append(class_index)
            confidences_list.append(float(score))
            boxes_list.append([x1,y1,x2,y2])

    max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

    if max_value_ids == ():
        pass

    else:

        for max_valueid in max_value_ids:

            max_class_id = max_valueid[0]

            print(max_class_id)
            if max_class_id == np.NaN:
                pass

            else:

                box = boxes_list[max_class_id]

                predicted_class_id = class_ids_list[max_class_id]
                predicted_class_label = classes[predicted_class_id]
                prediction_confidence = confidences_list[max_class_id]


                text = "{},  {:.2f}% ".format(predicted_class_label, prediction_confidence*100)
                cv2.putText(img, text, (box[0], box[3]+ 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                            (255,0,255), 2)
                
                # Draw the bounding box
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (125, 255, 51), thickness = 2)
            
    cv2.imshow("img",img)

    cv2.waitKey(1)



if __name__== "__main__":
    initialize()

    while(cap.isOpened()):

        ret, img = cap.read()

        detect_object(img)



cap.release()

cv2.destroyAllWindows()
 
