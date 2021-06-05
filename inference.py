import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

frozen_graph_path = 'output/export_saved_model/frozen_inference_graph.pb'
 
#0.49179258942604065
 
# Read the graph.
with tf.gfile.FastGFile(frozen_graph_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
 
with tf.Session() as sess:
    
    # Set the defualt session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
 
    # Read the Image
    img = cv2.imread('dataset/images/maksssksksss0.jpg')
    
    # Get the rows and cols of the image.
    rows = img.shape[0]
    cols = img.shape[1]
    
    # Resize the image to 300x300, this is the size the model was trained on
    inp = cv2.resize(img, (300, 300))
    
    # Convert OpenCV's BGR image to RGB
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
 
    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
 
    # These are the classes which we want to detect
    classes = {1: "with_mask", 2: "without_mask", 3: "mask_weared_incorrect"}
    
    # Get the total number of Detections
    num_detections = int(out[0][0])
    
    # Loop for each detection
    for i in range(num_detections):
        
        # Get the probability of that class
        score = float(out[1][0][i])
        
        # Check if the score of the detection is big enough
        if score > 0.450:
                                
            # Get their Class ID
            classId = int(out[3][0][i])
 
            # Get the bounding box coordinates of that class
            bbox = [float(v) for v in out[2][0][i]]
            
            # Get the class name
            class_name = classes[classId]

            
            # Get the actual bounding box coordinates
            x = int(bbox[1] * cols)
            y = int(bbox[0] * rows)
            right = int(bbox[3] * cols)
            bottom = int(bbox[2] * rows)
            
            # Show the class name and the confidence
            cv2.putText(img, "{} {:.2f}%".format(class_name, score*100), (x, bottom+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
            
            # Draw the bounding box
            cv2.rectangle(img, (x, y), (right, bottom), (125, 255, 51), thickness = 1)
 
cv2.imshow('img', img)
cv2.waitKey(0)

