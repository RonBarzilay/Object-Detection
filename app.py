# required packages: versions
# pip install tensorflow: 2.5.0
# pip install opencv-python: 4.5.2.52
# pip install keras-nightly	2.5.0.dev2021032900
# pip install imageai	2.1.5	2.1.6
# in order to run:
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

# For measuring time elapsed
import time

# From library imageai.Detection take class ObjectDetection
from imageai.Detection import ObjectDetection
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


detector = ObjectDetection()

model_path = "./models/yolo-tiny.h5"
input_path = "./input/checkImage.jpg"
output_path = "./output/newImage.jpg"

detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel("normal")

start = time.time()
detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path,
                                            minimum_percentage_probability=10)
end = time.time()

print("It took ", end - start, "Seconds to find all objects")

for index, eachItem in enumerate(detection):
    print(index+1, ": ")
    print(eachItem["name"], " : ", eachItem["percentage_probability"])