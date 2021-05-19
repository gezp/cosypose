from PIL import Image
import numpy as np
from cosypose.cosypose_detector import CosyposeDetector

if __name__ == "__main__":
    #image
    path="test.png"
    img = np.array(Image.open(path))
    #detector
    camera_k = np.array([[585.75607,    0,      320.5 ],\
                        [  0,      585.75607, 240.5],\
                        [  0,        0,        1,     ]])
    detector = CosyposeDetector(camera_k=camera_k)
    input_msg="#1:detect all,2:detect obj_000007##"
    while(1):
        mode = input(input_msg)
        mode = int(mode)
        if mode == 1:
            results =detector.detect_all(img)
            if len(results) !=0:
                detector.print(results)
            else:
                print("not detected")
        elif mode == 2:
            results = detector.detect(img,"obj_000007")
            if len(results) !=0:
                detector.print(results)
            else :
                print("not detected")
    rclpy.shutdown()     