import numpy as np
import cv2


class UI:
    def __init__(self, windowName=None) -> None:
        self.frameSize = 512
        self.displayFrame = np.zeros((self.frameSize, 3*self.frameSize, 3), dtype=np.uint8)
        self.windowName = windowName
        if windowName is None:
            self.windowName = "TEST"
    
    def display(self, textList, frame1, frame2):

        self.displayFrame[:, 512:2*512, :] = frame1
        self.displayFrame[:, 2*512:3*512, :] = frame2
        self.displayFrame[:, :512, :] = 0
        for i in range(len(textList)):
            cv2.putText(self.displayFrame, textList[i], (0, 50*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        
        cv2.imshow(self.windowName, self.displayFrame)