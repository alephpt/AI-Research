import cv2
import threading

cams = set()


class Camera(threading.Thread):
    def __init__(self, ID):
        threading.Thread.__init__(self)
        self.handle = f"Camera {ID}"
        self.cameraID = ID

    def run(self):
        print(f"Starting camera {self.handle}.")

        cv2.namedWindow(self.handle)
        cam = cv2.VideoCapture(self.cameraID)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if self.cam and self.cam.isOpened():
            ret, frame = self.cam.read()
        else:
            ret = None

        while ret:
            ret, frame = self.cam.read()
            cv2.imshow(self.handle, frame)
            key = cv2.waitKey(20)
            if key in [27, ord('Q'), ord('q')]:
                ret = None
                break

        print(f"Stopping camera {self.handle}.")
        cv2.destroyWindow(self.handle)
        cam.release()


def main():
    # get the number of webcams from the user
    num = int(input("Enter the number of webcams: "))

    # get all webcams and print their names
    for i in range(num):
        cams.add(Camera(i))

    for cam in cams:
        cam.start()

    # if the user presses esc, while the threads are running, stop all webcams and exit
    # while True:


if __name__ == "__main__":
    main()
