import cv2
import mediapipe as mp
import time

class FaceDetectionModule:
    def __init__(self, detection_confidence=0.75):
        self.cap = cv2.VideoCapture(0)
        self.ptime = 0

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(detection_confidence)

    def detect_faces(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.faceDetection.process(imgRGB)
        h, w, c = frame.shape

        if results.detections:
            for id, detection in enumerate(results.detections):
                locs = detection.location_data.relative_bounding_box
                x = int(locs.xmin * w)
                y = int(locs.ymin * h)
                dx = int(locs.width * w)
                dy = int(locs.height * h)
                cv2.rectangle(frame, (x, y), (x + dx, y + dy), (0, 255, 0), 2)
                cv2.putText(frame, f'{int(detection.score[0] * 100)}%', (x, y - 5),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

        ctime = time.time()
        fps = 1 / (ctime - self.ptime)
        self.ptime = ctime
        cv2.putText(frame, f"{int(fps)} fps", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        return frame, fps

    def release_resources(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def start_detection(self):
        while True:
            frame, fps = self.detect_faces()

            if frame is None:
                break

            cv2.imshow("Face Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release_resources()

if __name__ == "__main__":
    detector = FaceDetectionModule()
    detector.start_detection()
