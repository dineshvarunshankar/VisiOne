import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

root = os.getcwd()
videoPath = os.path.join(root, "walk1.mp4")
videoCapObj = cv.VideoCapture(videoPath)
# Sparse Lucas–Kanade flow
def lucasKanade(videoCapObj):

    shiTomasiCornerParams = dict(maxCorners=20, qualityLevel=0.3, minDistance=50, blockSize=7)
    lucasKanadeParams = dict(winSize=(15, 15), maxLevel=2,
                             criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    randomColors = np.random.randint(0, 255, (100, 3))

    ret, frameFirst = videoCapObj.read()
    if not ret:
        print("Video not found")
        return

    frameGrayPrev = cv.cvtColor(frameFirst, cv.COLOR_BGR2GRAY)
    cornersPrev = cv.goodFeaturesToTrack(frameGrayPrev, mask=None, **shiTomasiCornerParams)
    mask = np.zeros_like(frameFirst)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = 30
    height, width, _ = frameFirst.shape
    videoWriter = cv.VideoWriter('lucas_kanade_output.mp4', fourcc, fps, (width, height))

    while True:
        ret, frame = videoCapObj.read()
        if not ret:
            break

        frameGrayCur = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        cornersCur, foundStatus, _ = cv.calcOpticalFlowPyrLK(
            frameGrayPrev, frameGrayCur, cornersPrev, None, **lucasKanadeParams
        )

        if cornersCur is not None:
            cornersMatchedCur = cornersCur[foundStatus == 1]
            cornersMatchedPrev = cornersPrev[foundStatus == 1]

            for i, (curCorner, prevCorner) in enumerate(zip(cornersMatchedCur, cornersMatchedPrev)):
                xCur, yCur = curCorner.ravel()
                xPrev, yPrev = prevCorner.ravel()

                cv.line(mask, (int(xCur), int(yCur)),
                        (int(xPrev), int(yPrev)),
                        randomColors[i].tolist(), 2)
                cv.circle(frame, (int(xCur), int(yCur)),
                          5, randomColors[i].tolist(), -1)

            img = cv.add(frame, mask)

            cv.imshow("Lucas-Kanade Optical Flow", img)

            videoWriter.write(img)

            frameGrayPrev = frameGrayCur.copy()
            cornersPrev = cornersMatchedCur.reshape(-1, 1, 2)

        if cv.waitKey(15) & 0xFF == ord('q'):
            break

    videoWriter.release()

    videoCapObj.release()
    cv.destroyAllWindows()



# Dense Farnebäck flow
def denseOpticalFlow(videoCapObj):

    ret, frameFirst = videoCapObj.read()
    if not ret:
        print("Failed to load video")
        return

    imgPrev = cv.cvtColor(frameFirst, cv.COLOR_BGR2GRAY)

    imgHSV = np.zeros_like(frameFirst)
    imgHSV[..., 1] = 255

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = 30  # Output frame rate
    height, width, _ = frameFirst.shape
    videoWriter = cv.VideoWriter('dense_optical_flow.mp4', fourcc, fps, (width, height))

    while True:
        ret, frameCur = videoCapObj.read()
        if not ret:
            break

        imgCur = cv.cvtColor(frameCur, cv.COLOR_BGR2GRAY)

        flow = cv.calcOpticalFlowFarneback(
            prev=imgPrev,
            next=imgCur,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN
        )

        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Hue encodes flow direction.
        imgHSV[..., 0] = ang * 180 / np.pi / 2

        # Brightness encodes flow magnitude.
        imgHSV[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        imgBGR = cv.cvtColor(imgHSV, cv.COLOR_HSV2BGR)

        cv.imshow("Dense Optical Flow (Farneback)", imgBGR)

        videoWriter.write(imgBGR)

        imgPrev = imgCur

        if cv.waitKey(15) & 0xFF == ord('q'):
            break

    videoWriter.release()

    cv.destroyAllWindows()



if __name__ == "__main__":
    denseOpticalFlow(videoCapObj)
