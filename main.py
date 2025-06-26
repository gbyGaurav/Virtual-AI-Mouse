import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

# Webcam and screen settings
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction for active region
smoothening = 7

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Hand detector
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
print(f"Screen size: {wScr}x{hScr}")

while True:
    success, img = cap.read()
    if not success:
        continue

    try:
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if len(lmList) >= 13:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            fingers = detector.fingersUp()
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

            # Moving Mode: Index finger up
            if fingers[1] == 1 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # Clamp mouse movement to screen bounds
                x_mouse = max(0, min(wScr - 1, wScr - clocX))
                y_mouse = max(0, min(hScr - 1, clocY))

                print(f"Moving to: ({int(x_mouse)}, {int(y_mouse)})")
                autopy.mouse.move(int(x_mouse), int(y_mouse))
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

                plocX, plocY = clocX, clocY

            # Clicking Mode: Index and middle fingers up
            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineInfo = detector.findDistance(8, 12, img)
                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    print("Click!")
                    autopy.mouse.click()

    except Exception as e:
        print("Error:", e)

    # FPS Display
    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime - pTime != 0 else 0
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("AI Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
