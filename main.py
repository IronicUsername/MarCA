from PIL import ImageGrab
import numpy as np
import time
import cv2
import win32gui

font = cv2.FONT_HERSHEY_SIMPLEX
face_cascade = cv2.CascadeClassifier('./harr_cascades/haarcascade_frontalface_default.xml')
face_cascade_alt = cv2.CascadeClassifier('./harr_cascades/haarcascade_frontalface_alt.xml')
face_cascade_alt2 = cv2.CascadeClassifier('./harr_cascades/haarcascade_frontalface_alt2.xml')
face_profile = cv2.CascadeClassifier('./harr_cascades/haarcascade_profileface.xml')
face_cascade_smile = cv2.CascadeClassifier('./harr_cascades/haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier('./harr_cascades/haarcascade_eye.xml')
upperbody_cascade = cv2.CascadeClassifier('./harr_cascades/haarcascade_upperbody.xml')
lowerbody_cascade = cv2.CascadeClassifier('./harr_cascades/haarcascade_lowerbody.xml')

imgTemplate = cv2.imread('./img/car-suv.jpg', 0)
w, h = imgTemplate.shape[::-1]


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def findWindow(windowName):
    try:
        window = win32gui.GetWindowRect(win32gui.FindWindow(None, windowName))
        return window
    except Exception as e:
        print('An error occurred:', e)


def detectFace(img):
    faces = face_cascade_alt.detectMultiScale(img, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  # rot
        cv2.putText(img, 'Face', (x-30, y-5), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        roi_color = img[y: y+h, x: x+w]
        eyes = eye_cascade.detectMultiScale(roi_color)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew-5, ey+eh-5), (0, 255, 0), 2)  # gruen
            cv2.putText(roi_color, 'Eye', (ex, ey-5), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


def detectCarType(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, imgTemplate, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)


def increaseBrightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def main():
    last_time = time.time()
    while(True):
        observingWindow = findWindow('faces - Google Search - Mozilla Firefox')
        # observingWindow = findWindow(
        #     'UAI_DE - VS10134.AUTOBAHNINKASSO.DE - Remotedesktopverbindung')
        screen = np.array(ImageGrab.grab(bbox=(observingWindow)))
        # screen = increaseBrightness(screen, value=50)
        detectCarType(screen)
        detectFace(screen)
        print('Loop took {} seconds'.format(time.time()-last_time))

        cv2.imshow('screen', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        last_time = time.time()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()

'''
cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    masked_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked_img = cv2.Canny(masked_img, threshold1=200, threshold2=300)
    masked_img = cv2.GaussianBlur(masked_img, (3, 3), 0)
    vertices = np.array([[0, 1920], [0, 0], [1080, 0], [1920, 1080]], np.int32)
    masked_img = roi(masked_img, [vertices])

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    profiles = face_profile.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)  # rot
        cv2.putText(img, 'Face', (x-30, y-5), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        roi_gray = gray[y: y+h, x: x+w]
        roi_color = img[y: y+h, x: x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew-5, ey+eh-5), (0, 255, 0), 2)  # gruen
            cv2.putText(roi_color, 'Eye', (ex, ey-5), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    for (xp, yp, wp, hp) in profiles:
        cv2.rectangle(img, (xp, yp), (xp+wp, yp+hp), (255, 0, 0), 2)  # blau

    cv2.imshow('img', img)
    cv2.imshow('cv', masked_img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    for (xx, yy, ww, hh) in profiles:
        cv2.rectangle(img, (xx, yy), (xx+ww, yy+hh), (255, 0, 0), 2)
        roi_gray = gray[yy:yy+hh, xx:xx+ww]
        roi_color = img[yy:yy+hh, xx:xx+ww]
        eyes = eye_cascade.detectMultiScale(roi_gray)
'''
