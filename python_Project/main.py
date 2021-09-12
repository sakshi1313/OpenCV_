import cv2
import os
import numpy as np

def empty(x):
    return x


# def stackImages(scale, imgArray):
#     rows = len(imgArray)
#     cols = len(imgArray[0])
#     rowsAvailable = isinstance(imgArray[0], list)
#     width = imgArray[0][0].shape[1]
#     height = imgArray[0][0].shape[0]
#     if rowsAvailable:
#         for x in range(0, rows):
#             for y in range(0, cols):
#                 if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
#                 else:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
#                                                 None, scale, scale)
#                 if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
#         imageBlank = np.zeros((height, width, 3), np.uint8)
#         hor = [imageBlank] * rows
#         hor_con = [imageBlank] * rows
#         for x in range(0, rows):
#             hor[x] = np.hstack(imgArray[x])
#         ver = np.vstack(hor)
#     else:
#         for x in range(0, rows):
#             if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
#                 imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
#             else:
#                 imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
#             if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
#         hor = np.hstack(imgArray)
#         ver = hor
#     return ver


# def hsv_val(img):
#
#     cv2.namedWindow("TrackBars")
#     cv2.resizeWindow("TrackBars", 640, 240)
#     cv2.createTrackbar("Hue Min", "TrackBars", 7, 179, empty)
#     cv2.createTrackbar("Hue Max", "TrackBars", 43, 179, empty)
#     cv2.createTrackbar("Sat Min", "TrackBars", 56, 255, empty)
#     cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
#     cv2.createTrackbar("Val Min", "TrackBars", 61, 255, empty)
#     cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)
#
#
#     imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     while True:
#         h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
#         h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
#         s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
#         s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
#         v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
#         v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
#         #print(h_min, h_max, s_min, s_max, v_min, v_max)
#         lower = np.array([h_min, s_min, v_min])
#         upper = np.array([h_max, s_max, v_max])
#         mask = cv2.inRange(imgHSV, lower, upper)
#         imgResult = cv2.bitwise_and(img, img, mask=mask)
#         kernel = np.ones((7, 7), np.uint8)
#         img_erosion2 = cv2.erode(mask, kernel, iterations=2)
#         imgStack = stackImages(0.6, ([img, imgHSV], [img_erosion2, imgResult]))
#
#         cv2.namedWindow("Stacked Image", cv2.WINDOW_NORMAL)
#         cv2.imshow("Stacked Image", imgStack)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
file_name=[]

def input_images(folder):
    images=[]

    for filename in os.listdir(folder):
        img=cv2.imread(os.path.join(folder,filename))
        file_name.append(filename)
        if img is not None:
            images.append(img)
    return images

def cordinate(n):
    x_arr = n[::2]
    x= sum(x_arr)
    x = x / len(x_arr)

    y_arr = n[1::2]
    y = sum(y_arr)
    y = y / len(y_arr)

    return x,y



img2=input_images("D:\ML\pythonProject1\input_images")
i=0
for i in range(len(img2)):
    # hsv_val(img2[i])    To get the hsv values of colors
    arr=[]
    img=img2[i]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # yellow
    l_y = np.array([25, 50, 70], np.uint8)
    u_y = np.array([35, 255, 255], np.uint8)
    mask_y = cv2.inRange(hsv, l_y, u_y)
    kernel = np.ones((7, 7), np.uint8)
    img_erosion = cv2.erode(mask_y, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(img_erosion,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if (area > 800):
            approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True)
            cv2.drawContours(img, [approx], 0, (0, 0, 255), 5)
            n = approx.ravel()
            x_cor,y_cor=cordinate(n)
            arr.append([int(x_cor),int(y_cor),1])

            # print(n)
            # print("contour done")

            cv2.putText(img, "Y", (int(x_cor), int(y_cor)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 0))


    # orange
    l_o = np.array([10, 50, 70], np.uint8)
    u_o = np.array([24, 255, 255], np.uint8)
    mask_o = cv2.inRange(hsv, l_o, u_o)
    kernel = np.ones((7, 7), np.uint8)
    img_erosion = cv2.erode(mask_o, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(img_erosion,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 0, 0), 3)

    for contour in contours:
        area = cv2.contourArea(contour)
        if (area > 800):

            approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True)
            cv2.drawContours(img, [approx], 0, (0, 0, 255), 5)
            n = approx.ravel()
            x_cor,y_cor=cordinate(n)
            arr.append([int(x_cor), int(y_cor),2])
            cv2.putText(img, "O", (int(x_cor), int(y_cor)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 0))

    # red block
    l_r = np.array([0, 50, 70], np.uint8)
    u_r = np.array([9, 255, 255], np.uint8)
    mask_r = cv2.inRange(hsv, l_r, u_r)
    kernel = np.ones((7, 7), np.uint8)
    img_erosion = cv2.erode(mask_r, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(img_erosion,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 0, 0), 3)

    for contour in contours:
        area = cv2.contourArea(contour)
        if (area > 800):


            approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True)
            cv2.drawContours(img, [approx], 0, (0, 0, 255), 5)
            n = approx.ravel()
            x_cor, y_cor = cordinate(n)
            arr.append([int(x_cor),int(y_cor),3])


            cv2.putText(img, "R", (int(x_cor), int(y_cor)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 0))

    # green block

    l_g = np.array([36, 50, 70], np.uint8)
    u_g = np.array([89, 255, 255], np.uint8)
    mask_g = cv2.inRange(hsv, l_g, u_g)
    kernel = np.ones((7, 7), np.uint8)
    img_erosion = cv2.erode(mask_g, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(img_erosion,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 0, 0), 3)

    for contour in contours:
        area = cv2.contourArea(contour)
        if (area > 800):


            approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True)
            cv2.drawContours(img, [approx], 0, (0, 0, 255), 5)
            n = approx.ravel()
            x_cor, y_cor = cordinate(n)

            arr.append([int(x_cor), int(y_cor),4])

            cv2.putText(img, "G", (int(x_cor), int(y_cor)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 0))


    # blue block
    l_b = np.array([90, 50, 70], np.uint8)
    u_b = np.array([128, 255, 255], np.uint8)
    mask_b = cv2.inRange(hsv, l_b, u_b)
    res_b = cv2.bitwise_and(img, img, mask=mask_b)
    kernel = np.ones((7, 7), np.uint8)
    img_erosion = cv2.erode(mask_b, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(img_erosion,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(img, contours, -1, (0, 0, 0), 3)

    for contour in contours:
        area = cv2.contourArea(contour)
        if (area > 800):
            approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True)
            cv2.drawContours(img, [approx], 0, (0, 0, 255), 5)
            n = approx.ravel()
            x_cor,y_cor=cordinate(n)
            arr.append([int(x_cor), int(y_cor),5])

            # print(n)
            # print("contour done")

            cv2.putText(img, "B", (int(x_cor), int(y_cor)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 0))


    # white
    l_w = np.array([0, 0, 231], np.uint8)
    u_w = np.array([180, 0, 255], np.uint8)
    mask_w = cv2.inRange(hsv, l_w, u_w)
    kernel = np.ones((7, 7), np.uint8)
    img_erosion = cv2.erode(mask_w, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(img_erosion,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 0, 0), 3)

    for contour in contours:
        area = cv2.contourArea(contour)
        if (area > 800):

            approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True)
            cv2.drawContours(img, [approx], 0, (0, 0, 255), 5)
            n = approx.ravel()
            x_cor, y_cor = cordinate(n)
            arr.append([int(x_cor), int(y_cor),6])

            cv2.putText(img, "W", (int(x_cor), int(y_cor)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 0))

    value=sorted(arr , key=lambda k: k[1])
    first=value[0:3]
    second=value[3:6]
    third=value[6:9]

    for f,s,t in first,second,third:
        val_f= sorted(first, key=lambda k: k[0])
        val_s = sorted(second, key=lambda k: k[0])
        val_t = sorted(third, key=lambda k: k[0])
    final=val_f + val_s + val_t

    print(final)

    # print(value)
    print("*************************************************************")
    res=[]
    for val in final:
        res.append(val[2])

    file=open("D:\ML\pythonProject1\output\output_"+file_name[i][0:3]+".txt","w+")
    for j in range(3):
        file.write(str(res[j])+" ")
    file.write("\n")
    for j in range(3):
        file.write(str(res[3+j])+" ")
    file.write("\n")
    for j in range(3):
        file.write(str(res[6+j])+" ")
    file.close()

    cv2.namedWindow("detect", cv2.WINDOW_NORMAL)
    #cv2.imshow("detect", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
