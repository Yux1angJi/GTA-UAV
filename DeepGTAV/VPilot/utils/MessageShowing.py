mapStencilBuffer = {1: 20, 2: 60, 3: 100, 7: 140, 16: 180, 248: 220, 255: 255}

def showStencilBuffer(idx):
    message = messages[idx]
    if message["StencilBuffer"] != None and message["StencilBuffer"] != "":
        # print(message["occlusionImage"])
        nparr = np.fromstring(base64.b64decode(message["StencilBuffer"]), np.uint8)
        
        # np.unique(nparr, return_counts=True)

        nparr = np.array([mapStencilBuffer[i] for i in nparr])

        nparr = nparr.reshape((1080, 1920))
        # nparr = nparr * 1
        # nparr = 255 - nparr * 3
        nparr = cv2.convertScaleAbs(nparr)
        # img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
        cv2.imshow("StencilBuffer", nparr)
        cv2.waitKey(-1)


def showWaterBuffer(idx):
    # TODO 
    pass
    

# Manual showing of message
def showmessage(idx):
    message = messages[idx]

    bboxes = parseBBoxesVisDroneStyle(message["bbox2d"])
    bbox_image = add_bboxes(frame2numpy(message['frame'], (IMG_WIDTH,IMG_HEIGHT)), parseBBox_YoloFormatStringToImage(bboxes))
    
    nparr = np.fromstring(base64.b64decode(message["instanceSegmentationImageColor"]), np.uint8)
    segmentationImage = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    
    dst = cv2.addWeighted(bbox_image, 0.5, segmentationImage, 0.5, 0.0)
    # plt.figure(figsize=(15,15))
    # plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    # plt.show()
    cv2.imshow("CombinedImage", dst)
    cv2.waitKey(1)

def showSegmentationImage(idx):
    message = messages[idx]
    nparr = np.fromstring(base64.b64decode(message["instanceSegmentationImageColor"]), np.uint8)
    segmentationImage = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    cv2.imshow("SegmentationImage", segmentationImage)
    cv2.waitKey(1)


def showBBox(idx):
    message = messages[idx]
    bboxes = parseBBoxesVisDroneStyle(message["bbox2d"])
    bbox_image = add_bboxes(frame2numpy(message['frame'], (IMG_WIDTH,IMG_HEIGHT)), parseBBox_YoloFormatStringToImage(bboxes))
    cv2.imshow("BBoxImage", bbox_image)
    cv2.waitKey(1)

def showOcclusionImage(idx):
    message = messages[idx]
    if message["occlusionImage"] != None and message["occlusionImage"] != "":
        # print(message["occlusionImage"])
        nparr = np.fromstring(base64.b64decode(message["occlusionImage"]), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
        cv2.imshow("occlusionImage", img)
        cv2.waitKey(1)

def showStencil1(idx):
    message = messages[idx]
    if message["segmentationImage"] != None and message["segmentationImage"] != "":
        # print(message["occlusionImage"])
        nparr = np.fromstring(base64.b64decode(message["segmentationImage"]), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
        cv2.imshow("segmentationImage", img)
        cv2.waitKey(1)


def ShowUsefulMessages():
    for idx in range(len(messages)):
        message = messages[idx]
        if message["bbox2d"] != None:
            print(idx)



# For parsing the stencil message with individual stencil values
# stencilMessage = stencilMessage[8:]

def parseMultipleStencil(stencilMessage):
    msgs = stencilMessage.split("StencilVal")
    vals = [msg[:2] for msg in msgs]
    imgs = [msg[2:] for msg in msgs]


    nparr = np.fromstring(base64.b64decode(imgs[1]), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    cv2.imshow("img1", img)
    cv2.waitKey(1)

    nparr = np.fromstring(base64.b64decode(imgs[2]), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    cv2.imshow("img2", img)
    cv2.waitKey(1)

    nparr = np.fromstring(base64.b64decode(imgs[3]), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    cv2.imshow("img3", img)
    cv2.waitKey(3)

    nparr = np.fromstring(base64.b64decode(imgs[4]), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    cv2.imshow("img4", img)
    cv2.waitKey(4)

    nparr = np.fromstring(base64.b64decode(imgs[5]), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    cv2.imshow("img5", img)
    cv2.waitKey(5)

    nparr = np.fromstring(base64.b64decode(imgs[6]), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    cv2.imshow("img6", img)
    cv2.waitKey(6)


cv2.destroyAllWindows()
