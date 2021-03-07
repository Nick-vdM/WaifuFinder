import pickle
import random

import colorthief
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def get_palette(image, color_count=10, quality=10):
    """This is from the colour thief library. For some reason they decided it should only be possible to load in images
    from the file path instead of cv2 imread, so I ripped out their function and put it into here. It'll probably
    get marked for plagiarism"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    image = Image.fromarray(image)
    image = image.convert('RGBA')

    width, height = image.size
    pixels = image.getdata()
    pixel_count = width * height
    valid_pixels = []
    for i in range(0, pixel_count, quality):
        r, g, b, a = pixels[i]
        # If pixel is mostly opaque and not white
        if a >= 125:
            if not (r > 250 and g > 250 and b > 250):
                valid_pixels.append((r, g, b))

    # Send array to quantize function which clusters values
    # using median cut algorithm
    cmap = colorthief.MMCQ.quantize(valid_pixels, color_count)
    return cmap.palette


def displayImage(image):
    cv2.imshow("title", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def loadFromPickle(fileName):
    file = open(fileName, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def saveToPickle(data, fileName):
    f = open(fileName, 'wb')
    pickle.dump(data, f)
    f.close()


class faceInformation:
    """
    This extracts important information about a face all into a class - specifically:
    * The hair colour
    * The eye colour
    * The variables of a sift operator
    Essentially the idea is that all anime faces are processed with this beforehand then we can throw all
    this information into a dataset to look through quickly

    The difference between the human and the anime face is that face cropping algorithms and eye recognition algorithms
    work differently. Since our haar and lbp sets are specifically trained for human-like faces, they don't
    function as well in the anime environment.

    Our anime face just grabs the three most common quantization colours and assumes the first is the skin.
    The second is the hair and the third is the eyes
    """

    def __init__(self):
        self.ID = 0
        self.hairColour = (0, 0, 0)
        self.eyeColour = (0, 0, 0)
        self.edgeSift = None

    def processHumanFace(self, imagePath, imageID):
        """Initialises all the self variables to access from the outside"""
        self.filePath = imagePath
        imageWithFace = cv2.imread(self.filePath, 1)
        self.ID = imageID  # Save the identifier
        # Crop the face out
        grayImageWithFace = cv2.cvtColor(imageWithFace, cv2.COLOR_RGB2GRAY)
        faceRect = self._findFaceCoordinates(grayImageWithFace)
        if len(faceRect) == 0:
            return
        face = self._cropImage(imageWithFace, faceRect)
        grayFace = self._cropImage(grayImageWithFace, faceRect)

        # Get eye colour
        eyesRect = self._findEyeCoordinates(grayImageWithFace)
        eye = self._cropImage(imageWithFace, eyesRect)
        if len(eye) > 0:
            self.eyeColour = self._getEyeColour(eye)
        else:
            self.eyeColour = (0, 0, 0)

        # hair colour
        self.hairColour = self._getHairColour(face)

        # Grab the edges with hough lines
        edges = cv2.Canny(grayFace, 100, 200)

        # Grab sift points
        sift = cv2.xfeatures2d.SIFT_create()
        self.edgeSift = \
            sift.detectAndCompute(edges, None)

    def processAnimeFace(self, imagePath, imageID):
        self.ID = imageID
        self.filePath = imagePath

        imageWithFace = cv2.imread(self.filePath, 1)
        grayFace = cv2.cvtColor(imageWithFace, cv2.COLOR_RGB2GRAY)
        imageWithFace = self.cropAnimeFace(imageWithFace, grayFace)

        ct = colorthief.ColorThief(imagePath)
        palette = ct.get_palette(color_count=3)

        self.hairColour = palette[1]
        self.eyeColour = palette[2]
        edges = cv2.Canny(grayFace, 100, 200)
        sift = cv2.xfeatures2d.SIFT_create()

        self.edgeSift = sift.detectAndCompute(edges, None)

    def cropAnimeFace(self, imageWithFace, grayImage):
        face_cascade = cv2.CascadeClassifier('tools/lbpcascade_animeface.xml')
        rects = face_cascade.detectMultiScale(grayImage, 1.1, 4)
        if len(rects) == 0:
            return imageWithFace  # no face was found so just use the whole image
        imageWithFace = self._cropImage(imageWithFace, rects[0])
        return imageWithFace

    def foldKeyPoints(self):
        """Since keypoints can't be pickled we have to 'fold' them before we pickle them"""
        folded = [[], self.edgeSift[1]]
        for keyPoint in self.edgeSift[0]:
            folded[0].append((keyPoint.pt, keyPoint.size,
                              keyPoint.angle, keyPoint.response,
                              keyPoint.octave,
                              keyPoint.class_id))
        self.edgeSift = folded

    def unfoldKeyPoints(self):
        unfoldedSiftOutput = [[], self.edgeSift[1]]
        for keypoint in self.edgeSift[0]:
            unfoldedSiftOutput[0].append(cv2.KeyPoint(x=keypoint[0][0], y=keypoint[0][1], _size=keypoint[1],
                                                      _angle=keypoint[2],
                                                      _response=keypoint[3], _octave=keypoint[4],
                                                      _class_id=keypoint[5]))
        self.edgeSift = unfoldedSiftOutput

    def foundFace(self):
        """If the face variable was set to empty, no face was found"""
        return len(self.face) == 0

    @staticmethod
    def _findFaceCoordinates(grayImage):
        """
        Returns the coordinates of where a face is in the image
        Method is grabbed from https://sonsuzdesign.blog/2020/06/28/simple-face-detection-in-python/
        :param grayImage: the image of a face in a gray image format
        :return: the coordinates: [top left, bottom right] [(y, x), (y, x)}
        """
        face_cascade = cv2.CascadeClassifier('tools/haarcascade_frontalface_default.xml')
        rects = face_cascade.detectMultiScale(grayImage, 1.1, 4)
        if len(rects) == 0:
            print("Returning no face")
            return []  # let them know there's nothing
        return rects[0]

    @staticmethod
    def _findEyeCoordinates(grayImage):
        eyes_cascade = cv2.CascadeClassifier('tools/haarcascade_eye.xml')
        rects = eyes_cascade.detectMultiScale(grayImage, 1.1, 4)
        print("Eye rects", rects)
        if len(rects) == 0:
            return []
        # we really don't need to detect more than one eye
        return rects[0]

    @staticmethod
    def _cropImage(image, rect):
        """
        Just a function for outside readability
        :param image:
        :param rect:
        :return:
        """
        if len(rect) == 0:
            return np.array([])
        (x, y, w, h) = rect
        return image[y:y + h, x:x + w]

    @staticmethod
    def _getEyeColour(cropOfEye):
        # Blur it a bit
        g_cropOfEye = cv2.cvtColor(cropOfEye, cv2.COLOR_RGB2GRAY)
        g_cropOfEye = cv2.medianBlur(g_cropOfEye, 13)
        circles = cv2.HoughCircles(g_cropOfEye, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=0, maxRadius=0)
        if circles is not None:
            # We found the iris and can make it smaller
            circles = np.uint16(np.around(circles))

            # Granted the pupil is black and the white of the eyes is white we should just be able to iterate
            # over a square of this and set the colour to the named version of the eye colour
            circle = circles[0][0]
            cropOfEye = cropOfEye[circle[1] - circle[2]:circle[1] + circle[2],
                        circle[0] - circle[2]:circle[0] + circle[2]]

        return get_palette(cropOfEye, 5, 10)[0]  # most dominant colour from quantization

    @staticmethod
    def _getHairColour(croppedFace):
        """We're just going to grab the top 10% of the face and calculate the average colour"""
        topOfFace = croppedFace[0:croppedFace.shape[1] // 10, 0:croppedFace.shape[1]]
        return get_palette(topOfFace, 5, 10)[0]  # most dominant colour from quantization


def compareSifts(output1, output2, bf_Matcher):
    """Outputs should be (kp, des) so we can store them lazily"""
    kp1 = output1[0]
    des1 = output1[1]
    kp2 = output2[0]
    des2 = output2[1]
    numberOfKeypoints = min(len(kp1), len(kp2))
    try:
        matches = bf_Matcher.knnMatch(des1, des2, k=2)
    except:
        print("Type error thing happened")
        return 0.0

    good_matches = []
    for i, j in matches:
        if i.distance < 0.75 * j.distance:
            # We consider them to be good matches if
            good_matches.append([i])

    # fraction of what are good matches
    if numberOfKeypoints == 0:
        return 0
    return len(good_matches) / numberOfKeypoints


def generateOverallRanks(edgeSiftRankings, eyeColourRankings, hairColourRankings):
    ranks = {}  # ID : sumOfRanks -> lowest rank is
    # They should all be the same len so
    print(len(edgeSiftRankings), len(eyeColourRankings), len(hairColourRankings))
    for i in range(len(eyeColourRankings)):
        if eyeColourRankings[i][0] in ranks:
            ranks[eyeColourRankings[i][0]] += i
        else:
            ranks[eyeColourRankings[i][0]] = i

        if hairColourRankings[i][0] in ranks:
            ranks[hairColourRankings[i][0]] += i
        else:
            ranks[hairColourRankings[i][0]] = i

        if edgeSiftRankings[i][0] in ranks:
            ranks[edgeSiftRankings[i][0]] += i
        else:
            ranks[edgeSiftRankings[i][0]] = i
    return sorted(ranks, key=ranks.__getitem__)  # change us into a sorted list


def tupleDifference(tuple1, tuple2):
    colourDifference = 0
    for i in range(len(tuple1)):
        colourDifference += (tuple1[i] - tuple2[i]) ** 2
    colourDifference = np.sqrt(colourDifference)
    return colourDifference


def calculateRankings(faceData, mainFace):
    eyeColourRankings = []
    hairColourRankings = []
    edgeSiftRankings = []
    bf_matcher = cv2.BFMatcher()
    index = 0
    onePercentFaceLen = len(faceData) // 100
    for face in faceData:
        if index % onePercentFaceLen == 0:
            print(str(index // onePercentFaceLen) + " Percent done")
        index += 1
        eyeColourRankings.append(
            (
                face.ID,
                tupleDifference(mainFace.eyeColour, face.eyeColour)
            )
        )
        hairColourRankings.append(
            (
                face.ID,
                tupleDifference(mainFace.eyeColour, face.eyeColour)
            )
        )

        edgeSiftRankings.append(
            (
                face.ID,
                compareSifts(mainFace.edgeSift, face.edgeSift, bf_matcher)
            )
        )
    # Now we go ahead and sort each one accordingly
    eyeColourRankings = sorted(eyeColourRankings, key=lambda i: i[1])
    hairColourRankings = sorted(eyeColourRankings, key=lambda i: i[1])
    edgeSiftRankings = sorted(eyeColourRankings, key=lambda i: i[1])
    return edgeSiftRankings, eyeColourRankings, hairColourRankings


def searchKClosestFace(mainFace, faceData, K=10):
    # Would use numpy, but they are /arrays/, which suck at dynamically expanding
    edgeSiftRankings, eyeColourRankings, hairColourRankings = \
        calculateRankings(faceData, mainFace)
    ranks = generateOverallRanks(edgeSiftRankings, eyeColourRankings, hairColourRankings)
    topKFiles = []  # grab the file paths from the ids
    nameLookup = {f.ID: f.filePath for f in faceData}
    for i in range(K):
        topKFiles.append(nameLookup[ranks[i]])
    return topKFiles


def saveAllAnimeFaces(path, fileToSaveTo):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    faces = []
    processedFaces = 0
    filesLen = len(files)
    onePercent = filesLen // 100
    for f in files:
        if processedFaces % onePercent == 0:
            print(str(processedFaces // onePercent), "percent faces complete")
        try:
            filename = int(f[0:-4])  # cut out .png. Could break on .jpeg
        except:
            filename = int(f[0:-5])
        waifuImageData = faceInformation()
        waifuImageData.processAnimeFace(path + "/" + f, filename)
        waifuImageData.foldKeyPoints()
        faces.append(waifuImageData)
        processedFaces += 1
    saveToPickle(faces, fileToSaveTo)
    print("Done processing all the faces!")


def loadFaceInformation(filePath):
    """Loads in a face information path -> load in and unfold each"""
    faces = loadFromPickle(filePath)
    for f in faces:
        f.unfoldKeyPoints()
    return faces


def displayImages(mainImagePath, imagePathsList):
    """Show a group of images using pyplot"""
    f, spots = plt.subplots(1, 1 + len(imagePathsList), figsize=(10, 10))
    image = cv2.cvtColor(cv2.imread(mainImagePath, 1), cv2.COLOR_BGR2RGB)
    spots[0].imshow(image)
    spots[0].set_title("Main image")
    for i in range(1, len(spots)):
        plt.axis('off')
        image = cv2.cvtColor(cv2.imread(imagePathsList[i - 1], 1), cv2.COLOR_BGR2RGB)
        spots[i].imshow(image)
        spots[i].set_title(str(i) + " best image")
    plt.show()


def imagefigure(image):
    fig, ax = plt.subplots()
    ax.imshow(Image.open(image))
    return fig


def rectangleFigure(rgb, title):
    fig, ax = plt.subplots()
    ax.set_facecolor(
        (rgb[0] * float(1 / 255),
         rgb[1] * float(1 / 255),
         rgb[2] * float(1 / 255))
    )
    ax.set_title(title)
    return fig


def rateSystem(path, realPeople=False):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    random.shuffle(files)
    imageHairEye = []
    index = 0
    validImages = 0
    while validImages < 100:
        print(validImages, files[index])
        imageData = faceInformation()
        if realPeople:
            try:  # It seems the people dataset has issues
                imageData.processHumanFace(path + "/" + files[index], "doesn't matter")
            except:
                index += 1
                continue
        else:
            imageData.processAnimeFace(path + "/" + files[index], "doesn't matter")
        imageHairEye.append(
            (path + "/" + files[index], imageData.hairColour, imageData.eyeColour)
        )
        validImages += 1
        index += 1

    pdf = PdfPages('ImageHairEyes.pdf')
    for i in range(len(imageHairEye)):
        f = imagefigure(imageHairEye[i][0])
        pdf.savefig(f)
        plt.close(f)

        f = rectangleFigure(imageHairEye[i][1], "hair")
        pdf.savefig(f)
        plt.close(f)

        f = rectangleFigure(imageHairEye[i][2], 'eyes')
        pdf.savefig(f)
        plt.close(f)
    pdf.close()


if __name__ == "__main__":
    pickleFilePath = input("Do you want to make a waifu pickle file? "
                           "Type the directory path\n>")
    if len(pickleFilePath) != 0:
        outputPath = input("Where do you want to save it?\n>")
        saveAllAnimeFaces(pickleFilePath, outputPath)

    humanface = input("What face do you want to try?\n>")
    print("You entered '" + humanface + "'")
    waifuPickle = input("Do you have a waifu pickle?")
    if len(waifuPickle) == 0:
        print("No? Okay I'll pick the default one")
        waifuPickle = 'data/waifuFaces.pickle'

    mainFace = faceInformation()
    mainFace.processHumanFace(humanface, 1)
    faceData = loadFaceInformation('data/waifuFaces.pickle')
    topFiles = searchKClosestFace(mainFace, faceData)
    displayImages(humanface, topFiles)
