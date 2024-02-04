
import cv2
import matplotlib.pyplot as plt

print (cv2.__version__)

imagePath = "./imgs/random.jpg"
img = cv2.imread(imagePath)

print("Image shape:", img.shape)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('on')
plt.show()

# set dimension for cropping image
x, y, width, depth = 50, 10, 950, 500
image_cropped = img[y:(y+depth), x:(x+width)]

# create a copy of the cropped image to be used later
image_template = image_cropped.copy()

# convert image to Grayscale
image_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

# remove axes and show image
plt.axis("off")
plt.imshow(image_gray, cmap = "gray")

haarcascade = "./haarcascade_frontalface_default.xml"

# create an instance of the Face Detection Cascade Classifier
detector = cv2.CascadeClassifier(haarcascade)

# Detect faces using the haarcascade classifier on the "grayscale image"
faces = detector.detectMultiScale(image_gray)

# Print coordinates of detected faces
print("Faces:\n", faces)

for face in faces:
#     save the coordinates in x, y, w, d variables
    (x,y,w,d) = face
    # Draw a white coloured rectangle around each face using the face's coordinates
    # on the "image_template" with the thickness of 2
    cv2.rectangle(image_template,(x,y),(x+w, y+d),(255, 255, 255), 2)

plt.axis("off")
plt.imshow( cv2.cvtColor(image_template, cv2.COLOR_BGR2RGB))
plt.title('Face Detection')

LBFmodel = "./lbfmodel.yaml"

# Create an instance of the Facial landmark Detector with the model
landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

# Detect landmarks on "image_gray"
_, landmarks = landmark_detector.fit(image_gray, faces)

# Create a copy of the original image to draw landmarks and lines on
image_template = image_cropped.copy()

# Draw landmarks on "image_template"
for landmark in landmarks:
    for x, y in landmark[0]:
        # Display landmarks on "image_template" with white color in BGR and thickness 1
        cv2.circle(image_template, (int(x), int(y)), 1, (255, 255, 255), 1)


# Display the image with landmarks and lines
plt.axis("on")
plt.imshow( cv2.cvtColor(image_template, cv2.COLOR_BGR2RGB))
cv2.imwrite( "output.jpg",cv2.cvtColor(image_template, cv2.COLOR_BGR2RGB))
plt.title('Facial Landmarks with Lines')
plt.show()
