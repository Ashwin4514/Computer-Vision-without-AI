import cv2
import pytesseract
from pytesseract import Output
from PIL import Image 

#Mentioning env variable to use tesseract engine for OCR.
pytesseract.pytesseract.tesseract_cmd= r''

#Reading the Image and resizing it 
image= cv2.imread("fineMind/3642684.jpg")
image_2= cv2.imread("fineMind/3648476-1.jpg")
image_3= cv2.imread("fineMind/3657993.jpg")


#preprocessing the image to detect text from that image.
image_gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image_gray_2= cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY)
image_gray_3= cv2.cvtColor(image_3,cv2.COLOR_BGR2GRAY)

#comverting the image to grayscale and then removing noise using gaussian blur.

image_gray= cv2.GaussianBlur(image_gray,(5,5),1)
image_gray_2= cv2.GaussianBlur(image_gray_2,(5,5),1)
image_gray_3= cv2.GaussianBlur(image_gray_3,(5,5),1)


#trying to make a database to identify the relevant information
d= pytesseract.image_to_data(image_gray,output_type=Output.DICT)
d_1= pytesseract.image_to_data(image_gray_2,output_type=Output.DICT)
#image_gray_3_rotate= image_gray_3.transpose (Image.ROTATE_90)
d_2= pytesseract.image_to_data(image_gray_3,output_type=Output.DICT)

#Showing the Original Image
image= cv2.resize(image,(500,500),fx=0.5,fy=0.5)        
cv2.imshow("Original Image-1",image)
image_2= cv2.resize(image_2,(500,500),fx=0.5,fy=0.5)        
cv2.imshow("Original Image-2",image_2)
image_3= cv2.resize(image_3,(500,500),fx=0.5,fy=0.5)        
cv2.imshow("Original Image-3",image_3)
#printing the required  text
text= pytesseract.image_to_string(image_gray)
print("For Original Image-1: \n")
print("PAN number:")
print(d["text"][41])
print("Date of Birth:")
print(d["text"][82])
print("\n For Original Image-2: \n")
print("PAN number:")
print(d_1["text"][38])
print("Date of Birth:")
print(d_1["text"][76])
print("\n For Original Image-3: \n")
print(d_2["text"])

cv2.waitKey(0)
