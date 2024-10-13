import cv2
import numpy as np

# Read the image
image_path = "Task 1\\IMG-0044-00001.jpg"
image = cv2.imread(image_path)

# Convert the image from BGR to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the HSV range for the blue color
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

# Create a mask using the HSV range
mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Optionally, use dilation to enhance the mask
kernel = np.ones((5, 5), np.uint8)
dilated_mask = cv2.dilate(mask, kernel, iterations=1)

# Apply the mask to extract the blue circle
result = cv2.bitwise_and(image, image, mask=dilated_mask)

cv2.imwrite("Extracted.jpg", result)

# Display the original image and the result
cv2.imshow("Original Image", image)
cv2.imshow("Extracted Blue Circle", result)
cv2.waitKey(0)
cv2.destroyAllWindows()