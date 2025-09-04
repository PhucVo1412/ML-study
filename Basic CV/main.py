import cv2

image = cv2.imread('./hum.png')
converted_image = cv2.cvtColor(image,  cv2.COLOR_RGB2HSV)

# cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('hum1.png', converted_image)

combined = cv2.hconcat([image, converted_image])
cv2.imshow('Comparison', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()