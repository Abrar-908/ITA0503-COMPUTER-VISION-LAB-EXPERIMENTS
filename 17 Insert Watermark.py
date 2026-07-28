import cv2

# Read image
image = cv2.imread("sample.jpg")

# Watermark text
text = "© ABRAR"

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
position = (30, 50)
font_scale = 1
color = (255, 255, 255)   # White
thickness = 2

watermarked = cv2.putText(image, text, position, font,
                           font_scale, color, thickness, cv2.LINE_AA)

# Show
cv2.imshow("Original Image", image)
cv2.imshow("Watermarked Image", watermarked)

cv2.waitKey(0)
cv2.destroyAllWindows()