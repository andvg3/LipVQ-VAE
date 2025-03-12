import cv2
import numpy as np


def draw_arrows(
    image, start_points, direction, length=50, color=(0, 255, 0), thickness=2
):
    for start in start_points:
        end = (start[0] + direction[0] * length, start[1] + direction[1] * length)
        cv2.arrowedLine(image, start, end, color, thickness, tipLength=0.2)
    return image


# Load an image
image = cv2.imread("/home/anvuong/Desktop/query_2.jpg")

# Define start points near each other
start_points = [(359, 195), (365, 200), (370, 205), (375, 210), (384, 210)]

# Define direction (dx, dy) for arrows (rightward and slightly downward)
direction = (-2, 1)  # Rightward

# Draw arrows on the image
image = draw_arrows(image, start_points, direction)

# Show image
cv2.imshow("Annotated Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
