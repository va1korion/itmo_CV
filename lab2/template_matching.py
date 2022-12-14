import numpy as np
import imutils
import cv2

dir = "examples/rat/"
main = cv2.imread(dir+"modified.png")
template = cv2.imread(dir+"template.png")
(templateHeight, templateWidth) = template.shape[:2]

# ищем template на основном изображении,
result = cv2.matchTemplate(main, template, cv2.TM_SQDIFF)

# ищем минимум в хитмапе
(_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)


topLeft = minLoc
botRight = (topLeft[0] + templateWidth, topLeft[1] + templateHeight)
roi = main[topLeft[1]:botRight[1], topLeft[0]:botRight[0]]

mask = np.zeros(main.shape, dtype="uint8")

new_image = cv2.addWeighted(main, 0.5, mask, 1, 0)


# выделяем нужную нам область на изображении
new_image[topLeft[1]:botRight[1], topLeft[0]:botRight[0]] = roi

# display the images
cv2.imshow("Main_image", imutils.resize(new_image, height=650))
cv2.imshow("Template", template)
cv2.imwrite(dir+"template_matched.png", new_image)
cv2.waitKey(0)
