import cv2

path = r'C:\Users\luubi\Documents\PythonCode\Datasets\one\372008355484671519_001.jpg'
img = cv2.imread(path,0)
w,h = img.shape
print(w , h)
for i in range(0,w):
    for j in range(0,h):
        point = img[i][j]
        if 255-point < 150:
            img[i][j] = 0
        else:
            img[i][j] = img[i][j] + 100

cv2.imshow('img',img)


cv2.waitKey()










