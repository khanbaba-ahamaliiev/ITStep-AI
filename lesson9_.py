import cv2
import ultralytics


model = ultralytics.YOLO("yolo11s-pose.pt")

image = cv2.imread("data/lesson_pose/human.jpg")
cv2.imshow("human", image)

results = model.predict(image)

result = results[0]

res_img = result.plot()

cv2.imshow("res_img", res_img)

keypoints = result.keypoints

xy = keypoints.xy
xy = xy.cpu().numpy()
xy = xy[0]
xy = xy.astype(int)

x_right_hand, y_right_hand = xy[10]
print(x_right_hand, y_right_hand)

cv2.circle(
    image,
    (x_right_hand, y_right_hand),
    15,
    color=(255, 0, 0),
    thickness=-1
)

cv2.imshow("result human", image)

cv2.putText(
    image,
    "right hand",
    (x_right_hand, y_right_hand - 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 0),
    2,

)

cv2.imshow("result human 2", image)

cv2.waitKey(0)