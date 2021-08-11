from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

count = 0

def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])
    global count
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
    if np.abs(euler_angle[0])<=25 and np.abs(euler_angle[1])<=25 and np.abs(euler_angle[2])<=25:
        count = count+1

    return reprojectdst, euler_angle

K = [6.2500000000000000e+002, 0.0, 3.1250000000000000e+002,
     0.0, 6.2500000000000000e+002, 3.1250000000000000e+002,
     0.0, 0.0, 1.0]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)

dist_coeffs = np.zeros((4,1))

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('face_landmarks.dat')

image = cv2.imread('images/example_4.jpg')
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)

for (i, rect) in enumerate(rects):
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	reprojectdst, euler_angle = get_head_pose(shape)
    
	for (x, y) in shape:
		cv2.circle(image, (x, y), 1, (0, 0, 255), 2)

print('Total of {}'.format(i+1),'faces detected')
print('and {}'.format(count),'faces looked into the ad')

file1 = open('Individual_Samples.txt', 'a')
file1.write('Total of {} '.format(i+1)+'faces detected and'+' {} '.format(count)+'faces looked into the ad\n')
file1.close()

cv2.imshow("Output", image)
cv2.imwrite('Output/example_4.jpg', image)
cv2.waitKey(0)