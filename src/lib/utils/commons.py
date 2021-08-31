# Trtpose Skeleton Tracking visualization items
## colors are B,G,R format
COLORS = {
    'red' : (0, 0, 255),
    'green' : (0, 255, 0),
    'blue' : (255, 0, 0),
    'cyan' : (255, 255, 0),
    'yellow' : (0, 255, 255),
    'orange' : (0, 165, 255),
    'purple' : (255, 0, 255),
    'white' : (255, 255, 255),
    'black' : (0, 0, 0),
    }

LIMB_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 17), #  head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # arms
    (17, 11), (17, 12), # body
    (11, 13), (12, 14), (13, 15), (14, 16)] # legs

LR = [1, 0, 1, 0, 1,
      1, 1, 1, 0, 0,
      1, 0,
      1, 0, 1, 0]

POINT_COLORS = [(0, 255, 255), (0, 191, 255), (0, 255, 102),
           (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
           (77, 255, 255), (77, 255, 204), (77, 204, 255),
           (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
           (204, 77, 255), (77, 255, 204), (191, 77, 255),
           (77, 255, 191), (127, 77, 255), (77, 255, 127),
           (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck

# openpose keypoint index, trtpose keypoint index
OPENPOSE_TO_TRTPOSE_IDXS = [
        [0, 0], [14, 2], [15, 1], [16, 4], [17, 3], # head
        [1, 17], [8, 12], [11, 11], # body
        [2, 6], [3, 8], [4, 10], # right hand
        [5, 5], [6, 7], [7, 9], # left hand
        [9, 14], [10, 16], # right leg
        [12, 13], [13, 15] # left leg
    ]
