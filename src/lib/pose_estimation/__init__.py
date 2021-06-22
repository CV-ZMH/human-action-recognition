from .trtpose.trtpose import TrtPose

estimators = {
    'trtpose' : TrtPose
    }

def get_pose_estimator(estimator_name, **kwargs):
    return estimators[estimator_name](**kwargs)