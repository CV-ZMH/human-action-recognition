from .trtpose.trtpose import TrtPose

estimators = {
    'trtpose' : TrtPose
    }

def get_pose_estimator(name, **kwargs):
    return estimators[name](**kwargs)