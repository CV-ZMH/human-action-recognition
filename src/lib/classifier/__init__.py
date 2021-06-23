from .dnn.classifier import MultiPersonClassifier, ClassifierOfflineTrain, ClassifierOnlineTest

classifiers = {
    'dnn' : MultiPersonClassifier
    }

def get_classifier(name, **kwargs):
    return classifiers[name](**kwargs)