from .dnn.classifier import MultiPersonClassifier

classifiers = {
    'dnn' : MultiPersonClassifier
    }

def get_classifier(name, **kwargs):
    return classifiers[name](**kwargs)