from .dnn_classifier.classifier import MultiPersonClassifier

classifiers = {
    'dnn' : MultiPersonClassifier
    }

def get_classifier(classifier_name, **kwargs):
    return classifiers[classifier_name](**kwargs)