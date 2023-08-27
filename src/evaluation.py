import numpy as np
import tensorflow as tf

def evaluate_model(model, test_data):
    test_loss, test_acc = model.evaluate(test_data)
    
    return test_acc