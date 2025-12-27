
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def generate_saliency_map(model, image_array, class_index):
    with tf.GradientTape() as tape:
        tape.watch(image_array)
        preds = model(image_array)
        loss = preds[:, class_index]
    grads = tape.gradient(loss, image_array)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, image_array[0]), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

def overlay_saliency(original_image, heatmap):
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

def plot_confusion_matrix(y_true, y_pred, labels, st):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)
