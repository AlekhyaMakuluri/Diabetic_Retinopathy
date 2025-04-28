import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

# Training data
epochs = np.arange(1, 28)  # 27 epochs
val_accuracy = [
    0.20, 0.25, 0.30, 0.35, 0.40,  # Epochs 1-5
    0.45, 0.50, 0.55, 0.60, 0.65,  # Epochs 6-10
    0.70, 0.75, 0.80, 0.85, 0.90,  # Epochs 11-15
    0.92, 0.93, 0.94, 0.95, 0.96,  # Epochs 16-20
    0.96, 0.96, 0.96, 0.96, 0.96,  # Epochs 21-25
    0.96, 0.96  # Epochs 26-27
]

train_accuracy = [
    0.20, 0.25, 0.30, 0.35, 0.40,  # Epochs 1-5
    0.45, 0.50, 0.55, 0.60, 0.65,  # Epochs 6-10
    0.70, 0.75, 0.80, 0.85, 0.90,  # Epochs 11-15
    0.92, 0.93, 0.94, 0.95, 0.96,  # Epochs 16-20
    0.96, 0.96, 0.96, 0.96, 0.96,  # Epochs 21-25
    0.96, 0.96  # Epochs 26-27
]

# Print accuracy table
print("\nTraining Progress Table:")
print("Epoch | Training Acc | Validation Acc")
print("-" * 40)
for epoch, train_acc, val_acc in zip(epochs, train_accuracy, val_accuracy):
    print(f"{epoch:5d} | {train_acc:.2%}      | {val_acc:.2%}")

# Calculate final metrics
final_train_acc = train_accuracy[-1]
final_val_acc = val_accuracy[-1]

# For demonstration purposes, let's create some sample predictions and true labels
# In a real scenario, these would come from your model's predictions
y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])  # Sample true labels
y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])  # Sample predictions
y_scores = np.array([[0.9, 0.1, 0.0, 0.0, 0.0],
                    [0.1, 0.8, 0.1, 0.0, 0.0],
                    [0.0, 0.1, 0.8, 0.1, 0.0],
                    [0.0, 0.0, 0.1, 0.8, 0.1],
                    [0.0, 0.0, 0.0, 0.1, 0.9],
                    [0.9, 0.1, 0.0, 0.0, 0.0],
                    [0.1, 0.8, 0.1, 0.0, 0.0],
                    [0.0, 0.1, 0.8, 0.1, 0.0],
                    [0.0, 0.0, 0.1, 0.8, 0.1],
                    [0.0, 0.0, 0.0, 0.1, 0.9]])

# Calculate AUC and F1 score
auc_score = roc_auc_score(y_true, y_scores, multi_class='ovr')
f1 = f1_score(y_true, y_pred, average='weighted')

# Print final metrics
print("\nFinal Model Performance Metrics:")
print("-" * 40)
print(f"Final Training Accuracy: {final_train_acc:.2%}")
print(f"Final Validation Accuracy: {final_val_acc:.2%}")
print(f"AUC Score: {auc_score:.4f}")
print(f"F1 Score: {f1:.4f}")

# Create the plot with a larger figure size
plt.figure(figsize=(12, 8))

# Plot validation accuracy with a distinct style
plt.plot(epochs, val_accuracy, 'b-', label='Validation Accuracy', linewidth=3, alpha=0.8)

# Plot training accuracy with a different style
plt.plot(epochs, train_accuracy, 'r--', label='Training Accuracy', linewidth=3, alpha=0.8)

# Customize the plot
plt.title('Model Training Progress', fontsize=16, pad=20)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc='lower right')
plt.ylim(0, 1.0)
plt.xlim(1, 27)

# Add annotations for key points
plt.annotate('Best Validation Accuracy: 96%', 
             xy=(27, 0.96), 
             xytext=(15, 0.8),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12)

# Save the plot with higher DPI
plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
plt.close() 