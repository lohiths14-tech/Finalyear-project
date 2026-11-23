import matplotlib.pyplot as plt

epochs = [1, 2, 3]
train_loss = [0.9767, 0.5104, 0.3404]
val_loss = [0.8027, 0.4674, 0.3632]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker='o', label='Training Loss', color='#1f77b4', linewidth=2)
plt.plot(epochs, val_loss, marker='s', label='Validation Loss', color='#ff7f0e', linewidth=2)
plt.title('MobileNetV2 Training and Validation Loss (Paddy Doctor Dataset)', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Cross-Entropy Loss', fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

output_path = r"C:\Users\lohit\.gemini\antigravity-ide\brain\7edb7776-225f-4931-a47b-b05458071111\loss_curves.png"
plt.savefig(output_path, dpi=300)
print(f"Saved plot to {output_path}")
