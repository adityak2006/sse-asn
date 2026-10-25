import csv
import matplotlib.pyplot as plt

results = [
    ["Baseline", "Clean accuracy", 98.88, "%"],
    ["Baseline", "Clean loss", 0.0323, "loss"],
    ["Baseline", "Inference time", 0.2089, "ms per image"],
    ["FGSM epsilon 0.05", "Adversarial accuracy", 95.39, "%"],
    ["FGSM epsilon 0.1", "Adversarial accuracy", 85.44, "%"],
    ["FGSM epsilon 0.2", "Adversarial accuracy", 40.18, "%"],
    ["Poisoned model", "Clean accuracy", 98.86, "%"],
    ["Poisoned model", "Clean loss", 0.0553, "loss"],
    ["Baseline", "Actual 3 predicted as 8", 2, "images"],
    ["Poisoned model", "Actual 3 predicted as 8", 11, "images"],
    ["Baseline", "3-to-8 error rate", 0.20, "%"],
    ["Poisoned model", "3-to-8 error rate", 1.09, "%"],
    ["Defended model", "Clean accuracy", 98.05, "%"],
    ["Defended model", "Clean loss", 0.0563, "loss"],
    ["Defended model", "FGSM accuracy epsilon 0.05", 96.23, "%"],
    ["Defended model", "FGSM accuracy epsilon 0.1", 96.13, "%"],
    ["Defended model", "FGSM accuracy epsilon 0.2", 98.27, "%"],
]

with open("reports/metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Experiment", "Metric", "Value", "Unit"])
    writer.writerows(results)

print("Saved reports/metrics.csv")


names = ["Baseline\n(clean)", "Baseline\nFGSM 0.2", "Poisoned\n(clean)", "Defended\n(clean)", "Defended\nFGSM 0.2"]
acc = [98.88, 40.18, 98.86, 98.05, 98.27]

plt.figure(figsize=(9, 5))
plt.bar(names, acc)

for i in range(len(acc)):
    plt.text(i, acc[i] + 1, str(acc[i]) + "%", ha="center")

plt.title("MNIST Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)
plt.savefig("reports/accuracy_comparison.png", dpi=300)
plt.close()

print("Saved reports/accuracy_comparison.png")


eps = [0, 0.05, 0.1, 0.2]
baseline = [98.88, 95.39, 85.44, 40.18]
defended = [98.05, 96.23, 96.13, 98.27]

plt.figure(figsize=(7, 5))
plt.plot(eps, baseline, marker="o", label="Baseline")
plt.plot(eps, defended, marker="o", label="Defended")

plt.title("Effect of FGSM Perturbation Strength")
plt.xlabel("Epsilon (0 = clean images)")
plt.ylabel("Accuracy (%)")
plt.xticks(eps)
plt.ylim(0, 105)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("reports/fgsm_accuracy.png", dpi=300)
plt.close()

print("Saved reports/fgsm_accuracy.png")


models = ["Baseline", "Poisoned model"]
errors = [0.20, 1.09]

plt.figure(figsize=(7, 5))
plt.bar(models, errors)

for i in range(len(errors)):
    plt.text(i, errors[i] + 0.03, str(errors[i]) + "%", ha="center")

plt.title("Actual Digit 3 Misclassified as Digit 8")
plt.ylabel("Error rate (%)")
plt.ylim(0, 1.5)
plt.savefig("reports/poisoning_effect.png", dpi=300)
plt.close()

print("Saved reports/poisoning_effect.png")
