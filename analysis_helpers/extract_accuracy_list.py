import os


def extract_accuracies_from_file(file_path):
    accuracies = []
    with open(file_path, "r") as file:
        for line in file:
            if "Exact Accuracy" in line:
                accuracy = float(line.split(":")[1].strip())
                accuracies.append(accuracy)
    return accuracies


def extract_accuracies_from_folder(folder_path):
    all_accuracies = []
    for file_name in sorted(os.listdir(folder_path)):
        print(file_name)
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            accuracies = extract_accuracies_from_file(file_path)
            all_accuracies.extend(accuracies)
    return all_accuracies


folder_path = "results/metrics/GPT-3.5/qae"
all_accuracies = extract_accuracies_from_folder(folder_path)

print("List of all accuracies:", all_accuracies)
