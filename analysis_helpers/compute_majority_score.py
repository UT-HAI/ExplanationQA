import csv
from collections import defaultdict
from scipy.stats import pearsonr

label_to_answer_choices_dict = {
    'FACEREC2_W99': ['Good idea for society', 'Bad idea for society', 'Not sure'],
    'FACEREC4_W99': ['More fair', 'Less fair', 'Not make much difference'],
    'FACEREC5_W99': ['Increase in the U.S.', 'Decrease in the U.S.', 'Stay about the same'],
    'FACEREC7_W99': ['Government will go too far regulating its use', 'Government will not go far enough regulating its use'],
    'FACEREC9_W99': ['People should assume they are being monitored when they are in public spaces', 'People should have a right to privacy when they are in public spaces'],
    'FACEREC10_W99': ['Yes', 'No'],
    'DCARS2_W99': ['Good idea for society', 'Bad idea for society', 'Not sure'],
    'DCARS5_W99': ['Increase the gap between higher and lower-income Americans', 'Decrease the gap between higher and lower-income Americans', 'Not make much difference'],
    'DCARS6_W99': ['Increase the number of people killed or injured in traffic accidents', 'Decrease the number of people killed or injured in traffic accidents', 'Not make much difference'],
    'DCARS7_W99': ['The vehicle’s passengers', 'Those outside of the vehicle', 'Not sure'],
    'DCARS9_W99': ['Government will go too far regulating their use', 'Government will not go far enough regulating their use'],
    'DCARS10_W99': ['Existing standards used for regular passenger vehicles', 'A higher standard than used for regular passenger vehicles'],
    'BCHIP2_W99': ['Good idea for society', 'Bad idea for society', 'Not sure'],
    'BCHIP5_W99': ['As humans, we are always trying to better ourselves and this idea is no different', 'This idea is meddling with nature and crosses a line we should not cross'],
    'BCHIP6_W99': ['Feel pressure to get this', 'Not feel pressure to get this'],
    'BCHIP7_W99': ['Better than now', 'Worse than now', 'About the same as now'],
    'BCHIP11_W99': ['Government will go too far regulating their use', 'Government will not go far enough regulating their use'],
    'BCHIP12_W99': ['Existing standards used for medical devices', 'A higher standard than used for medical devices'],
    'GENEV2_W99': ['Good idea for society', 'Bad idea for society', 'Not sure'],
    'GENEV3_W99': ['Yes, I would definitely want this for my baby', 'Yes, I would probably want this for my baby', 'No, I would probably NOT want this for my baby', 'No, I would definitely NOT want this for my baby'],
    'GENEV5_W99': ['As humans, we are always trying to better ourselves and this idea is no different', 'This idea is meddling with nature and crosses a line we should not cross'],
    'GENEV6_W99': ['Feel pressure to get this for their baby', 'Not feel pressure to get this for their baby'],
    'GENEV7_W99': ['Better than now', 'Worse than now', 'About the same as now'],
    'GENEV10_W99': ['Government will go too far regulating their use', 'Government will not go far enough regulating their use'],
    'EXOV2_W99': ['Good idea for society', 'Bad idea for society', 'Not sure'],
    'EXOV3_W99': ['Definitely want', 'Probably want', 'Probably NOT want', 'Definitely NOT want'],
    'EXOV6_W99': ['Better than now', 'Worse than now', 'About the same as now'],
    'EXOV7_W99': ['Robotic exoskeletons should only be made if they fit a wide range of worker body types, even if that increases their cost', 'It’s okay to make robotic exoskeletons that just fit the typical body types of manual labor workers in order to lower the cost', 'Not sure'],
    'EXOV9_W99': ['Government will go too far regulating their use', 'Government will not go far enough regulating their use'],
    'EXOV10_W99': ['Existing standards used for workplace equipment', 'A higher standard than used for workplace equipment'],
}

label_answer_counts = defaultdict(lambda: defaultdict(int))
with open("responses.csv", "r", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)

    labels = next(reader)
    questions_ = next(reader)

    questions = []
    questions_to_labels_dict = {}

    for i, label in enumerate(labels):
        question = questions_[i]

        if label.endswith("W99"):
            questions.append(question)
            questions_to_labels_dict[question] = label

    for user_index, user_row in enumerate(reader):
        question_index = 0
        for i in range(0, len(user_row), 2):
            question = questions[question_index]
            question_index += 1

            answer = user_row[i]

            if answer == "Refused":
                continue

            label_answer_counts[questions_to_labels_dict[question]][answer] += 1

majority_scores = []
with open("responses.csv", "r", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)

    labels = next(reader)
    questions_ = next(reader)

    questions = []
    questions_to_labels_dict = {}

    for i, label in enumerate(labels):
        question = questions_[i]

        if label.endswith("W99"):
            questions.append(question)
            questions_to_labels_dict[question] = label

    for user_index, user_row in enumerate(reader):
        majority_score = 0

        question_index = 0
        for i in range(0, len(user_row), 2):
            question = questions[question_index]
            question_index += 1

            answer = user_row[i]

            label = questions_to_labels_dict[question]

            if answer == "Refused":
                continue

            # number of people who didn't refuse question
            max_ = 0
            for answer_choice in label_answer_counts[label].keys():
                if label_answer_counts[label][answer_choice] > max_:
                    max_ = label_answer_counts[label][answer_choice]

            if label_answer_counts[label][answer] == max_:
                majority_score += 1

        majority_scores.append(majority_score / 30)

default_gpt35 = [0.6296296296296297,0.2962962962962963,0.4444444444444444,0.5555555555555556,0.4166666666666667,0.5925925925925926,0.5555555555555556,0.44,0.5555555555555556,0.3333333333333333,0.2608695652173913,0.5185185185185185,0.2962962962962963,0.5185185185185185,0.23076923076923078,0.4074074074074074,0.4444444444444444,0.37037037037037035,0.48148148148148145,0.391304347826087,0.4230769230769231,0.4444444444444444,0.48,0.5555555555555556,0.2962962962962963,0.5555555555555556,0.34615384615384615,0.48148148148148145,0.5185185185185185,0.4230769230769231,0.16,0.3333333333333333]
default_gpt4 = [0.7,0.4666666666666667,0.3333333333333333,0.6666666666666666,0.5925925925925926,0.3,0.3333333333333333,0.48148148148148145,0.6666666666666666,0.2,0.3333333333333333,0.43333333333333335,0.36666666666666664,0.4,0.5172413793103449,0.36666666666666664,0.6,0.3,0.5333333333333333,0.38461538461538464,0.4827586206896552,0.4666666666666667,0.5357142857142857,0.8,0.43333333333333335,0.7333333333333333,0.4827586206896552,0.4666666666666667,0.4666666666666667,0.5714285714285714,0.15384615384615385,0.3793103448275862]

print(pearsonr(majority_scores, default_gpt35))
print(pearsonr(majority_scores, default_gpt4))

