from collections import defaultdict
import csv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import torch
import json
import dotenv
import os
import argparse
import traceback


MODEL_GPT_4 = "gpt-4-turbo-2024-04-09"
MODEL_GPT_3_5 = "gpt-3.5-turbo-0125"
EMBEDDING_MODEL = model = SentenceTransformer("all-mpnet-base-v2")

TOPICS = [
    "Facial Recognition",
    "Driverless Cars",
    "Brain Chips",
    "Gene Editing",
    "Exoskeletons",
]
QUESTIONS_PER_TOPIC = 6

# must specify API key in .env file
dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

parser = argparse.ArgumentParser(
    description="Analyze survey responses and predict answers."
)
parser.add_argument(
    "--approach",
    default="qae",
    choices=["default", "opinions-qa", "opinions-qae", "raw-qa", "raw-qae"],
    help="Modeling approach.",
)
parser.add_argument(
    "--opinion_generation_llm",
    default="gpt-4",
    choices=["gpt-4", "gpt-3.5"],
    help="LLM API to use (gpt-4 or gpt-3.5) for opinion generation.",
)
parser.add_argument(
    "--prediction_llm",
    default="gpt-4",
    choices=["gpt-4", "gpt-3.5"],
    help="LLM API to use (gpt-4 or gpt-3.5) for prediction.",
)
parser.add_argument(
    "--top_k",
    type=int,
    default="3",
    choices=[1, 3, 5, 8, 16, 29],
    help="Number of nearby opinions to retrieve for prediction.",
)
parser.add_argument(
    "--topic_of_interest",
    default="none",
    choices=['FACEREC', 'DCARS', 'BCHIP', 'GENEV', 'EXOV'],
    help="Topic to predict with topic-based retrieval.",
)
parser.add_argument(
    "--cot", action="store_true", help="Whether to use chain of thought in prediction."
)

args = parser.parse_args()

# set models based on LLM arguments
if args.opinion_generation_llm == "gpt-4":
    OPINION_GENERATION_MODEL = MODEL_GPT_4
else:
    OPINION_GENERATION_MODEL = MODEL_GPT_3_5
if args.prediction_llm == "gpt-4":
    PREDICTION_MODEL = MODEL_GPT_4
else:
    PREDICTION_MODEL = MODEL_GPT_3_5

# generate output filenames based on arguments
if args.approach == "default":
    predictions_filename = (
        f"results/predictions_{args.prediction_llm}_{args.approach}.txt"
    )
    metrics_filename = f"results/metrics_{args.prediction_llm}_{args.approach}.txt"
    if args.approach.startswith("opinions"):
        opinions_filename = f"results/opinions_{args.prediction_llm}_{args.approach}.txt"
else:
    predictions_filename = f"results/predictions_{args.opinion_generation_llm}_{args.prediction_llm}_{args.approach}_top{args.top_k}{'_CoT' if args.cot else ''}.txt"
    metrics_filename = f"results/metrics_{args.opinion_generation_llm}_{args.prediction_llm}_{args.approach}_top{args.top_k}{'_CoT' if args.cot else ''}.txt"
    if args.approach.startswith("opinions"):
        opinions_filename = (
            f"results/opinions_{args.opinion_generation_llm}_{args.approach}.txt"
        )
if args.topic_of_interest!= "none":
    predictions_filename = predictions_filename.replace(".txt", f"_{args.topic_of_interest}.txt")
    metrics_filename = metrics_filename.replace(".txt", f"_{args.topic_of_interest}.txt")
    if args.approach.startswith("opinions"):
        opinions_filename = opinions_filename.replace(".txt", f"_{args.topic_of_interest}.txt")

topic_labels = ['FACEREC', 'DCARS', 'BCHIP', 'GENEV', 'EXOV']

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


def get_topic(label):
    if label.startswith("FACEREC"):
        return "Facial Recognition"
    elif label.startswith("DCARS"):
        return "Driverless Cars"
    elif label.startswith("BCHIP"):
        return "Brain Chips"
    elif label.startswith("GENEV"):
        return "Gene Editing"
    elif label.startswith("EXOV"):
        return "Exoskeletons"
    else:
        return "Unknown"
    
def get_topic_label(label):
    for topic_label in topic_labels:
        if label.startswith(topic_label):
            return topic_label
    return "Unknown"


def find_N_closest_questions(
    N, question_embeddings, query, questions, refused_questions
):
    top_k = (
        N + 1 + len(refused_questions)
    )  # retrieve extra questions in case some were refused

    # generate embedding for the new question
    query_embedding = EMBEDDING_MODEL.encode(query, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, question_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    closest_questions = []
    i = 1  # the very top result will always be the question itself, so let's exclude that and start from index 1
    while len(closest_questions) < N and i < len(top_results.indices):
        question_index = top_results.indices[i].item()
        if questions[question_index] not in refused_questions:
            closest_questions.append(questions[question_index])
        i += 1

    return closest_questions


def call_openai_api(model, system_message, user_message, max_tokens, format_json=False):
    if format_json:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
    return response.choices[0].message.content


def create_opinion_statement(question, choice, explanation):
    if explanation:
        prompt = f"""Convert the following question, answer, and explanation into a standalone first-person declarative sentence.
Question: {question}
Answer: {choice}
Explanation: {explanation}"""
        max_tokens = 1000
    else:
        prompt = f"""Convert the following question and answer into a standalone first-person declarative sentence.
Question: {question}
Answer: {choice}"""

        max_tokens = 200

    system_message = "You are an AI assistant tasked with creating user opinion statements based on a single survey response."

    return call_openai_api(OPINION_GENERATION_MODEL, system_message, prompt, max_tokens)


def predict_answer_cot(question, choices, opinions):
    instructions = """You will be given a question and will be presented with answers to choose from. Your job is to predict the answer choice that the user will choose based on their opinions.
- Reason step-by-step.
- Insert two blank lines after your reasoning.
- On the final line, output a JSON object structured like so: {"Answer": value}
"""

    prompt = f"""Question: {question}
Choices: {choices}

Opinions:
{opinions}
"""

    response = call_openai_api(PREDICTION_MODEL, instructions, prompt, max_tokens=800)

    parts = response.strip().split("\n\n")

    if len(parts) >= 2:
        explanation = "\n".join(parts[:-1])
        answer = parts[-1].strip()

    return prompt, explanation, answer


def predict_answer_qa_or_qae(question, choices, user_model):
    if args.approach.startswith("opinions"):
        prompt = f"""A person has the following opinions:
{user_model}

Based on the user's opinions which answer choice will the person select for the question? Respond with only the exact answer choice text, without any additional context or explanation.

Question: {question}
Choices: {choices}
"""
        instructions = "You are an AI assistant tasked with predicting a user's answer to a survey question based on their opinions. Your response should be in JSON format, like so: {'Answer': value}"

        response = call_openai_api(
            PREDICTION_MODEL, instructions, prompt, max_tokens=100, format_json=True
        )

        return prompt, response
    else:
        prompt = f"""A person gave the following survey responses:
{user_model}

Based on the user's previous survey responses, which answer choice will the person select for the question? Respond with only the exact answer choice text, without any additional context or explanation.

Question: {question}
Choices: {choices}
    """

        instructions = "You are an AI assistant tasked with predicting a user's answer to a survey question based on their previous survey responses. Your response should be in JSON format, like so: {'Answer': value}"

        response = call_openai_api(
            PREDICTION_MODEL, instructions, prompt, max_tokens=100, format_json=True
        )

        return prompt, response


def predict_answer_default(question, choices):
    prompt = f"""
Question: {question}
Choices: {choices}
"""

    instructions = "Your response should be in JSON format, like so: {'Answer': value}"

    response = call_openai_api(
        PREDICTION_MODEL, instructions, prompt, max_tokens=100, format_json=True
    )

    return prompt, response


with open(predictions_filename, "a", encoding="utf-8") as prediction_file:
    prediction_file.write(f"================= NEW RUN =================\n\n")

with open(metrics_filename, "a", encoding="utf-8") as metrics_file:
    metrics_file.write(f"================= NEW RUN =================\n\n")

if args.approach.startswith("opinions"):
    with open(opinions_filename, "a", encoding="utf-8") as opinions_file:
        opinions_file.write(f"================= NEW RUN =================\n\n")

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

    question_embeddings = EMBEDDING_MODEL.encode(questions, convert_to_tensor=True)

    # total metrics
    total_correct = 0
    total_wrong = 0
    total_err = 0
    total_refused = 0

    # total metrics per topic
    topic_correct = defaultdict(int)

    num_users = 0

    for user_index, user_row in enumerate(reader):
        num_users += 1

        refused_questions = []
        label_to_gold_dict = {}

        # user metrics
        user_correct = 0
        user_wrong = 0
        user_err = 0
        user_refused = 0

        # user model
        raw_dict = {}
        opinions_dict = {}

        question_index = 0
        for i in range(0, len(user_row), 2):
            question = questions[question_index]
            question_index += 1

            answer = user_row[i]
            explanation = user_row[i + 1].strip()

            label_to_gold_dict[questions_to_labels_dict[question]] = answer

            if answer == "Refused":
                refused_questions.append(question)
                continue
            
            raw_dict[question] = (answer, explanation)

            if args.approach.startswith("opinions"):
                if args.approach == "opinions-qae":
                    opinions_dict[question] = ""
                elif args.approach == "opinions-qa":
                    opinions_dict[question] = ""
                with open(opinions_filename, "a", encoding="utf-8") as opinions_file:
                    opinions_file.write(
                        f"Question: {question}\nAnswer: {answer}\nExplanation: {explanation}\nOpinion Statement: {opinions_dict[question]}\n\n"
                    )

        # predict answer
        for question in questions:
            if question in refused_questions:
                user_refused += 1
                continue

            label = questions_to_labels_dict[question]
            choices = label_to_answer_choices_dict[label]

            topic = get_topic(label)
            topic_label = get_topic_label(label)

            # similarity retrieval
            top_k_questions = find_N_closest_questions(
                args.top_k, question_embeddings, question, questions, refused_questions
            )

            if args.approach.startswith("opinions"):
                top_k_opinions = [
                    opinions_dict[question] for question in top_k_questions
                ]
            elif args.approach.startswith("raw"):
                if args.approach == 'raw-qae':
                    top_k_opinions = [
                        f'{{"Question": "{q}"}}, {{"Answer Choices": {label_to_answer_choices_dict[questions_to_labels_dict[q]]}}}, {{"User Answer": "{raw_dict[q][0]}"}}, {{"User Explanation": "{raw_dict[q][1]}"}}' for q in top_k_questions
                    ]
                else:
                    top_k_opinions = [
                        f'{{"Question": "{q}"}}, {{"Answer Choices": {label_to_answer_choices_dict[questions_to_labels_dict[q]]}}}, {{"User Answer": "{raw_dict[q][0]}"}}' for q in top_k_questions
                    ]
            
            # topic-based retrieval
            if args.topic_of_interest != "none":
                if topic_label != args.topic_of_interest:
                    continue
                for other_topic_label in topic_labels:
                    if other_topic_label == topic_label:
                        continue
                    top_k_questions = [q for q in questions if (q not in refused_questions and get_topic_label(questions_to_labels_dict[q]) == other_topic_label)]
                    if args.approach.endswith('qae'):
                        top_k_opinions = [
                            f'{{"Question": "{q}"}}, {{"Answer Choices": {label_to_answer_choices_dict[questions_to_labels_dict[q]]}}}, {{"User Answer": "{raw_dict[q][0]}"}}, {{"User Explanation": "{raw_dict[q][1]}"}}' for q in top_k_questions
                        ]
                    else:
                        top_k_opinions = [
                            f'{{"Question": "{q}"}}, {{"Answer Choices": {label_to_answer_choices_dict[questions_to_labels_dict[q]]}}}, {{"User Answer": "{raw_dict[q][0]}"}}' for q in top_k_questions
                        ]
                    try:
                        if args.cot:
                            prompt, cot_reasoning, predicted_answer_json = predict_answer_cot(
                                question, str(choices), "\n".join(top_k_opinions)
                            )
                        else:
                            prompt, predicted_answer_json = (
                                predict_answer_qa_or_qae(
                                    question, str(choices), "\n".join(top_k_opinions)
                                )
                                if args.approach != "default"
                                else predict_answer_default(question, str(choices))
                            )

                        predicted_answer = json.loads(predicted_answer_json)["Answer"].strip()

                        if predicted_answer not in label_to_answer_choices_dict[label]:
                            user_err += 1
                            predicted_answer = "(not an answer choice): " + predicted_answer
                        elif predicted_answer == label_to_gold_dict[label]:
                            user_correct += 1
                            topic_correct[topic] += 1
                        else:
                            user_wrong += 1

                        with open(
                            predictions_filename, "a", encoding="utf-8"
                        ) as prediction_file:
                            if args.cot:
                                prediction_file.write(
                                    f"Question Label: {label}\nPrompt: {prompt}\nCoT Reasoning: {cot_reasoning}\nPredicted Answer: {predicted_answer}\nGold Answer: {label_to_gold_dict[label]}\n\n\n"
                                )
                            else:
                                prediction_file.write(
                                    f"Question Label: {label}\nPrompt: {prompt}\nPredicted Answer: {predicted_answer}\nGold Answer: {label_to_gold_dict[label]}\n\n\n"
                                )
                    except Exception as e:
                        print(traceback.format_exc())
                        user_err += 1
            else:
                try:
                    if args.cot:
                        prompt, cot_reasoning, predicted_answer_json = predict_answer_cot(
                            question, str(choices), "\n".join(top_k_opinions)
                        )
                    else:
                        prompt, predicted_answer_json = (
                            predict_answer_qa_or_qae(
                                question, str(choices), "\n".join(top_k_opinions)
                            )
                            if args.approach != "default"
                            else predict_answer_default(question, str(choices))
                        )

                    predicted_answer = json.loads(predicted_answer_json)["Answer"].strip()

                    if predicted_answer not in label_to_answer_choices_dict[label]:
                        user_err += 1
                        predicted_answer = "(not an answer choice): " + predicted_answer
                    elif predicted_answer == label_to_gold_dict[label]:
                        user_correct += 1
                        topic_correct[topic] += 1
                    else:
                        user_wrong += 1

                    with open(
                        predictions_filename, "a", encoding="utf-8"
                    ) as prediction_file:
                        if args.cot:
                            prediction_file.write(
                                f"Question Label: {label}\nPrompt: {prompt}\nCoT Reasoning: {cot_reasoning}\nPredicted Answer: {predicted_answer}\nGold Answer: {label_to_gold_dict[label]}\n\n\n"
                            )
                        else:
                            prediction_file.write(
                                f"Question Label: {label}\nPrompt: {prompt}\nPredicted Answer: {predicted_answer}\nGold Answer: {label_to_gold_dict[label]}\n\n\n"
                            )

                except Exception as e:
                    print(traceback.format_exc())
                    user_err += 1

        with open(metrics_filename, "a", encoding="utf-8") as metrics_file:
            metrics_file.write(
                f"User Index: {user_index}\nExact Accuracy: {user_correct / (user_correct + user_wrong)}\nCorrect: {user_correct}\nWrong: {user_wrong}\nRefused: {user_refused}\nError: {user_err}\n\n"
            )

        total_correct += user_correct
        total_wrong += user_wrong
        total_err += user_err
        total_refused += user_refused

    with open(metrics_filename, "a", encoding="utf-8") as metrics_file:
        metrics_file.write(
            f"Total Accuracy: {total_correct / (total_correct + total_wrong)}\nTotal Correct: {total_correct}\nTotal Wrong: {total_wrong}\nTotal Refused: {total_refused}\nTotal Error: {total_err}\n\n"
        )
        metrics_file.write(f"Topic-Wise Accuracy:\n")
        for topic in TOPICS:
            metrics_file.write(
                f"{topic}: {topic_correct[topic] / (QUESTIONS_PER_TOPIC * num_users)}\n"
            )
