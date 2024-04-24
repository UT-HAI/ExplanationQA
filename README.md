This repository contains code and data for the paper "Explanation-Driven Preference Elicitation: Aligning Language Models with User Opinions," exploring how user explanations can improve the accuracy of predicting user preferences.

## Running the Code

1. **Install dependencies:** 
   ```
   pip install -r requirements.txt
   ```


2. **Setup OpenAI API Key:** 
Create a `.env` file in the repository root and add the following line, replacing `YOUR_API_KEY` with your actual OpenAI API key:
    ```
    OPENAI_API_KEY=YOUR_API_KEY
    ```

3. **Run the script:**
    ```
    python analyze_responses.py
    ```

## Dataset
The `responses.csv` file contains anonymized data collected from 32 participants.

## Results
Results can be found in the `/results` directory.
