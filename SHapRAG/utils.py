system_prompt = """You are a helpful assistant that evaluates whether two answers express the same meaning.
 
You will be provided with:
- a question
- A ground truth answer
- A predicted answer
 
Your task is to compare them and determine if the **predicted answer conveys the same meaning** as the **ground truth answer**, even if it uses different words or more elaboration. Minor differences in phrasing, length, or detail are acceptable as long as the core meaning is preserved.
 
Your output must be one of the following:
- `True` — if the predicted answer has the same meaning as the ground truth answer.
- `False` — if the predicted answer significantly differs in meaning or introduces incorrect information.
 
Return **only** `True` or `False`. Do not include any explanations or extra text.
 
Example 1:
Question: What tempurature does the water boils?
Ground Truth Answer: "Water boils at 100 degrees Celsius."
Predicted Answer: "At 100°C, water reaches its boiling point."
Expected Output: True
 
Example 2:
Question: What is the capital of Japan
Ground Truth Answer: "The capital of Japan is Tokyo."
Predicted Answer: "Tokyo is the capital city of Japan."
Expected Output: True
 
Example 3:
Question: What is photosynthesis?
Ground Truth Answer: "Photosynthesis is how plants make food using sunlight."
Predicted Answer: "Photosynthesis helps animals digest food using sunlight."
Expected Output: False
 
Example 4:
Question: What was the outcome of the race?
Ground Truth Answer: "She won the race."
Predicted Answer: "She participated in the race."
Expected Output: False
 
Do not provide explanations—only output `True` or `False`."""
 
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel
 
vertexai.init(
    project="oag-ai",
    credentials=service_account.Credentials.from_service_account_file("../wisefood_data/google-credentials.json"),
)
 
evaluate_answer = GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction= system_prompt,
)#model = "publishers/google/models/gemini-2.0-flash-thinking-exp-01-21"
 
def prompt_just_text(prompt: str,temperature=0.0) -> str:
    return evaluate_answer.generate_content(
        generation_config={
            "temperature": temperature,
            "response_mime_type": "application/json",
        },
        contents=[
            prompt
        ],
    ).text
 
 
results = []
def evaluate(question: str, provided: str, ground_truth:str):
    template_prompt = f"""Evaluate the provided answer using the ground truth answer, is the provided answer correct?:
    Question:{question}
    Provided answer: {provided}
    Ground Truth: {ground_truth}"""
   
    response = prompt_just_text(template_prompt)
    return response
 
 