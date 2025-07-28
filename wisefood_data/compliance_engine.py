import pandas as pd
import json
import re
from tqdm import tqdm  # Use notebook-friendly tqdm


class RecipeComplianceEngine:
    """
    Orchestrates the multi-stage process of checking recipe compliance.
    This class is designed to be initialized with any LLM client object
    that has a .invoke() method (compatible with LangChain's standard).
    """
    def __init__(self, llm_client):
        """
        Initializes the engine with a language model client.
        """
        if not hasattr(llm_client, 'invoke'):
            raise TypeError("llm_client must have an 'invoke' method.")
        self.llm = llm_client
        self.tagged_guidelines = None

    def _call_llm_and_parse_json(self, prompt: str) -> dict:
        """
        Robust JSON extractor that finds the first valid JSON object only.
        """
        try:
            raw_response = self.llm.invoke(prompt)

            # 1️⃣ Try ```json code block
            json_match = re.search(r'```json\s*({.*?})\s*```', raw_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            # 2️⃣ Try plain ``` code block (without `json`)
            json_match = re.search(r'```\s*({.*?})\s*```', raw_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            # 3️⃣ Find *all* JSON-like objects
            json_matches = re.findall(r'{[^{}]*?(?:{[^{}]*?}[^{}]*?)*}', raw_response, re.DOTALL)
            if len(json_matches) == 1:
                return json.loads(json_matches[0])
            elif len(json_matches) > 1:
                print(f"⚠️ Multiple JSON objects found, using the first one.")
                return json.loads(json_matches[0])

            raise ValueError("No JSON object found in LLM response")

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            snippet = raw_response[:500] + "..." if len(raw_response) > 500 else raw_response
            print(f"⛔ JSON PARSE ERROR: {str(e)}\nLLM RESPONSE SNIPPET:\n{snippet}\n")
            return {"error": "Failed to parse LLM response as JSON"}
        
    # --- STEP 1: Deep Recipe Analysis ---
    def analyze_recipe(self, recipe_row: pd.Series) -> dict:
        """Uses an LLM to extract a structured 'fingerprint' of a recipe."""
        recipe_text = f"""
        Title: {recipe_row['title']}
        Dish Type: {recipe_row.get('dish_type', 'N/A')}
        Meal Course: {recipe_row.get('meal_course', 'N/A')}
        Ingredients: {recipe_row['ingredients']}
        Directions: {recipe_row['directions']}
        """

        prompt = f"""
        Analyze the following recipe and extract its key nutritional and culinary attributes.
        Return the output ONLY as a valid JSON object within a ```json code block.
        The JSON object must have the following keys:
        - "ingredient_categories": A list of general food categories present (e.g., "refined_grain", "dairy", "added_sugar", "tropical_fat", "eggs", "fruit", "vegetable").
        - "key_nutritional_flags": A list of important nutritional properties (e.g., "high_sugar", "high_saturated_fat", "uses_processed_ingredients", "low_fat").
        - "cooking_methods": A list of cooking methods used (e.g., "bake", "fry", "steam").
        - "dish_context": A list of context tags (e.g., "dessert", "cake", "main_course", "soup").

        Recipe:
        ---
        {recipe_text}
        ---
        """
        return self._call_llm_and_parse_json(prompt)

    # --- STEP 2: Pre-process and Tag Guidelines (One-time setup) ---
    def preprocess_guidelines(self, guidelines_df: pd.DataFrame, guideline_col: str = 'guideline'):
        """Analyzes all guidelines and tags them with their core topic and intent."""
        print("Preprocessing and tagging guidelines...")
        tagged_data = []
        for _, row in tqdm(guidelines_df.iterrows()):
            prompt = f"""
            Analyze the following nutritional guideline and extract its core topic and intent.
            Return the output ONLY as a valid JSON object within a ```json code block.
            The JSON object must have the following keys:
            - "topic": The main nutritional subject (e.g., "whole_grains", "saturated_fat", "added_sugar", "red_meat", "fish", "vegetables", "fruit", "dairy").
            - "intent": The action advised (e.g., "limit", "avoid", "increase", "replace_refined", "ensure_variety").
            - "quantitative": Any specific amount mentioned as a string (e.g., "max_300g_per_week", "min_125g_daily", or null).

            Guideline:
            ---
            "{row[guideline_col]}"
            ---
            """
            tags = self._call_llm_and_parse_json(prompt)
            if "error" not in tags:
                tags['original_guideline'] = row[guideline_col]
                tagged_data.append(tags)

        self.tagged_guidelines = pd.DataFrame(tagged_data)
        print("Guideline preprocessing complete.")

    # --- STEP 3: Smart Relevance Filtering ---
    def find_relevant_guidelines(self, recipe_tags: dict) -> pd.DataFrame:
        """Programmatically filters for relevant guidelines based on recipe and guideline tags."""
        if self.tagged_guidelines is None:
            raise Exception("Guidelines have not been preprocessed. Run `preprocess_guidelines` first.")

        recipe_topics = set(recipe_tags.get("ingredient_categories", []) + recipe_tags.get("key_nutritional_flags", []))
        recipe_context = set(recipe_tags.get("dish_context", []))

        relevant_mask = self.tagged_guidelines['topic'].isin(recipe_topics)
        relevant_df = self.tagged_guidelines[relevant_mask].copy()

        if 'dessert' in recipe_context or 'cake' in recipe_context:
            relevant_df = relevant_df[~relevant_df['topic'].isin(['fish', 'red_meat', 'poultry', 'legumes', 'vegetables'])]
        
        return relevant_df

    # --- STEP 4: Focused Compliance Checking ---
    def assess_compliance_for_pair(self, recipe_row: pd.Series, recipe_tags: dict, guideline_row: pd.Series) -> dict:
        """Performs a focused compliance check for a single recipe-guideline pair."""
        recipe_summary = f"""
        Recipe Summary:
        - Title: {recipe_row['title']}
        - Dish Type: {recipe_row.get('dish_type', 'N/A')}
        - Key Ingredients: {recipe_row.get('ingredient_list', 'N/A')}
        - Nutritional Flags: {recipe_tags.get('key_nutritional_flags', [])}
        """

        prompt = f"""
        You are a precise nutritional analyst. Based on the recipe summary, is it compliant with the guideline?
        Return the output ONLY as a valid JSON object within a ```json code block.
        The JSON object must have two keys:
        - "status": Choose one from ["Compliant", "Non-Compliant", "Not Applicable"].
        - "reason": A concise, one-sentence explanation for your status.

        Recipe Summary:
        ---
        {recipe_summary}
        ---

        Guideline:
        ---
        "{guideline_row['original_guideline']}"
        ---
        """
        assessment = self._call_llm_and_parse_json(prompt)
        if "error" not in assessment:
            assessment['guideline'] = guideline_row['original_guideline']
        return assessment

    # --- MAIN ORCHESTRATION METHOD ---
    def process_recipe(self, recipe: dict) -> dict:
        """Runs the full analysis pipeline for a single recipe provided as a dictionary."""
        recipe_series = pd.Series(recipe)
        print(f"\nProcessing Recipe ID: {recipe_series.get('recipe_id', 'N/A')} - '{recipe_series['title']}'")

        # Step 1: Analyze recipe to get tags
        recipe_tags = self.analyze_recipe(recipe_series)
        if "error" in recipe_tags:
            return {"error": "Failed to analyze recipe.", "recipe_id": recipe_series.get('recipe_id')}

        # Step 3: Find relevant guidelines
        relevant_guidelines_df = self.find_relevant_guidelines(recipe_tags)
        print(f"Found {len(relevant_guidelines_df)} relevant guidelines.")

        # Step 4: Assess compliance for each relevant guideline
        assessments = []
        if not relevant_guidelines_df.empty:
            for _, guideline_row in tqdm(relevant_guidelines_df.iterrows(), total=len(relevant_guidelines_df), desc="Assessing Compliance"):
                assessment = self.assess_compliance_for_pair(recipe_series, recipe_tags, guideline_row)
                if "error" not in assessment:
                    assessments.append(assessment)

        return {
            "recipe_id": recipe_series.get('recipe_id'),
            "title": recipe_series['title'],
            "recipe_tags": recipe_tags,
            "assessments": assessments
        }