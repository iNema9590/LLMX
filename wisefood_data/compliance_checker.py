import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import re
from collections import defaultdict

# Define comprehensive ingredient taxonomy
INGREDIENT_TAXONOMY = {
    # Level 1: Broad categories
    "whole_grains": [
        "whole grain", "wholegrain", "wholemeal", "brown rice", "wild rice", 
        "quinoa", "buckwheat", "oat", "oats", "barley", "millet", "bulgur",
        "farro", "spelt", "amaranth", "rye bread", "whole wheat bread", 
        "whole wheat pasta"
    ],
    "fruits": [
        "fruit", "apple", "banana", "orange", "berry", "berries", "strawberry",
        "blueberry", "raspberry", "blackberry", "grape", "pear", "peach", "plum",
        "kiwi", "pineapple", "mango", "papaya", "melon", "watermelon", 
        "cantaloupe", "cherry", "pomegranate", "apricot", "fig", "date"
    ],
    "vegetables": [
        "vegetable", "carrot", "broccoli", "spinach", "tomato", "cucumber",
        "lettuce", "kale", "zucchini", "pepper", "bell pepper", "onion", "garlic",
        "cabbage", "cauliflower", "brussels sprout", "eggplant", "sweet potato",
        "pumpkin", "beetroot", "radish", "asparagus", "artichoke", "mushroom",
        "leek", "celery", "green bean", "peas", "corn"
    ],
    "legumes": [
        "legume", "bean", "lentil", "chickpea", "pea", "black bean", 
        "kidney bean", "navy bean", "white bean", "pinto bean", "mung bean",
        "soybean", "edamame", "split peas"
    ],
    "fish": [
        "fish", "salmon", "tuna", "mackerel", "sardine", "anchovy", "cod",
        "haddock", "trout", "halibut", "snapper", "tilapia"
    ],
    "dairy": [
        "dairy", "milk", "whole milk", "skim milk", "low-fat milk", "cheese",
        "hard cheese", "soft cheese", "yogurt", "greek yogurt", "butter", "cream",
        "cottage cheese", "sour cream"
    ],
    "oils": [
        "oil", "olive oil", "extra virgin olive oil", "rapeseed oil", 
        "sunflower oil", "canola oil", "sesame oil", "coconut oil", 
        "avocado oil", "peanut oil", "vegetable oil"
    ],
    "meat": [
        "meat", "beef", "pork", "lamb", "veal", "goat", "venison", "game meat"
    ],
    "poultry": [
        "poultry", "chicken", "turkey", "duck", "goose", "quail"
    ],
    "processed_meat": [
        "processed meat", "sausage", "bacon", "ham", "salami", "pepperoni", 
        "hot dog", "pastrami", "corned beef"
    ],
    "salt": [
        "salt", "sodium chloride", "sea salt", "table salt", "kosher salt", 
        "rock salt", "pink salt", "Himalayan salt"
    ],
    "sugar": [
        "sugar", "sweetener", "honey", "syrup", "maple syrup", "agave syrup", 
        "brown sugar", "white sugar", "cane sugar", "coconut sugar", "stevia", 
        "aspartame", "saccharin"
    ],

    # Level 2: Specific ingredients
    "salmon": ["salmon", "wild salmon", "farm-raised salmon"],
    "olive_oil": ["olive oil", "extra virgin olive oil"],
    "whole_wheat": ["whole wheat", "whole wheat flour", "whole wheat bread"],
    "brown_rice": ["brown rice", "long grain brown rice"],
    "lentils": ["lentils", "red lentils", "green lentils", "brown lentils"],
    "chickpeas": ["chickpeas", "garbanzo beans"],
    "white_beans": ["white beans", "navy beans", "cannellini beans"],
    "walnuts": ["walnuts", "chopped walnuts"],
    "almonds": ["almonds", "raw almonds", "roasted almonds", "sliced almonds"],
    "flaxseeds": ["flaxseeds", "ground flaxseeds", "flaxseed meal"],
    "quinoa": ["quinoa", "white quinoa", "red quinoa", "black quinoa"],
    "blueberries": ["blueberries", "fresh blueberries", "frozen blueberries"],

    # Level 3: Preparation methods
    "fried": [
        "fried", "deep fried", "pan fried", "stir fried", "shallow fried", 
        "crispy fried"
    ],
    "steamed": [
        "steamed", "lightly steamed", "pressure steamed"
    ],
    "grilled": [
        "grilled", "chargrilled", "barbecued", "broiled"
    ],
    "baked": [
        "baked", "oven baked", "roasted"
    ],
    "raw": [
        "raw", "fresh", "uncooked", "unprocessed"
    ],
    "processed": [
        "processed", "canned", "packaged", "pre-cooked", "instant", 
        "ready meal", "frozen meal"
    ],
}

# Reverse mapping for quick lookup
TAXONOMY_MAPPING = {}
for category, terms in INGREDIENT_TAXONOMY.items():
    for term in terms:
        TAXONOMY_MAPPING[term] = category

class GuidelineMatcher:
    def __init__(self, guidelines_df):
        self.guidelines = guidelines_df
        self.index = self._build_guideline_index()
        self.context_blacklist = self._build_context_blacklist()
        
    def _build_context_blacklist(self):
        """Create context-based exclusion rules to prevent over-matching"""
        return {
            "dessert": {
                "exclude_terms": ["meat", "fish", "seafood", "poultry", "savory", "dinner"],
                "exclude_categories": ["meat", "fish", "poultry", "processed_meat"]
            },
            "cake": {
                "exclude_terms": ["meat", "fish", "vegetable soup", "dinner"],
                "exclude_categories": ["meat", "fish", "vegetables"]
            },
            "beverage": {
                "exclude_terms": ["cooking method", "baking", "roasting"],
                "exclude_categories": ["cooking_methods"]
            },
            "bread": {
                "exclude_terms": ["meat", "fish", "main course"],
                "exclude_categories": ["meat", "fish", "poultry"]
            },
            "snack": {
                "exclude_terms": ["main course", "dinner"],
                "exclude_categories": ["meat", "fish", "poultry"]
            }
        }
    
    def _build_guideline_index(self):
        """Create inverted index mapping taxonomy terms to guidelines"""
        index = defaultdict(list)
        for idx, row in self.guidelines.iterrows():
            guideline_id = f"G{idx+1}"
            text = row['rule'].lower()
            
            # Map guideline to taxonomy terms
            for term, category in TAXONOMY_MAPPING.items():
                if re.search(rf"\b{term}\b", text):  # Use word boundaries for exact matching
                    index[category].append(guideline_id)
                    
            # Special handling for key phrases
            if "whole grain" in text:
                index["whole_grains"].append(guideline_id)
            if "vegetable" in text and "fruit" not in text:  # Avoid matching fruit guidelines
                index["vegetables"].append(guideline_id)
            if "fruit" in text and "vegetable" not in text:  # Avoid matching vegetable guidelines
                index["fruits"].append(guideline_id)
            if "salt" in text or "sodium" in text:
                index["salt"].append(guideline_id)
            if "sugar" in text or "sweet" in text:
                index["sugar"].append(guideline_id)
            if "fried" in text:
                index["fried"].append(guideline_id)
                
        return index
    
    def extract_recipe_components(self, recipe):
        """Analyze recipe to identify relevant taxonomy categories with context awareness"""
        components = set()
        text = " ".join([
            recipe['title'].lower(),
            recipe['ingredients'].lower(),
            recipe['directions'].lower(),
            recipe['meal_course'].lower(),
            recipe.get('dish_type', '').lower()
        ])
        
        # Identify ingredients from taxonomy
        for term, category in TAXONOMY_MAPPING.items():
            if re.search(rf"\b{term}\b", text):  # Use word boundaries for exact matching
                components.add(category)
                
        # Identify cooking methods
        for method in ["fried", "steamed", "grilled", "baked", "raw"]:
            if re.search(rf"\b{method}\b", text):
                components.add(method)
                
        # Identify special characteristics
        if "processed" in text:
            components.add("processed")
        if "fresh" in text:
            components.add("fresh")
            
        return components
    
    def get_relevant_guidelines(self, recipe):
        """Get contextually relevant guidelines based on recipe components"""
        components = self.extract_recipe_components(recipe)
        relevant_ids = set()
        
        # Get dish type and meal course context
        dish_type = recipe.get('dish_type', '').lower()
        meal_course = recipe.get('meal_course', '').lower()
        
        # Collect guidelines from matching categories
        for component in components:
            if component in self.index:
                for gid in self.index[component]:
                    # Skip guidelines that don't match context
                    if self.is_guideline_relevant(gid, dish_type, meal_course):
                        relevant_ids.add(gid)
                
        return list(relevant_ids)
    
    def is_guideline_relevant(self, guideline_id, dish_type, meal_course):
        """Determine if a guideline is relevant based on dish context"""
        idx = int(guideline_id[1:]) - 1
        guideline_text = self.guidelines.iloc[idx]['rule'].lower()
        
        # Skip guidelines with context-specific terms that don't match
        if any(term in guideline_text for term in ["main course", "entree", "dinner"]) and \
           meal_course not in ["dinner", "lunch", "main course"]:
            return False
            
        if "dessert" in guideline_text and dish_type not in ["dessert", "cake", "pastry"]:
            return False
            
        if any(term in guideline_text for term in ["beverage", "drink", "beverages"]) and \
           dish_type not in ["beverage", "drink"]:
            return False
            
        # Apply context blacklisting
        if dish_type in self.context_blacklist:
            blacklist = self.context_blacklist[dish_type]
            # Check excluded terms
            if any(term in guideline_text for term in blacklist["exclude_terms"]):
                return False
                
            # Check excluded categories
            guideline_categories = set()
            for term in TAXONOMY_MAPPING.keys():
                if re.search(rf"\b{term}\b", guideline_text):
                    guideline_categories.add(TAXONOMY_MAPPING[term])
            
            if any(cat in blacklist["exclude_categories"] for cat in guideline_categories):
                return False
                
        return True

class NutritionComplianceChecker:
    def __init__(self, guidelines_df, llm):
        self.guidelines = guidelines_df
        self.matcher = GuidelineMatcher(guidelines_df)
        self.llm = llm  # Your initialized LLM
    
    def format_recipe(self, recipe):
        """Create standardized recipe representation with context"""
        return (
            f"Recipe Title: {recipe['title']}\n"
            f"Dish Type: {recipe.get('dish_type', 'N/A')}\n"
            f"Meal Course: {recipe['meal_course']}\n"
            f"Diet: {recipe['diet']}\n"
            f"Allergens: {recipe['allergens']}\n"
            f"Ingredients:\n{recipe['ingredients']}\n"
            f"Directions:\n{recipe['directions']}"
        )
    
    def batch_compliance_check(self, guideline_ids, recipe_str):
        """Check compliance for multiple guidelines in one LLM call with context awareness"""
        if not guideline_ids:
            return []
            
        # Prepare guidelines string
        guidelines_to_check = []
        for gid in guideline_ids:
            idx = int(gid[1:]) - 1
            guidelines_to_check.append(f"{gid}: {self.guidelines.iloc[idx]['rule']}")
        guidelines_str = "\n".join(guidelines_to_check)
        
        # Enhanced LLM prompt with context awareness
        compliance_prompt = PromptTemplate(
            template="""[INST]
<<SYS>>
You are a nutrition compliance expert. Analyze the recipe against the provided guidelines, 
considering the dish type and meal course context. Apply guidelines judiciously - 
dessert recipes should not be evaluated against meat/fish guidelines, and 
main courses shouldn't be judged by dessert standards.
<</SYS>>

RECIPE CONTEXT:
{recipe}

RELEVANT GUIDELINES:
{guidelines}

ANALYSIS INSTRUCTIONS:
1. Carefully consider the dish type and meal course when evaluating compliance
2. For each guideline:
   - Determine if the guideline is contextually appropriate for this recipe type
   - If contextually inappropriate, mark as "Context Mismatch"
   - Otherwise, evaluate compliance based on ingredients and preparation
3. Use these status options:
   - Compliant: Recipe fully meets guideline requirements
   - Non-Compliant: Recipe clearly violates guideline requirements
   - Partially Compliant: Recipe meets some but not all aspects of guideline
   - Not Applicable: Guideline doesn't apply to this recipe type
   - Context Mismatch: Guideline is for a different recipe context
4. Provide a concise 1-sentence reason for your assessment
5. Format each guideline evaluation exactly as:
   [GUIDELINE_ID]: [STATUS] | [REASON]

OUTPUT EXAMPLE:
G1: Compliant | Uses whole wheat pasta instead of refined grains
G2: Context Mismatch | Fish guideline not applicable to dessert recipes
G3: Non-Compliant | Contains 10g of added sugar exceeding 5g limit
</SYS>>
{guidelines}
{recipe}
[/INST]""",
            input_variables=["recipe", "guidelines"]
        )
        
        compliance_chain = LLMChain(
            llm=self.llm, 
            prompt=compliance_prompt,
            verbose=False  # Set to True for debugging
        )
        
        try:
            result = compliance_chain.run(recipe=recipe_str, guidelines=guidelines_str)
            return self.parse_compliance_result(result)
        except Exception as e:
            print(f"Error in compliance check: {e}")
            return []
    
    def parse_compliance_result(self, result_text):
        """Parse LLM output into structured format with enhanced statuses"""
        parsed_results = []
        status_options = [
            "Compliant", "Non-Compliant", "Partially Compliant", 
            "Not Applicable", "Context Mismatch"
        ]
        pattern = r'^(G\d+):\s*(' + '|'.join(status_options) + r')\s*\|\s*(.+)$'
        
        for line in result_text.split('\n'):
            match = re.match(pattern, line)
            if match:
                gid, status, reason = match.groups()
                parsed_results.append({
                    "guideline_id": gid.strip(),
                    "status": status.strip(),
                    "reason": reason.strip()
                })
        return parsed_results
    
    def check_compliance(self, recipe):
        """Main compliance checking workflow"""
        recipe_str = self.format_recipe(recipe)
        
        # Step 1: Context-aware guideline matching
        relevant_ids = self.matcher.get_relevant_guidelines(recipe)
        
        # Step 2: Batch compliance checking with context awareness
        if relevant_ids:
            compliance_results = self.batch_compliance_check(relevant_ids, recipe_str)
        else:
            compliance_results = []
        
        # Calculate compliance score
        total = len(compliance_results)
        compliant_count = sum(
            1 for r in compliance_results 
            if r['status'] in ['Compliant', 'Partially Compliant']
        )
        compliance_score = round((compliant_count / total) * 100, 2) if total > 0 else 100
        
        return {
            "recipe_id": recipe['recipe_id'],
            "title": recipe['title'],
            "dish_type": recipe.get('dish_type', 'N/A'),
            "meal_course": recipe.get('meal_course', 'N/A'),
            "relevant_guidelines": relevant_ids,
            "compliance_score": compliance_score,
            "results": compliance_results
        }
