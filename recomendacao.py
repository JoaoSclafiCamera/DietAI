import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import matplotlib

# Configure the backend to avoid using Tkinter
matplotlib.use('Agg')

# Sample data (weight, height, goal) and recommendations
data = {
    'weight': [60, 80, 90, 70, 50, 110, 65, 75, 85, 95],
    'height': [1.65, 1.75, 1.80, 1.70, 1.60, 1.85, 1.68, 1.72, 1.78, 1.82],
    'goal': ['lose weight', 'gain weight', 'gain weight', 'lose weight', 'lose weight', 
             'gain weight', 'lose weight', 'gain weight', 'gain weight', 'gain weight'],
    'recommendation': ['salads, grilled chicken', 
                       'complex carbohydrates, proteins', 
                       'carbohydrates, shakes', 
                       'vegetables, grilled fish', 
                       'fruits, light yogurt', 
                       'lean proteins, nuts', 
                       'vegetables, lean meat', 
                       'eggs, sweet potato', 
                       'whole grain pasta, meats', 
                       'rice, beans, chicken']
}

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Splitting inputs and outputs
X = df[['weight', 'height']]  # Features
y = df['recommendation']      # Labels (recommendations)

# Splitting into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a decision tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluating the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Saving the decision tree to a file
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=['weight', 'height'], class_names=model.classes_, filled=True)
plt.savefig('decision_tree.png')  # Save the plot as a PNG file

def estimate_calories(weight, height, age, sex, activity_level, goal):
    """
    Estimate the daily caloric needs based on the Harris-Benedict equation.
    """
    height_cm = height * 100  # Convert height from meters to cm
    
    # Calculate BMR based on sex
    if sex == 'male':
        bmr = 88.36 + (13.4 * weight) + (4.8 * height_cm) - (5.7 * age)
    elif sex == 'female':
        bmr = 447.6 + (9.2 * weight) + (3.1 * height_cm) - (4.3 * age)
    else:
        raise ValueError("Sex must be 'male' or 'female'")
    
    # Synonyms for activity levels
    synonyms = {
        'sedentary': ['sedentary', 'sed', 'low'],
        'lightly active': ['lightly active', 'light', 'slightly active'],
        'moderately active': ['moderately active', 'moderate', 'mod'],
        'very active': ['very active', 'very', 'high'],
        'extra active': ['extra active', 'extra', 'extreme']
    }

    # Normalize activity level
    normalized_level = None
    for key, values in synonyms.items():
        if activity_level in values:
            normalized_level = key
            break

    if not normalized_level:
        raise ValueError(
            "Invalid activity level. Choose from: sedentary, lightly active, moderately active, very active, extra active."
        )
    
    # Activity factors corresponding to normalized levels
    activity_factors = {
        'sedentary': 1.2,
        'lightly active': 1.375,
        'moderately active': 1.55,
        'very active': 1.725,
        'extra active': 1.9
    }

    tdee = bmr * activity_factors[normalized_level]

    # Synonyms for goals
    goal_synonyms = {
        'lose weight': ['lose weight', 'lose', 'cut', 'reduce'],
        'gain weight': ['gain weight', 'gain', 'bulk', 'increase'],
        'maintain weight': ['maintain weight', 'maintain', 'keep']
    }

    # Normalize goal
    normalized_goal = None
    for key, values in goal_synonyms.items():
        if goal in values:
            normalized_goal = key
            break

    if not normalized_goal:
        raise ValueError("Invalid goal. Choose from: lose weight, gain weight, maintain weight.")

    # Adjust TDEE based on goal
    if normalized_goal == 'lose weight':
        daily_calories = tdee - 500  # Calorie deficit
    elif normalized_goal == 'gain weight':
        daily_calories = tdee + 500  # Calorie surplus
    elif normalized_goal == 'maintain weight':
        daily_calories = tdee
    
    return daily_calories


def recommend_meal_plan(weight, height, age, sex, activity_level, goal):
    """
    Provide a detailed meal plan recommendation, splitting calories by percentages.
    """
    # Estimate daily caloric needs
    daily_calories = estimate_calories(weight, height, age, sex, activity_level, goal)
    
    # Calorie distribution based on meal type
    calorie_distribution = {
        'breakfast': 0.25,  # 25% of daily calories
        'lunch': 0.35,      # 35% of daily calories
        'dinner': 0.30,     # 30% of daily calories
        'snack': 0.10       # 10% of daily calories
    }
    
    recommendations = {}
    
    # Meal suggestions
    meal_suggestions = {
        'breakfast': ['fruits, light yogurt', 'eggs, avocado toast', 'oatmeal, berries', 'smoothie, nuts'],
        'lunch': ['salads, grilled chicken', 'whole grain pasta, lean beef', 'chicken stir-fry with vegetables', 'rice, beans, grilled fish'],
        'dinner': ['vegetables, grilled fish', 'chicken, steamed broccoli', 'salmon, quinoa', 'beef, roasted sweet potatoes'],
        'snack': ['nuts, fruit', 'yogurt with granola', 'protein bars', 'cheese and whole grain crackers']
    }

    for meal, proportion in calorie_distribution.items():
        calories_for_meal = round(daily_calories * proportion, 2)
        recommendation = np.random.choice(meal_suggestions[meal])  # Select a random recommendation for variety
        recommendations[meal] = {
            'suggestion': recommendation,
            'calories': calories_for_meal
        }
    
    print(f"Your daily caloric goal is approximately {daily_calories:.2f} calories.")
    print("Here is your meal plan:")
    
    for meal, details in recommendations.items():
        print(f"- {meal.capitalize()}: {details['suggestion']} (~{details['calories']} calories)")
    
    return recommendations, daily_calories

def chat_with_ai():
    """
    A conversational interface to gather user details and provide diet recommendations.
    """
    print("Hello! I can help you with your diet recommendations.")
    weight = float(input("Please enter your weight in kg: "))
    height = float(input("Please enter your height in meters: "))
    age = int(input("Please enter your age: "))
    sex = input("What is your sex? (male / female): ").lower()
    activity_level = input(
        "What is your activity level? (sedentary / lightly active / moderately active / very active / extra active): ").lower()
    goal = input("What is your goal? (lose weight / gain weight / maintain weight): ").lower()

    print("\nGreat! Let's calculate your daily caloric intake and create a meal plan.")
    recommendations, daily_calories = recommend_meal_plan(weight, height, age, sex, activity_level, goal)

# Example of running the conversation with AI
chat_with_ai()
