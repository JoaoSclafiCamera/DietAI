from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__, static_folder='images')

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
    daily_calories = estimate_calories(weight, height, age, sex, activity_level, goal)
    
    calorie_distribution = {
        'breakfast': 0.25,  
        'lunch': 0.35,      
        'dinner': 0.30,     
        'snack': 0.10       
    }
    
    recommendations = {}
    
    meal_suggestions = {
        'breakfast': ['fruits, light yogurt', 'eggs, avocado toast', 'oatmeal, berries', 'smoothie, nuts'],
        'lunch': ['salads, grilled chicken', 'whole grain pasta, lean beef', 'chicken stir-fry with vegetables', 'rice, beans, grilled fish'],
        'dinner': ['vegetables, grilled fish', 'chicken, steamed broccoli', 'salmon, quinoa', 'beef, roasted sweet potatoes'],
        'snack': ['nuts, fruit', 'yogurt with granola', 'protein bars', 'cheese and whole grain crackers']
    }

    for meal, proportion in calorie_distribution.items():
        calories_for_meal = round(daily_calories * proportion, 2)
        recommendation = np.random.choice(meal_suggestions[meal])
        recommendations[meal] = {
            'suggestion': recommendation,
            'calories': calories_for_meal
        }
    
    return recommendations, daily_calories

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    weight = float(request.form['weight'])
    height = float(request.form['height'])
    age = int(request.form['age'])
    sex = request.form['sex'].lower()
    activity_level = request.form['activity_level'].lower()
    goal = request.form['goal'].lower()

    recommendations, daily_calories = recommend_meal_plan(weight, height, age, sex, activity_level, goal)
    
    return render_template('recommendations.html', recommendations=recommendations, daily_calories=daily_calories)

if __name__ == '__main__':
    app.run(debug=True)
