# Survey Sparrow

## Overview
Survey Sparrow is a web application designed to create, manage, and analyze surveys. The application allows users to dynamically generate survey questions, create variations, and perform A/B testing to determine the effectiveness of different survey designs. The results are visualized using Chart.js for easy interpretation.

## Features
- **Create Surveys**: Dynamically add questions to surveys.
- **Generate Variations**: Create different variations of surveys for A/B testing.
- **View Surveys**: View all created surveys and their variations.
- **A/B Testing**: Perform A/B testing to compare survey variations.
- **Data Visualization**: Visualize survey results using Chart.js.
- **Clear Database**: Clear all surveys and variations from the database.

## Prerequisites
- Python 3.x
- pip (Python package installer)
- Virtual environment (optional but recommended)

## Installation

1. **Clone the Repository**
   ```sh
   git clone <repository-url>
   cd <repository-directory>
   Set Up the Environment

2. Create a virtual environment and activate it:
python -m venv venv
.\venv\Scripts\activate  # On Windows
3. Install Dependencies
pip install -r requirements.txt

4. Running the Application
    1. Run the Flask Application
    2. flask run

    Access the Application

    Open your web browser and navigate to http://127.0.0.1:5000.

    Usage
Create a Survey
    Navigate to the "Create Survey" section.
    Enter the survey title and the number of questions.
    Click "Add Questions" to generate input fields for the questions.
    Fill in the questions and click "Create Survey".
Generate Variations
    Navigate to the "Generate Variations" section.
    Enter the Survey ID of the created survey.
    Click "Generate Variations" to create different variations of the survey.
View All Surveys
    Click "View All Surveys" to see a list of all created surveys and their variations.
Start A/B Testing
    Click "AB Test Surveys" to start the A/B testing process.
    The system will split users into groups and present different survey variations to each group.
Clear Database
    Click the "Clear Database" button to remove all surveys and variations from the database.
Visualize Results
    Survey results can be visualized using the Chart.js integration.
    The results will show the effectiveness of different survey designs based on user feedback.
Dataset Information
    Dataset Generation: A synthetic dataset was generated for the project, consisting of various survey questions and user responses. This dataset was used to train machine learning models and perform A/B testing.
    Data Points: The dataset includes survey titles, questions, variations, user responses, and feedback on survey effectiveness.
    Usage: The dataset was utilized to generate survey variations, analyze user feedback, and determine the effectiveness of different survey designs through A/B testing.
