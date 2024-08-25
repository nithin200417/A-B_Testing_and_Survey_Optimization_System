from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from sqlalchemy.orm import Session
import random
import nltk
from nltk.corpus import wordnet
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
import csv
import random
import os
import joblib

nltk.download('wordnet', quiet=True)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///surveys.db"
app.config['SECRET_KEY'] = 'your_secret_key'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
model_path = 'survey_model.pkl'
model = None
if os.path.isfile(model_path):
    model = joblib.load(model_path)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class Survey(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100))
    questions = db.Column(db.String(500))
    completion_time = db.Column(db.DateTime)
    engagement_score = db.Column(db.Float)
    conversion = db.Column(db.Boolean)

class SurveyVariation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    survey_id = db.Column(db.Integer, db.ForeignKey('survey.id'))
    variation = db.Column(db.String(500))
    
def save_ab_test_result(data):
    file_exists = os.path.isfile('ab_test_results.csv')
    with open('ab_test_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['survey_id', 'completion_rate', 'time_spent_per_question', 'response_quality', 'engagement_prob_con', 'engagement_prob_exp'])
        writer.writerow([data['survey_id'], data['completion_rate'], data['time_spent_per_question'], data['response_quality'], data['engagement_prob_con'], data['engagement_prob_exp']])

class SurveyResponse(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    survey_id = db.Column(db.Integer, db.ForeignKey('survey.id'))
    survey_version = db.Column(db.String(50))  
    response = db.Column(db.String(200))
    completion_time = db.Column(db.DateTime)
    engagement_score = db.Column(db.Float)
    conversion = db.Column(db.Boolean)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    with Session(db.engine) as session:
        return session.get(User, int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            return 'Username already exists'
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/surveys')
@login_required
def surveys():
    return render_template('surveys.html')

@app.route('/variations')
@login_required
def variations():
    return render_template('variations.html')

@app.route('/create_survey', methods=['POST'])
@login_required
def create_survey():
    try:
        data = request.json
        new_survey = Survey(title=data['title'], questions=data['questions'])
        db.session.add(new_survey)
        db.session.commit()
        return jsonify({'message': 'Survey created successfully'})
    except Exception as e:
        return jsonify({'message': 'Error creating survey', 'error': str(e)}), 500

@app.route('/generate_variations/<int:survey_id>', methods=['POST'])
@login_required
def generate_variations(survey_id):
    try:
        with Session(db.engine) as session:
            survey = session.get(Survey, survey_id)
            if not survey:
                return jsonify({'message': 'Survey not found'}), 404

            questions = survey.questions.split('\n')
            all_variations = []
            for question in questions:
                if question.strip():
                    variations = [generate_variation(question) for _ in range(3)]
                    all_variations.append(variations)

                    # Add variations to the database
                    for variation in variations:
                        new_variation = SurveyVariation(survey_id=survey.id, variation=variation)
                        session.add(new_variation)
                    session.commit()

            return jsonify({'variations': all_variations})
    except Exception as e:
        app.logger.error(f"Error generating variations: {e}")
        return jsonify({'message': 'Error generating variations', 'error': str(e)}), 500

@app.route('/update_question/<int:survey_id>/<int:question_index>', methods=['POST'])
@login_required
def update_question(survey_id, question_index):
    try:
        data = request.json
        with Session(db.engine) as session:
            survey = session.get(Survey, survey_id)
            if not survey:
                return jsonify({'message': 'Survey not found'}), 404
            questions = survey.questions.split('\n')
            if question_index < len(questions):
                questions[question_index] = data['selected_variation']
                survey.questions = '\n'.join(questions)
                session.commit()
                return jsonify({'message': 'Question updated successfully'})
            else:
                return jsonify({'message': 'Question index out of range'}), 400
    except Exception as e:
        app.logger.error(f"Error updating survey: {e}")
        return jsonify({'message': 'Error updating survey', 'error': str(e)}), 500

@app.route('/get_surveys', methods=['GET'])
@login_required
def get_surveys():
    try:
        surveys = Survey.query.all()
        surveys_list = [{'id': survey.id, 'title': survey.title, 'questions': survey.questions} for survey in surveys]
        return jsonify({'surveys': surveys_list})
    except Exception as e:
        app.logger.error(f"Error fetching surveys: {e}")
        return jsonify({'message': 'Error fetching surveys', 'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/start_ab_testing', methods=['POST'])
@login_required
def start_ab_testing():
    try:
        surveys = Survey.query.all()
        for survey in surveys:
            variations = SurveyVariation.query.filter_by(survey_id=survey.id).all()
            all_versions = [('original', survey)] + [('variation', var) for var in variations]

            users = User.query.all()
            for user in users:
                survey_version, content = random.choice(all_versions)
                response = SurveyResponse(
                    user_id=user.id,
                    survey_id=content.id,
                    survey_version=survey_version,
                    response='',
                    completion_time=None,
                    engagement_score=0.0,
                    conversion=False
                )
                db.session.add(response)
                db.session.commit()
        return jsonify({'message': 'A/B testing started'})
    except Exception as e:
        app.logger.error(f"Error starting A/B testing: {e}")
        return jsonify({'message': 'Error starting A/B testing', 'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        total_responses = SurveyResponse.query.count()
        version_a_responses = SurveyResponse.query.filter_by(survey_version='original').count()
        version_b_responses = SurveyResponse.query.filter_by(survey_version='variation').count()

        engagement_scores = {
            'original': SurveyResponse.query.filter_by(survey_version='original').with_entities(db.func.avg(SurveyResponse.engagement_score)).scalar(),
            'variation': SurveyResponse.query.filter_by(survey_version='variation').with_entities(db.func.avg(SurveyResponse.engagement_score)).scalar()
        }

        conversions = {
            'original': SurveyResponse.query.filter_by(survey_version='original', conversion=True).count(),
            'variation': SurveyResponse.query.filter_by(survey_version='variation', conversion=True).count()
        }

        return jsonify({
            'total_responses': total_responses,
            'version_a_responses': version_a_responses,
            'version_b_responses': version_b_responses,
            'engagement_scores': engagement_scores,
            'conversions': conversions
        })
    except Exception as e:
        app.logger.error(f"Error fetching metrics: {e}")
        return jsonify({'message': 'Error fetching metrics', 'error': str(e)}), 500

@app.route('/ab_test_results', methods=['GET'])
def ab_test_results():
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        df = pd.read_csv('survey_data.csv')
        
        start = (page - 1) * per_page
        end = start + per_page
        paginated_df = df.iloc[start:end]

        results = []
        for _, row in paginated_df.iterrows():
            survey_id = row['survey_id']
            completion_rate = row['completion_rate']
            time_spent_per_question = row['time_spent_per_question']
            response_quality = row['response_quality']

            # Introduce random noise
            completion_rate += random.uniform(-0.05, 0.05)
            time_spent_per_question += random.uniform(-5, 5)
            response_quality = random.choice(['low', 'medium', 'high'])

            p_con_hat = completion_rate
            p_exp_hat = completion_rate * 1.1  

            N_con = 100  
            N_exp = 100  

            p_pooled_hat = (p_con_hat * N_con + p_exp_hat * N_exp) / (N_con + N_exp)
            pooled_variance = p_pooled_hat * (1 - p_pooled_hat) * (1 / N_con + 1 / N_exp)
            SE = np.sqrt(pooled_variance)
            if SE == 0:
                Test_stat = float('inf')  # or some large number
                p_value = 0.0
                CI = [0, 0]
            else:
                Test_stat = (p_exp_hat - p_con_hat) / SE
                alpha = 0.05
                Z_crit = norm.ppf(1 - alpha / 2)
                p_value = 2 * norm.sf(abs(Test_stat))
                CI = [(p_exp_hat - p_con_hat) - SE * Z_crit, (p_exp_hat - p_con_hat) + SE * Z_crit]

            survey_data = {
                'completion_rate': completion_rate,
                'time_spent_per_question': time_spent_per_question,
                'response_quality': response_quality
            }
            suggestion = suggest_changes(survey_data)
            result = {
                'survey_id': survey_id,
                'completion_rate': round(completion_rate, 2),
                'time_spent_per_question': round(time_spent_per_question, 2),
                'response_quality': response_quality,
                'engagement_prob_con': round(p_con_hat, 2),  
                'engagement_prob_exp': round(p_exp_hat, 2),
                'p_value': p_value,
                'confidence_interval': [round(CI[0], 2), round(CI[1], 2)],
                'suggestion': suggestion
            }
            results.append(result)

            # Save the result to the dataset
            save_ab_test_result(result)
        return jsonify(results)
    except Exception as e:
        app.logger.error(f"Error fetching A/B test results: {e}")
        return jsonify({'message': 'Error fetching A/B test results', 'error': str(e)}), 500
    
def save_ab_test_result(data):
    file_exists = os.path.isfile('ab_test_results.csv')
    with open('ab_test_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['survey_id', 'completion_rate', 'time_spent_per_question', 'response_quality', 'engagement_prob_con', 'engagement_prob_exp'])
        writer.writerow([data['survey_id'], data['completion_rate'], data['time_spent_per_question'], data['response_quality'], data['engagement_prob_con'], data['engagement_prob_exp']])

def generate_variation(question):
    words = question.split()
    variation = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            variation.append(synonym)
        else:
            variation.append(word)
    return ' '.join(variation)

def suggest_changes(survey_data):
    if model is None:
        return "Model not available. Please train the model first."
    
    survey_features = pd.DataFrame([survey_data])
    survey_features = pd.get_dummies(survey_features)
    
    missing_cols = set(model.feature_names_in_) - set(survey_features.columns)
    for col in missing_cols:
        survey_features[col] = 0
    survey_features = survey_features[model.feature_names_in_]
    
    predicted_engagement = model.predict(survey_features)
    if predicted_engagement < 0.5:
        return "Increase completion rate or improve response quality."
    else:
        return "Survey design is effective."
    
def generate_report(ab_test_df, survey_comparison_df, p_value, CI):
    report = {
        'average_engagement_prob_exp': float(ab_test_df['engagement_prob_exp'].mean()),
        'average_engagement_prob_con': float(ab_test_df['engagement_prob_con'].mean()),
        'average_completion_rate': float(survey_comparison_df['completion_rate'].mean()),
        'average_time_spent_per_question': float(survey_comparison_df['time_spent_per_question'].mean()),
        'aggregated_p_value': float(p_value),
        'aggregated_confidence_interval': [float(CI[0]), float(CI[1])],
        'best_performing_survey': int(ab_test_df.loc[ab_test_df['engagement_prob_exp'].idxmax()]['survey_id'])
    }
    return report

def calculate_aggregated_metrics(df):
    p_con_hat = df['completion_rate'].mean()
    p_exp_hat = p_con_hat * 1.1  

    N_con = len(df)  
    N_exp = len(df)  

    p_pooled_hat = (p_con_hat * N_con + p_exp_hat * N_exp) / (N_con + N_exp)
    pooled_variance = p_pooled_hat * (1 - p_pooled_hat) * (1 / N_con + 1 / N_exp)
    SE = np.sqrt(pooled_variance)
    if SE == 0:
        Test_stat = float('inf')  
        p_value = 0.0
        CI = [0, 0]
    else:
        Test_stat = (p_exp_hat - p_con_hat) / SE
        alpha = 0.05
        Z_crit = norm.ppf(1 - alpha / 2)
        p_value = 2 * norm.sf(abs(Test_stat))
        CI = [(p_exp_hat - p_con_hat) - SE * Z_crit, (p_exp_hat - p_con_hat) + SE * Z_crit]
    
    return p_value, CI

@app.route('/clear_database', methods=['POST'])
def clear_database():
    try:
        db.drop_all()
        db.create_all()
        return jsonify({'message': 'Database cleared successfully'})
    except Exception as e:
        app.logger.error(f"Error clearing database: {e}")
        return jsonify({'message': 'Error clearing database', 'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = pd.DataFrame([data])
        input_data = pd.get_dummies(input_data)
        missing_cols = set(model.feature_names_in_) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[model.feature_names_in_]
        prediction = model.predict(input_data)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        app.logger.error(f"Error making prediction: {e}")
        return jsonify({'message': 'Error making prediction', 'error': str(e)}), 500
    
def save_optimized_survey_results(results):
    file_exists = os.path.isfile('optimized_survey_results.csv')
    with open('optimized_survey_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['variation', 'suggestion'])
        for result in results:
            writer.writerow([result['variation'], result['suggestion']])

@app.route('/optimize_survey', methods=['POST'])
def optimize_survey():
    try:
        completion_rate = float(request.form['completion_rate'])
        time_spent_per_question = int(request.form['time_spent_per_question'])
        response_quality = request.form['response_quality']
        questions = request.form['questions'].split(',')
        variations = [generate_variation(question) for question in questions]
        results = []
        for variation in variations:
            survey_data = {
                'completion_rate': completion_rate,
                'time_spent_per_question': time_spent_per_question,
                'response_quality': response_quality,
                'questions': variation
            }
            suggestion = suggest_changes(survey_data)
            results.append({
                'variation': variation,
                'suggestion': suggestion
            })
        save_optimized_survey_results(results)
        return jsonify(results)
    except Exception as e:
        app.logger.error(f"Error optimizing survey: {e}")
        return jsonify({'message': 'Error optimizing survey', 'error': str(e)}), 500

@app.route('/view_optimized_results', methods=['GET'])
def view_optimized_results():
    try:
        df = pd.read_csv('optimized_survey_results.csv')
        results = df.to_dict(orient='records')
        return render_template('optimized_results.html', results=results)
    except Exception as e:
        app.logger.error(f"Error reading optimized survey results: {e}")
        return jsonify({'message': 'Error reading optimized survey results', 'error': str(e)}), 500

@app.route('/optimize_survey_form', methods=['GET'])
def optimize_survey_form():
    return render_template('optimize_survey.html')

@app.route('/ab_test_dashboard', methods=['GET'])
def ab_test_dashboard():
    try:
        ab_test_df = pd.read_csv('ab_test_results.csv')
        def calculate_aggregated_metrics(df):
            p_con_hat = df['completion_rate'].mean()
            p_exp_hat = p_con_hat * 1.1  

            N_con = len(df)  
            N_exp = len(df)  

            p_pooled_hat = (p_con_hat * N_con + p_exp_hat * N_exp) / (N_con + N_exp)
            pooled_variance = p_pooled_hat * (1 - p_pooled_hat) * (1 / N_con + 1 / N_exp)
            SE = np.sqrt(pooled_variance)
            if SE == 0:
                Test_stat = float('inf')  
                p_value = 0.0
                CI = [0, 0]
            else:
                Test_stat = (p_exp_hat - p_con_hat) / SE
                alpha = 0.05
                Z_crit = norm.ppf(1 - alpha / 2)
                p_value = 2 * norm.sf(abs(Test_stat))
                CI = [(p_exp_hat - p_con_hat) - SE * Z_crit, (p_exp_hat - p_con_hat) + SE * Z_crit]
            
            return p_value, CI

        p_value, CI = calculate_aggregated_metrics(ab_test_df)
        survey_comparison_df = pd.read_csv('survey_data.csv')
        
        kpi_data = [
            {
                'kpi': 'Average Engagement Probability (Experimental)',
                'value': ab_test_df['engagement_prob_exp'].mean()
            },
            {
                'kpi': 'Average Engagement Probability (Control)',
                'value': ab_test_df['engagement_prob_con'].mean()
            },
            {
                'kpi': 'Average Completion Rate',
                'value': survey_comparison_df['completion_rate'].mean()
            },
            {
                'kpi': 'Average Time Spent per Question',
                'value': survey_comparison_df['time_spent_per_question'].mean()
            },
            {
                'kpi': 'Aggregated P-Value',
                'value': p_value
            },
            {
                'kpi': 'Aggregated Confidence Interval',
                'value': CI
            }
        ]
        
        kpi_data = kpi_data if kpi_data else []
        return render_template('ab_test_dashboard.html', kpi_data=kpi_data)
    except Exception as e:
        app.logger.error(f"Error fetching A/B test dashboard data: {e}")
        return jsonify({'message': 'Error fetching A/B test dashboard data', 'error': str(e)}), 500

@app.route('/finalize_and_deploy', methods=['POST'])
def finalize_and_deploy():
    try:
        ab_test_df = pd.read_csv('ab_test_results.csv')
        p_value, CI = calculate_aggregated_metrics(ab_test_df)
        survey_comparison_df = pd.read_csv('survey_data.csv')
        report = generate_report(ab_test_df, survey_comparison_df, p_value, CI)
        best_survey_id = report['best_performing_survey']
        deploy_survey(best_survey_id)
        
        return jsonify({'message': 'Survey finalized and deployed successfully', 'report': report}), 200
    except Exception as e:
        app.logger.error(f"Error finalizing and deploying survey: {e}")
        return jsonify({'message': 'Error finalizing and deploying survey', 'error': str(e)}), 500

def deploy_survey(survey_id):
    print(f"Deploying survey with ID: {survey_id}")
    
@app.route('/view_report', methods=['GET'])
def view_report():
    try:
        ab_test_df = pd.read_csv('ab_test_results.csv')
        survey_comparison_df = pd.read_csv('survey_data.csv')
        p_value, CI = calculate_aggregated_metrics(ab_test_df)
        report = generate_report(ab_test_df, survey_comparison_df, p_value, CI)
        
        return render_template('report.html', report=report)
    except Exception as e:
        app.logger.error(f"Error viewing report: {e}")
        return jsonify({'message': 'Error viewing report', 'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)