from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
import random
from datetime import datetime
from app import db, Survey, SurveyVariation, SurveyResponse, User
from flask_migrate import Migrate

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///survey.db'
db = SQLAlchemy(app)
socketio = SocketIO(app)
migrate = Migrate(app, db)

class SurveyResponse(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50))
    survey_id = db.Column(db.Integer, db.ForeignKey('survey.id'))
    survey_version = db.Column(db.String(50))
    response = db.Column(db.String(200))
    completion_time = db.Column(db.DateTime)
    engagement_score = db.Column(db.Float)
    conversion = db.Column(db.Boolean)

@app.route('/assign_survey', methods=['POST'])
def assign_survey():
    user_id = request.json['user_id']
    survey_versions = ['A', 'B']
    assigned_version = random.choice(survey_versions)
    
    new_response = SurveyResponse(
        user_id=user_id,
        survey_version=assigned_version,
        response='',
        completion_time=None,
        engagement_score=0.0,
        conversion=False
    )
    db.session.add(new_response)
    db.session.commit()
    
    return jsonify({'user_id': user_id, 'assigned_version': assigned_version})

@app.route('/submit_response', methods=['POST'])
def submit_response():
    data = request.json
    response = SurveyResponse.query.filter_by(user_id=data['user_id'], survey_version=data['survey_version']).first()
    if response:
        response.response = data['response']
        response.completion_time = datetime.utcnow()
        response.engagement_score = data.get('engagement_score', 0.0)
        db.session.commit()
        socketio.emit('update_dashboard', {
            'user_id': data['user_id'],
            'survey_version': data['survey_version'],
            'engagement_score': response.engagement_score
        })
        return jsonify({'message': 'Response submitted successfully!'})
    return jsonify({'message': 'Response not found'}), 404

@app.route('/metrics', methods=['GET'])
def get_metrics():
    total_responses = SurveyResponse.query.count()
    version_a_responses = SurveyResponse.query.filter_by(survey_version='A').count()
    version_b_responses = SurveyResponse.query.filter_by(survey_version='B').count()
    
    engagement_scores = {
        'A': SurveyResponse.query.filter_by(survey_version='A').with_entities(db.func.avg(SurveyResponse.engagement_score)).scalar(),
        'B': SurveyResponse.query.filter_by(survey_version='B').with_entities(db.func.avg(SurveyResponse.engagement_score)).scalar()
    }

    conversions = {
        'A': SurveyResponse.query.filter_by(survey_version='A', conversion=True).count(),
        'B': SurveyResponse.query.filter_by(survey_version='B', conversion=True).count()
    }
    
    return jsonify({
        'total_responses': total_responses,
        'version_a_responses': version_a_responses,
        'version_b_responses': version_b_responses,
        'engagement_scores': engagement_scores,
        'conversions': conversions
    })

@app.route('/start_ab_testing', methods=['POST'])
#@login_required
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

@app.route('/reset_surveys', methods=['POST'])
def reset_surveys():
    try:
        db.drop_all()
        db.create_all()
        new_surveys = [
            {'title': 'Survey 1', 'questions': 'Question 1, Question 2, Question 3'},
            {'title': 'Survey 2', 'questions': 'Question 4, Question 5, Question 6'}
        ]

        for survey_data in new_surveys:
            survey = Survey(title=survey_data['title'], questions=survey_data['questions'])
            db.session.add(survey)
            db.session.commit()

            # Step 4: Add variations for each survey
            variations = [
                {'survey_id': survey.id, 'variation': 'Variation 1 for ' + survey.title},
                {'survey_id': survey.id, 'variation': 'Variation 2 for ' + survey.title}
            ]

            for var_data in variations:
                variation = SurveyVariation(survey_id=var_data['survey_id'], variation=var_data['variation'])
                db.session.add(variation)
            db.session.commit()

        return jsonify({'message': 'Surveys reset and new surveys added successfully'})
    except Exception as e:
        app.logger.error(f"Error resetting surveys: {e}")
        return jsonify({'message': 'Error resetting surveys', 'error': str(e)}), 500

@app.route('/clear_database', methods=['POST'])
def clear_database():
    try:
        db.drop_all()
        db.create_all()
        
        return jsonify({'message': 'Database cleared successfully'})
    except Exception as e:
        app.logger.error(f"Error clearing database: {e}")
        return jsonify({'message': 'Error clearing database', 'error': str(e)}), 500

def emit_metrics():
    metrics = {
        'engagement_scores': {
            'A': SurveyResponse.query.filter_by(survey_version='A').with_entities(db.func.avg(SurveyResponse.engagement_score)).scalar(),
            'B': SurveyResponse.query.filter_by(survey_version='B').with_entities(db.func.avg(SurveyResponse.engagement_score)).scalar()
        },
        'conversions': {
            'A': SurveyResponse.query.filter_by(survey_version='A', conversion=True).count(),
            'B': SurveyResponse.query.filter_by(survey_version='B', conversion=True).count()
        }
    }
    socketio.emit('update_chart', metrics)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True)