import pandas as pd
import numpy as np

num_rows = 10

data = {
    'survey_id': np.arange(1, num_rows + 1),
    'completion_rate': np.random.choice([0.6, 0.7, 0.8, 0.9], num_rows, p=[0.25, 0.25, 0.25, 0.25]),
    'time_spent_per_question': np.random.choice([25, 30, 35, 45], num_rows, p=[0.25, 0.25, 0.25, 0.25]),
    'response_quality': np.random.choice(['low', 'medium', 'high'], num_rows, p=[0.33, 0.33, 0.34])
}

df = pd.DataFrame(data)

df.to_csv('survey_data.csv', index=False)

print("Dataset generated and saved to 'survey_data.csv'")
