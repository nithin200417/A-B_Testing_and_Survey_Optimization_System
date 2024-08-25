from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import click

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///survey.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db)

@app.cli.command("init-db")
def init_db():
    """Initialize the database."""
    db.create_all()
    click.echo("Initialized the database.")

if __name__ == '__main__':
    app.run(debug=True)