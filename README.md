# udacity_mlops_project3
https://github.com/kikaka/udacity_mlops_project3

Project for prediction if a salary is above 50K per year based to pass udacity mlops project 3.

# Environment Set up
* Find the package requirements under requirements.txt

## GitHub Actions
* There is configured one GitHub Action based on the pre-made GitHub Action for python applications
* it runs pytest and flake8 on push and requires both to pass without error
* it uses also dvc for data versioning and pulls the data via dvc from an dvc AWS S3 remote repository

## Model
* It is a Logistic regression model, trained with default hyperparamters in scikit-learn 1.0.2  
* It is trained on census.csv, information on the dataset can be found <a href="https://archive.ics.uci.edu/ml/datasets/census+income" target="_blank">here</a>.
* You find details of the model in the model_card.md file
* 
## API Deployment
* The API is deployed on Heroku from my GitHub repository automaticly  deployments that only deploy if your continuous integration passes.
   * Hint: think about how paths will differ in your local environment vs. on Heroku.
   * Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
* Set up DVC on Heroku using the instructions contained in the starter directory.
* Set up access to AWS on Heroku, if using the CLI: `heroku config:set AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=yyy`
* Write a script that uses the requests module to do one POST on your live API.

## API Usage
* The API make predictions on posted data
* Find API Documentation under https://udacity-project-3-kaue.herokuapp.com/docs

