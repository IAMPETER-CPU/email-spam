from django.shortcuts import render,redirect
from django.conf import settings

# Create your views here.
import pandas as pd
from pathlib import Path
from django.shortcuts import render
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from .forms import MessageForm



DATASET_PATH = Path(settings.BASE_DIR) / 'data' / 'emails.csv'

dataset = pd.read_csv(DATASET_PATH)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dataset['text'])
X_train, X_test, y_train, y_test = train_test_split(X, dataset['spam'], test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)


def predict_message(message):
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)
    return 'Spam' if prediction[0] == 1 else 'Ham'


def Home(request):
    if request.method == 'POST':
        form = MessageForm(request.POST)
        if form.is_valid():
            message = form.cleaned_data['text']
            result = predict_message(message)

            # store result temporarily
            request.session['result'] = result
            return redirect('home')  # 👈 KEY LINE

    # GET request
    result = request.session.pop('result', None)
    form = MessageForm()

    return render(request, 'home.html', {
        'form': form,
        'result': result,
        'submitted': result is not None
    })
