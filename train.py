import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, cohen_kappa_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from parse import parse_texts

# Make sure the models directory exists
os.makedirs('models', exist_ok=True)

# Load and preprocess data
data = pd.read_csv('fake_or_real_news.csv')
data = data.drop([data.columns[0], 'title'], axis=1)
data['label'] = data['label'].replace({'FAKE': 0, 'REAL': 1})
data = data.sample(frac=1, random_state=25).reset_index(drop=True)

print("Total Entries: ", len(data))
print("Number of fake news articles: ", len(data[data['label'] == 0]))
print("Number of real news articles: ", len(data[data['label'] == 1]))

data['parsed_text'] = parse_texts(data['text'])

X = data['parsed_text']  
y = data['label']

# Split that data using an 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

# Validate train/test split with visualization
def validate_train_test_split(y_train, y_test):
    # Get label counts for train and test sets
    train_counts = pd.Series(y_train).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()
    
    # Calculate percentages
    train_pct = train_counts / len(y_train) * 100
    test_pct = test_counts / len(y_test) * 100
    
    # Set up the figure
    plt.figure(figsize=(10, 6))
    
    # Plotting chart w/ counts
    plt.subplot(1, 2, 1)
    x = np.arange(2)
    width = 0.35
    
    plt.bar(x - width/2, train_counts, width, label='Train',color='#6613F7')
    plt.bar(x + width/2, test_counts, width, label='Test', color ='#F6F14E')
    
    # Add labels and title
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Outcome Distribution in Train/Test Sets')
    plt.xticks(x, ['FAKE (0)', 'REAL (1)'])
    
    # Add percentage labels above bars
    for i, count in enumerate(train_counts):
        plt.text(i - width/2, count + 10, f"{count}", ha='center')
    
    for i, count in enumerate(test_counts):
        plt.text(i + width/2, count + 10, f"{count}", ha='center')
    
    plt.legend()
    
    # Percentage split chart
    plt.subplot(1, 2, 2)
    
    # Show 80/20 split with pie charts
    plt.pie([len(y_train), len(y_test)], 
            labels=[f'Train ({len(y_train)}, {len (y_train)/ (len(y_train) + len(y_test)):.1%})', 
                   f'Test ({len(y_test)}, {len(y_test) / (len(y_train) + len(y_test)):.1%})'],
            autopct='%1.1f%%',
            colors=['#6613F7', '#F6F14E'], 
            startangle=90)
    plt.title('Train/Test Split Ratio')
    
    plt.tight_layout()
    plt.savefig('models/train_test_split_validation.png')


# Call the validation function
validate_train_test_split(y_train, y_test)

# Vectorize the text data 
print("Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

 
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("Vectorizer saved to models/tfidf_vectorizer.pkl")


model_accuracies = {}

# Function to train, evaluate and save models
def train_evaluate_save_model(model, model_name):
    print(f"Training {model_name}...")
    model.fit(X_train_vectorized, y_train)
    
    y_pred = model.predict(X_test_vectorized)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[model_name] = accuracy
    f1 = f1_score(y_test, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_test, y_pred)

    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Cohen's Kappa: {kappa:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=['FAKE', 'REAL'], 
                yticklabels=['FAKE', 'REAL'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(f'models/{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))

    with open(f'models/{model_name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"{model_name} model saved to models/{model_name.lower().replace(' ', '_')}_model.pkl")

# Define and train models
# TODO: fine-tune hyper parameters
models = {
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=25),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=25),
    'Linear SVC': LinearSVC(random_state=25),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=25),
    'Decision Tree': DecisionTreeClassifier(random_state=25)
}

# Train, evaluate and save each model
for model_name, model in models.items():
    train_evaluate_save_model(model, model_name)

# Plot model accuracy comparison
def plot_model_accuracy_comparison(model_accuracies):
    plt.figure(figsize=(12, 6))
    
    # Sort models by accuracy for better visualization
    # This line was specifically was AI assisted
    sorted_models = dict(sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True))
    
    colors = cm.tab20(np.linspace(0, 1, 5)) 
    # Create bar chart
    bars = plt.bar(sorted_models.keys(), sorted_models.values(), color=colors)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig('models/model_accuracy_comparison.png')

# Generate the model accuracy comparison plot
plot_model_accuracy_comparison(model_accuracies)

print("All models have been trained, evaluated, and saved :)")