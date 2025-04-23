import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def train_and_save_models():
    print("Loading datasets...")
    # Load your datasets
    fake = pd.read_csv('Fake.csv')
    true = pd.read_csv('True.csv')
    
    # Add labels
    fake['outcome'] = 0
    true['outcome'] = 1
    
    # Merge datasets
    merged = pd.concat([fake, true], ignore_index=True)
    
    # Keep only text and outcome columns
    df = merged.drop(['title', 'subject', 'date'], axis=1)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("Preparing data for training...")
    # Split into features and target
    x = df['text']
    y = df['outcome']
    
    # Split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    
    # Vectorize the text
    print("Vectorizing text data...")
    vectorization = TfidfVectorizer()
    xvector_train = vectorization.fit_transform(x_train)
    xvector_test = vectorization.transform(x_test)
    
    # Train Decision Tree model
    print("Training Decision Tree model...")
    tree = DecisionTreeClassifier(random_state=1)
    tree.fit(xvector_train, y_train)
    tree_score = tree.score(xvector_test, y_test)
    print(f"Decision Tree accuracy: {tree_score:.4f}")
    
    # Train Random Forest model
    print("Training Random Forest model...")
    rf = RandomForestClassifier(random_state=1)
    rf.fit(xvector_train, y_train)
    rf_score = rf.score(xvector_test, y_test)
    print(f"Random Forest accuracy: {rf_score:.4f}")
    
    # Train Neural Network model
    print("Training Neural Network model...")
    nn = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=1)
    nn.fit(xvector_train, y_train)
    nn_score = nn.score(xvector_test, y_test)
    print(f"Neural Network accuracy: {nn_score:.4f}")
    
    # Save models and vectorizer
    print("Saving models...")
    joblib.dump(tree, 'models/decision_tree_model.pkl')
    joblib.dump(rf, 'models/random_forest_model.pkl')
    joblib.dump(nn, 'models/neural_network_model.pkl')
    joblib.dump(vectorization, 'models/tfidf_vectorizer.pkl')
    
    print("Models trained and saved successfully!")
    
if __name__ == "__main__":
    train_and_save_models()