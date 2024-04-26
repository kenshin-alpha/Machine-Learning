# Which columns will you use?
# Clean your columns
# Concatenate the columns needed for your embedding
# Create new column with concatenated and clean text
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# Clean text data
def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))  # Remove non-alphabetic characters
    text = text.lower()  # Convert text to lowercase
    return text


def clean_data(imdb_data):

    # # Load IMDb dataset
    # imdb_data = pd.read_csv("imdb_top_1000.csv")

    # Select only the specified columns
    selected_columns = ['Poster_Link', 'Series_Title', 'Released_Year', 'Certificate', 'Runtime',
                        'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 'Director', 'Star1',
                        'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Gross']
    imdb_data = imdb_data[selected_columns]



    imdb_data['Overview'] = imdb_data['Overview'].apply(clean_text)

    # Handle missing values
    imdb_data.dropna(subset=['IMDB_Rating', 'Meta_score'], inplace=True)  # Drop rows with missing rating values

    # Impute missing numerical values
    imputer = SimpleImputer(strategy='mean')
    imdb_data[['IMDB_Rating', 'Meta_score']] = imputer.fit_transform(imdb_data[['IMDB_Rating', 'Meta_score']])
    imdb_data['No_of_Votes'].fillna(0, inplace=True)  # Assuming missing votes means no votes
    imdb_data['Gross'].fillna(0, inplace=True)  # Assuming missing gross means no revenue

    # Save cleaned dataset
    imdb_data.to_csv("cleaned_imdb_data.csv", index=False)

    # Load IMDb dataset

    # Select columns needed for embedding
    selected_columns = ['Series_Title', 'Released_Year', 'Certificate', 'Runtime', 'Genre',
                        'IMDB_Rating', 'Overview', 'Meta_score', 'Director', 'Star1', 'Star2',
                        'Star3', 'Star4', 'No_of_Votes', 'Gross']

    # Concatenate selected columns
    imdb_data['Concatenated_Text'] = imdb_data[selected_columns].astype(str).apply(','.join, axis=1)

    # Save processed dataset
    imdb_data.to_csv("processed_imdb_data.csv", index=False)
