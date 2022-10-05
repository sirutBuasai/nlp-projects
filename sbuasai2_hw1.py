# External libraries
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from wordcloud import WordCloud

def main():
    """Main function"""
    # load data from csv files
    real_data = pd.read_csv('True.csv')
    fake_data = pd.read_csv('Fake.csv')

    # Task 1 -----------------------------------------------

    # clean the data by removing punctuations and special characters and convert string to lower case
    real_data = real_data.replace(r'[^A-Za-z0-9]+', ' ', regex=True)
    real_data['text'] = real_data['text'].str.lower()

    fake_data = fake_data.replace(r'[^A-Za-z0-9]+', ' ', regex=True)
    fake_data['text'] = fake_data['text'].str.lower()


    # tokenize texts
    real_data['tokenized_text'] = real_data['text'].apply(nltk.tokenize.word_tokenize)
    real_tokens = real_data['tokenized_text'].explode()
    real_tokens.dropna(inplace=True)
    real_tokens = real_tokens.to_list()

    fake_data['tokenized_text'] = fake_data['text'].apply(nltk.tokenize.word_tokenize)
    fake_tokens = fake_data['tokenized_text'].explode()
    fake_tokens.dropna(inplace=True)
    fake_tokens = fake_tokens.to_list()

    collection_tokens = real_tokens + fake_tokens


    # remove stop words
    stop_words = nltk.corpus.stopwords.words('english')
    stop_real = [x for x in real_tokens if x not in stop_words]
    stop_fake = [x for x in fake_tokens if x not in stop_words]
    stop_collection = [x for x in collection_tokens if x not in stop_words]


    # lemmatize tokens
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemm_real = [lemmatizer.lemmatize(x) for x in stop_real]
    lemm_fake = [lemmatizer.lemmatize(x) for x in stop_fake]
    lemm_collection = [lemmatizer.lemmatize(x) for x in stop_collection]


    # retrieve top 100 most common words
    real_freq = nltk.FreqDist(lemm_real).most_common(100)
    fake_freq = nltk.FreqDist(lemm_fake).most_common(100)
    collection_freq = nltk.FreqDist(lemm_collection).most_common(100)


    # download tables to excel spreadsheet
    real_df = pd.DataFrame(data=dict(real_freq), index=[0])
    real_df = (real_df.T)
    real_df.to_excel('real_freq.xlsx')

    fake_df = pd.DataFrame(data=dict(fake_freq), index=[0])
    fake_df = (fake_df.T)
    fake_df.to_excel('fake_freq.xlsx')

    collection_df = pd.DataFrame(data=dict(collection_freq), index=[0])
    collection_df = (collection_df.T)
    collection_df.to_excel('collection_freq.xlsx')


    # create wordcloud for analysis
    real_wordcloud = WordCloud().generate_from_frequencies(dict(real_freq))
    plt.imshow(real_wordcloud)
    plt.show()

    fake_wordcloud = WordCloud().generate_from_frequencies(dict(fake_freq))
    plt.imshow(fake_wordcloud)
    plt.show()

    collection_wordcloud = WordCloud().generate_from_frequencies(dict(collection_freq))
    plt.imshow(collection_wordcloud)
    plt.show()

    # ------------------------------------------------------

    # Task 2 -----------------------------------------------

if __name__ == "__main__":
    main()
