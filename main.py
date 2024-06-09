import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Telecharger les ressources NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Lire le fichier
with open('D:\\NLP_Fall_Forward\\Data\\Fall_Forward_speeche.txt', 'r', encoding='utf-8') as file:
    speech = file.read()
    
# fonction de preprocessing
def preprocess_speech(text):
    # insensible à la casse
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Supprimer les stopwords et les mots de moins de 3 caractères "ponctuations comprises"
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word.isalnum() and word not in stop_words and len(word) > 2]
    return words

words = preprocess_speech(speech)

# Calculer la fréquence des mots
fdist = FreqDist(words)
fdist.plot(30, cumulative=False)
common_words = fdist.most_common(30)
print('Les mots les plus frequents', common_words)

# # Taux de positivité et de négativité des mots
# positive_words = ['good', 'great', 'excellent', 'amazing', 'awesome', 'fantastic', 'outstanding', 'incredible', 'perfect', 'love']
# negative_words = ['bad', 'terrible', 'horrible', 'awful', 'worst', 'disgusting', 'hate']
# positive_count = 0
# negative_count = 0
# for word in words:
#     if word in positive_words:
#         positive_count += 1
#     elif word in negative_words:
#         negative_count += 1
# print('Taux de positivité:', positive_count / len(words))
# print('Taux de négativité:', negative_count / len(words))

# Nuage de mots
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(' '.join(words))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Wordcloud des mots les plus pertinents du discours de Denzel Washington "Fall Forward"')
plt.axis('off')
plt.show()

# sauvegarder les resultats 
with open('D:\\NLP_Fall_Forward\\Results\\common_words.txt', 'w', encoding='utf-8') as file:
    for word in common_words:
        file.write(f'{word[0]}: {word[1]}\n')
    # file.write(f'Taux de positivité: {positive_count / len(words)}\n')
    # file.write(f'Taux de négativité: {negative_count / len(words)}\n')
    # file.write('Nuage de mots généré avec succès\n')
    
wordcloud.to_file('results/wordcloud.png')
    
