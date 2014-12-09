# import modules & set up logging
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


sentences = gensim.models.word2vec.LineSentence("sentences.txt")

f = open('sentences.txt')
lines = f.readlines();
all_words = ""
for line in lines:
	words = line.split(' ')
	for word in words:
		if not word.strip() == "":
			all_words += word + " "

all_words = all_words.split(' ')

model = gensim.models.Word2Vec(sentences, size=50, min_count = 1)
f = open('vocab.txt')
vocab = f.readlines()
f.close()

f = open('temp.txt','w')
word2vec_vocab = ""
for word in vocab:
	if word.strip() in all_words and not word.strip().isspace():
		word2vec_vocab += word.strip() + " "
		s = str(model[word.strip()])
		f.write(s + "\n")
f.close()

word2vec_vocab = word2vec_vocab.split(" ")
word2vec_vocab = word2vec_vocab[:-1]

# gather trained vocab
f = open('word2vecVocab.txt', 'w')
for word in word2vec_vocab:
	f.write(word + "\n")
f.close()

f = open('temp.txt')
lines = f.readlines();
f.close()

# format output
f = open('word2vec.txt', 'w')
vec = ""
for line in lines:
	line = line.rstrip('\n')
	if line.endswith("]"):
		vec += line
		vec = vec[2:-1]
		f.write(vec + "\n")
		vec = ""
	else:
		vec += line
f.close()