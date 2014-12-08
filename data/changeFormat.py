

f = open('trimmedTrain.txt')
lines = f.readlines()
f.close()

f = open('sentences.txt','w')


sentences = [];
sentence = ""
for s in lines:
	if s.strip() == '':
		f.write(sentence + "\n")
		sentence = ''
	else:
		sentence += s.strip() + ' '

f.close()