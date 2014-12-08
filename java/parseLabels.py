f = open('neural.out')
lines = f.readlines()
f.close()

O_words = dict({'O': 0.0, 'ORG': 0.0, 'PER': 0.0, 'MISC': 0.0, 'LOC': 0.0})
ORG_words = dict({'O': 0.0, 'ORG': 0.0, 'PER': 0.0, 'MISC': 0.0, 'LOC': 0.0})
PER_words = dict({'O': 0.0, 'ORG': 0.0, 'PER': 0.0, 'MISC': 0.0, 'LOC': 0.0})
MISC_words = dict({'O': 0.0, 'ORG': 0.0, 'PER': 0.0, 'MISC': 0.0, 'LOC': 0.0})
LOC_words = dict({'O': 0.0, 'ORG': 0.0, 'PER': 0.0, 'MISC': 0.0, 'LOC': 0.0})

for line in lines:
	line = line.rstrip()
	labels = line.split('\t')
	if labels[1] == "O":
		O_words[labels[2]] += 1.0
	if labels[1] == "ORG":
		ORG_words[labels[2]] += 1.0
	if labels[1] == "PER":
		PER_words[labels[2]] += 1.0
	if labels[1] == "MISC":
		MISC_words[labels[2]] += 1.0
	if labels[1] == "LOC":
		LOC_words[labels[2]] += 1.0

print "\nO values:" 
print "---------------------------------------------" 
print O_words
O_total = 0.0
for val in O_words:
	O_total += O_words[val]
print O_total
for val in O_words:
	O_words[val] = O_words[val] / O_total
print O_words

print "\nORG values: " 
print "---------------------------------------------" 
print ORG_words
ORG_total = 0.0
for val in ORG_words:
	ORG_total += ORG_words[val]
print ORG_total
for val in ORG_words:
	ORG_words[val] = ORG_words[val] / ORG_total
print ORG_words

print "\nPER values: " 
print "---------------------------------------------" 
print PER_words
PER_total = 0.0
for val in PER_words:
	PER_total += PER_words[val]
print PER_total
for val in PER_words:
	PER_words[val] = PER_words[val] / PER_total
print PER_words

print "\nMISC values: " 
print "---------------------------------------------" 
print MISC_words
MISC_total = 0.0
for val in MISC_words:
	MISC_total += MISC_words[val]
print MISC_total
for val in MISC_words:
	MISC_words[val] = MISC_words[val] / MISC_total
print MISC_words

print "\nLOC values: " 
print "---------------------------------------------" 
print LOC_words
LOC_total = 0.0
for val in LOC_words:
	LOC_total += LOC_words[val]
print LOC_total
for val in LOC_words:
	LOC_words[val] = LOC_words[val] / LOC_total
print LOC_words






