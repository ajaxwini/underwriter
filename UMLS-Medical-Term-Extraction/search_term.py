from quickumls import *
text = '''
Hx Spinal surgery due to nerve impingement – full recovery
Mild Insomnia, 20XX – on Trazadone
Reflux, BPH. EKG XX/20XX nml Screening colonoscopy, 20XX, with polyps
X/XX/20XX CBC nml CMP nml 
'''

def give_med_terms(text):
	src = '/home/aman/aaa_proj_prac/Untitled Folder/QuickUMLS/quickumlsfiles/'
	matcher = QuickUMLS(src)
	result = matcher.match(text, best_match=True, ignore_syntax=False)
	set_terms = set()
	for i in result:
		for j in i:
			if j['similarity'] > 0.7:
				set_terms.add(j['ngram'])
	#for i in term:
	#	print(i)
	return set_terms

print(give_med_terms(text))
