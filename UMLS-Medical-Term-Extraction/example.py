from QuickUMLS.search_term import *
text = ''' x Spinal surgery due to nerve impingement – full recovery
Mild Insomnia, 20XX – on Trazadone
Reflux, BPH. EKG XX/20XX nml Screening colonoscopy, 20XX, with polyps
X/XX/20XX CBC nml CMP nml'''

print(give_med_terms(text))