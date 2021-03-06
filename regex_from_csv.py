# -*- coding: utf-8 -*-

import csv
from collections import defaultdict

import re
from datetime import datetime

number = r'\d{2,3}'

gender = r'(\b[Mm]ale?)|(\b[Ff]emale?)|(\bFEMALE)|(\bMALE)|(/b)|F/|M/'

Date = r'(([A-Z0-9][A-Z0-9]?[/-])?[A-Z0-9][A-Z0-9]?[/-][A-Z0-9][A-Z0-9][A-Z0-9]?[A-Z0-9]?)|([A-Za-z][A-Za-z][A-Za-z]\s..?[,]\s....)'

DOB  = r'(.*)?DOB|[Dd][aA][tT][eE]\s[oO][fF]\s[Bb][iI][rR][tT][hH]\s?(.*)?'

year_four_digit = r'\b(19|20)\d{2}(w+)?'
year_two_digit = r'\d{2}$(w+)?'

product_type = r'(\b[Pp]roduct\s[Tt]ype):\s?.*'
permanent = r'[Pp][eE][rR][mM]([aA][nN][aA][nN][tT])?'
term = r'[tT][eE][rR][mM]'

#Assuming USA currency dollar
amount_with_dollar = r'(\$\s?\d{1,3}(,\d{2,3})*(\.\d+)?)(\s?[kK]?)(\s?[mM]?[mM]?(illion)?(ILLION)?)([bB]?)'
amount_without_dollar = r'(\$?\s?\d{1,3}(,\d{2,3})*(\.\d+)?)(\s?[kK]?)(\s[mM]?[mM]?(illion)?(ILLION)?)([bB]?)((\s?[Yy][Ee][aA][rR][sS]?)?)'
faceamount = r'(\b[Ff]ace\s?[Aa]mount:?\s?.*)'
termamount = r'(.*)?[Tt][eE][rR][mM](.*)?'   			#Regex to read single line from first newline to next newline
seeking = r'(.*)?[Ss][eE][eE][kK]([iI][nN][gG])?(.*)?'
term_year = r'(y(ea)?r|Y(ea)?r|Y(ea)?r)'

weight = r'(.*)?\b[wW][eE][iI][gG][hH][tT]\s?(.*)?' 
weight_num = r'(\d*\.?\d+)\s?(lb|lbs|Lbs|LB|LBS|kg|Kg|KG|#)'		#r'(.*)\s?([lL][bB][sS]|[oO][zZ]|[gG]|[kK][Gg])' 

age_simple = r'(.*)?[Aa][Gg][Ee]\s?(.*)?'
age = r'(.*\s?[Yy]([eE][aA])?[rR]?[sS]?\s?([oO][lL][dD])?)'
age_from_gender = r'(.*)?(\b[Mm]ale?)|(\b[Ff]emale?)|(\bFEMALE)|(\bMALE)|(/b)\s?(.*)?' 

height_num = r'\d{1,2}'
height1 = r'((.*)?\s?([Ff][eE][eE][tT])((.*)?\s?([iI][nN][Cc][Hh][Ee][Ss]))?)'			#Two types of inches => "|”
height2 = r'.[\'|\’](\s?.[\"|\”])?' 											
feet = r'\d[\'|\’]'
inches = r'\d[\"|\”]' 

preferred = r'(.*)?(Preferred|preferred)\s?(.*)?'
height_word = r'Height|height'
weight_word = r'Weight|weight' 

build = r'(Build|build)\s?(.*)?'
build_weight = r'\d{3}'
build_height = r'\d\.\d'

smoker = r'(.*)?[sS][Mm][oO][Kk]\s?(.*)?' 
tobacco = r'(.*)?[Tt][oO][bB][aA][cC][cC][oO]\s?(.*)?'
no = r'[nN][oO]'

med = r'(.*)?\b[mM][eE][dD][iI][cC][aA][tT][iI][oO][nN]\s?(.*)?'

family = r'(.*)?(\b[Ff]amily)\s?(.*)?'
family_member = r'(.*)?(\b[Mm]om)|(\b[Ff]ather)|(\b[Dd]ad)|(\b[Ss]ister)|(\b[Bb]rother)|(\b[Hh]usband)|(\b[Ww]ife)\s?(.*)?'

lives = r'(.*)?(\b[Ll]ives)\s?(.*)?'
prop = r'(.*)?(\b[Pp]roperty)\s?(.*)?'

def reg(st,i):
	for line in st: #iterate through every line
		#return list of entities in that line
		num = re.search(number, line, re.I | re.U)
		x=0
		x = re.search(Date, line, re.I | re.U)
		'''if(x):
			print (x.group(0)+"\n")
			data[i][0]=(x.group(0))
		else:
			data[i][0]=" "
		'''
		
#-----------------------------------------------------------------		
		#Gender
		y = re.search(gender, line, re.I | re.U)
		if(y):
			if(y.group(0)=='F/'):
				data[i][0]='Female'
			elif(y.group(0)=='M/'):
				data[i][0]='Male'
			else:
				#print (y.group(0)+"\n")
				data[i][0]=(y.group(0))
		elif(y and num):
			data[i][0]=(y.group(0))
		else:
			data[i][0]=" "
		
#-----------------------------------------------------------------		
		#Year for DOB
		z = 0
		x1 = re.search(year_four_digit, line, re.I | re.U)
		if(x):
			x1 = re.search(year_four_digit, x.group(0), re.I | re.U)
			x2 = re.search(year_two_digit, x.group(0), re.I | re.U)				
			if(x1):
				data[i][1]=x1.group(0)
			elif(x2):
				z = x2.group(0)									
				#print('Last 2 digits of Year of birth='+z)
				data[i][1] = '19'+z
		elif(x1):
			x1 = re.search(year_four_digit, line, re.I | re.U)
			#print (x1.group(0))
			data[i][1]=x1.group(0)
		else:
			data[i][1]=" "
		
		
#-----------------------------------------------------------------		
		#Age in years
		age_reg = re.search(age, line, re.I | re.U)
		age_simple_reg = re.search(age_simple, line, re.I | re.U)
		dob = re.search(DOB, line, re.I | re.U)
		age_gender_reg = re.search(age_from_gender, line, re.I | re.U)
		
		if(age_gender_reg):										#Male 20
				am = re.search(number, age_gender_reg.group(0), re.I | re.U)
				if(am):
					data[i][2]=am.group(0)
			
		if(x1):													#20/03/1996
			#print ("DOB:"+x1.group(0))
			currentYear = datetime.now().year
			#print (currentYear-(int)(x1.group(0)))
			data[i][2]=((currentYear-(int)(x1.group(0))))
		
		else:
			if(x1 and dob):										#DOB 20/03/1996
				#print ("DOB:"+x1.group(0))
				currentYear = datetime.now().year
				#print (currentYear-(int)(x1.group(0)))
				data[i][2]=((currentYear-(int)(x1.group(0))))
			elif(x1 and y):										#Male 20/03/1996
				#print ("DOB:"+x1.group(0))
				currentYear = datetime.now().year
				#print (currentYear-(int)(x1.group(0)))
				data[i][2]=((currentYear-(int)(x1.group(0))))
			
			else:
				data[i][2]=' '
				
			if(age_reg):										#20 years ago
				age_num = age_reg.group(0)			
				an = re.search(number, age_num, re.I | re.U)
				if(an):				
					
					#print ("DOB:"+ an.group(0))
					data[i][2]=(an.group(0))
		
			if(age_simple_reg):								#Age 20
				age_num = age_simple_reg.group(0)			
				an = re.search(number, age_num, re.I | re.U)
				if(an):				
					#print ("DOB:"+ an.group(0))
					data[i][2]=(an.group(0))
				
		
			if((data[i][2]<'18') and data[i][1]!=" "):								#From Year of Birth
				currentYear = datetime.now().year
				data[i][2]=(currentYear-(int)(data[i][1]))
		
		
#-----------------------------------------------------------------		
		#Product Type
		z=re.search(product_type, line, re.I | re.U)
		perm_reg = re.search(permanent, line, re.I | re.U)
		term_type_reg = re.search(term, line, re.I | re.U)
		if(z): 
			#print (z.group(0)+"\n")
			data[i][3]=(z.group(0))
		elif(perm_reg):
			final_str = "Product Type: Permanent"
			data[i][3]=(final_str)
		elif(term_type_reg):
			final_str = "Product Type: Term"
			data[i][3]=(final_str)
		else:
			data[i][3]=" "
		
#-----------------------------------------------------------------		
		#Face Amount
		#am = re.search(amount, line, re.I | re.U)
		w = re.search(faceamount, line	, re.I | re.U)
		term_reg = re.search(termamount, line, re.I | re.U)
		seek_reg = re.search(seeking, line, re.I | re.U)
		#With faceAmount
		if(w):
			data[i][4]=(w.group(0))
#			amd = re.search(amount_with_dollar, w.group(0), re.I | re.U)
#			amwd = re.search(amount_without_dollar, w.group(0), re.I | re.U)			#Find 2nd regex in the same line of 1st regex 
#			if(amd):
#				data[i][4]=(amd.group(0))
#			elif(amwd):
#				data[i][4]=(amwd.group(0))

		#With term Amount
		elif(term_reg):
			amd = re.search(amount_with_dollar, term_reg.group(0), re.I | re.U)
			amwd = re.search(amount_without_dollar, term_reg.group(0), re.I | re.U)			#Find 2nd regex in the same line of 1st regex 
			
			if(amd):
				data[i][4]='Face Amount: '+(amd.group(0))
			elif(amwd):
				term_year_reg = re.search(term_year, amwd.group(0), re.I | re.U)
				if(term_year_reg):
					data[i][4]='Term Year: '+(amwd.group(0))
				else:
					data[i][4]='Face Amount: $'+(amwd.group(0))
		#With Seeking
		elif(seek_reg):
			amd = re.search(amount_with_dollar, seek_reg.group(0), re.I | re.U)
			amwd = re.search(amount_without_dollar, seek_reg.group(0), re.I | re.U)			#Find 2nd regex in the same line of 1st regex 
			if(amd):
				data[i][4]='Face Amount: '+(amd.group(0))
			elif(amwd):
				term_year_reg = re.search(term_year, amwd.group(0), re.I | re.U)
				if(term_year_reg):
					data[i][4]='Term Year: '+(amwd.group(0))
				else:
					data[i][4]='Face Amount: $'+(amwd.group(0))
		else:
			data[i][4]=" "
			
#-----------------------------------------------------------------		
		#Weight
		x=re.search(weight_num, line, re.I | re.U) 
		wt=re.search(weight, line, re.I | re.U)
		if(x): 
			#print (x.group(0)+"\n")
			data[i][5]=(x.group(0))
		elif(wt):
			am = re.search(weight_num,wt.group(0), re.I | re.U)
			if(am):
				data[i][5]=(am.group(0))
		else:
			data[i][5]=" "
			
		#Height
		ht = re.search(height1, line, re.I | re.U)
		htsym = re.search(height2, line, re.I | re.U)
		if(ht): 
			#print (ht.group(0)+"\n")
			data[i][6]=(ht.group(0))
		elif(htsym):
			f = re.search(feet, (htsym.group(0)), re.I | re.U)
			inch = re.search(inches, (htsym.group(0)), re.I | re.U)
			if(f):
				#print(f.group(0))
				am = re.search(height_num, (f.group(0)), re.I | re.U).group(0) + ' Feet'
				if(i):
					am+=re.search(height_num, (inch.group(0)), re.I | re.U).group(0) + ' Inches' 
				data[i][6]=am
		else:
			data[i][6]=" "
			
		#Preferred Height & Weight
		pr = ''
		pr = re.search(preferred, line, re.I | re.U)
		if(pr!='' and pr):
			h_reg = re.search(height_word, pr.group(0), re.I | re.U)
			if(h_reg):
				data[i][6] = "5 Feet 9 Inches"
			w_reg = re.search(weight_word, pr.group(0), re.I | re.U)
			if(w_reg):
				data[i][5] = "196 lbs"
			
		#Height & Weight from build
		bu = ''
		bu = re.search(build, line, re.I | re.U)
		if(bu):
			h_reg = re.search(build_height, bu.group(0), re.I | re.U)
			if(h_reg):
				data[i][6] = h_reg.group(0) + ' Feet'
				h =' '
			w_reg = re.search(build_weight, bu.group(0), re.I | re.U)
			if(w_reg):
				data[i][5] = w_reg.group(0)+ ' lbs'
				w = ' '
				
#-----------------------------------------------------------------		
		#Habit
		sm = re.search(smoker, line, re.I | re.U)
		tob = re.search(tobacco, line, re.I | re.U)
		if(sm): 
			if(re.search(no, sm.group(0), re.I | re.U)):
				data[i][7]="Non-Smoker"
			else:
				data[i][7]="Smoker"
		elif(tob):
			#print(tob.group(0))
			if(re.search(no, tob.group(0), re.I | re.U)):
				data[i][7]="Non-Tobacco"
			else:
				data[i][7]="Tobacco"
		else:
			data[i][7]=" "

#-----------------------------------------------------------------		
		#Medication & Treatment
		med_reg = (re.search(med,line, re.I | re.U))
		if(med_reg):
			if(re.search(no, med_reg.group(0), re.I | re.U)):
				data[i][8]="No Medication"
		else:															#Write else outsite condition (to stop rewriting of above cell)
			data[i][8]=""
			
	
#-----------------------------------------------------------------		
		#Family
		family_reg = (re.search(family,line, re.I | re.U))
		family_member_reg = (re.search(family_member,line, re.I | re.U))
		if(family_reg):
			data[i][9]=family_reg.groups()
		else:															#Write else outsite condition (to stop rewriting of above cell)
			data[i][9]=""
			
	
#-----------------------------------------------------------------		
		#Property
		lives_reg = (re.search(lives,line, re.I | re.U))
		prop_reg = (re.search(prop,line, re.I | re.U))
		if(lives_reg):
			data[i][10]=lives_reg.groups()
			if(prop_reg):
				data[i][10]=lives_reg.groups()+prop_reg.groups()
		elif(prop_reg):
			data[i][10]=prop_reg.groups()
		else:															#Write else outsite condition (to stop rewriting of above cell)
			data[i][10]=""
			
	
	
	wtr.writerows(data)




i=0
w, h = 12, 1;
data = [[" " for x in range(w)] for y in range(h)]
st = []
out = open('out.csv', 'w')
wtr= csv.writer( out )
wtr.writerow(['Gender','Year_of_birth','Age(years)','Product Type','Face Amount','Weight','Height','Habit','Medication','Family','Property'])
#strs = ["" for x in range(size)]

with open('under.csv') as f:
	rows = csv.reader(f)
	for row in rows:
		#print('\n----------------------------------------------')
		st.append(row[8])
		reg(st,i)
		#i+=1
		st=[]
		#print (row[8])  

out.close()


