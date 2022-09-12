## find abstract in full text, keep abstracts that are within the full text list
import pandas as pd
import sys, os
import numpy as np
import re

df1 = pd.read_csv(sys.argv[1], sep='\t') #abstract (3000 entry)
df2 = pd.read_csv(sys.argv[2], names=['full_text', 'label'], sep=',') #full text Document

print(df1.head())
print(df2.head())
#df3= df1.loc[df1.abstract.isin(df2['full_text'])]
#print(df3)
def isNaN(string):
	return string != string

logiclist=[]
for label, content in df1['title'].items():
	print('label:', label)
	print(content)
	if isNaN(content)==True:
		logiclist.append("no")
	else:
		if content.startswith(". "):
			content = content.split('. ')[1]
		elif content.startswith("0. "):
			content = content.split('. ')[1]
		elif content.startswith("1. "):
			content = content.split('. ')[1]
		elif content.startswith("2. "):
			content = content.split('. ')[1]
		elif content.startswith("3. "):
			content = content.split('. ')[1]
		elif content.startswith("4. "):
			content = content.split('. ')[1]
		elif content.startswith("5. "):
			content = content.split('. ')[1]
		elif content.startswith("6. "):
			content = content.split('. ')[1]
		elif content.startswith("7. "):
			content = content.split('. ')[1]
		elif content.startswith("8. "):
			content = content.split('. ')[1]
		elif content.startswith("9. "):
			content = content.split('. ')[1]
		else:
			pass
		content= re.sub('[^A-Za-z0-9]+', '', content)
		content= content.encode("utf-8")
		print('content:', content, sep='\n')
	
# 	for label2, content2 in df2['full_text'].items():
# 		if content in content2:
# 			print("yes")
# 		else:
# 			print("no")

#		df1["logic"] = np.where(df2["full_text"].str.contains(content), "yes", "no")
		newlist=[]
		for label2, content2 in df2['full_text'].items():
			#print(label2, content2)
			content2 =re.sub('[^A-Za-z0-9]+', '', content2)
			content2= content2.encode("utf-8")
			result = content2.lower().rfind(content.lower())
			#if content.lower() in content2.lower():
			if result != -1:
				newlist.append("yes")
			else:
				newlist.append("no")
		print(newlist)
		if "yes" in newlist:
			print("title has full text!")
			#df1["logic"]= "yes"
			#df1["logic"] = np.where(df["title"]==content, "yes", "no")
			logiclist.append("yes")
		else:
			#df1["logic"]= "no"
			logiclist.append("no")
		
# 		if content in df2['full_text'].any():
# 			print("yes")
# 			df1["logic"]= "yes"
# 		else:
# 			print("no")
# 			df1["logic"]= "no"
		
print(logiclist, len(logiclist))
df1['logic'] = logiclist
df1name = str(sys.argv[1])
df1.to_csv(path_or_buf=str(df1name)+"_logic.txt", sep="\t", header=True)

#df1["logic"] = np.where(df2["full_text"].str.contains(df1['abstract']), "yes", "no")
#print(df1.head())
#if df2[df2['full_text'].str.contains(df1.loc['abstract'])]:
#	print()