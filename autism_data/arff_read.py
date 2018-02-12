import io
import os
import zipfile
from zipfile import ZipFile
from urllib.request import urlretrieve
from scipy.io import arff
import pandas as pd

URL_child = "https://archive.ics.uci.edu/ml/machine-learning-databases/00419/Autism-Screening-Child-Data%20Plus%20Description.zip"
URL_adult = "https://archive.ics.uci.edu/ml/machine-learning-databases/00426/Autism-Adult-Data%20Plus%20Description%20File.zip"
URL_adolescent = "https://archive.ics.uci.edu/ml/machine-learning-databases/00420/Autism-Adolescent-Data%20Plus%20Description.zip"

ARFF_child = 'Autism-Child-Data.arff'
ARFF_child = 'Autism-Adult-Data.arff'
ARFF_adolescent = 'Autism-Adolescent-Data.arff'

def arff_to_df(URL=URL_child,arff_file='Autism-Child-Data.arff',force_download=False):
	if force_download or not os.path.exists(arff_file):
		zipped = urlretrieve(URL,'autism.zip')
		zipfile = ZipFile(zipped[0],'r')
		arff_file = zipfile.extract(arff_file)

	##extracting the data dictionary and column names (description)

	data, description = arff.loadarff(arff_file)
	
	columns = [i for i in description]

	new_columns = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
       'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age', 'gender',
       'ethnicity', 'jaundice', 'autism', 'country_of_res', 'used_app_before',
       'result','age_range', 'relation', 'Class/ASD']

	df = pd.DataFrame(data,columns=columns)
	
	df.columns = new_columns

	#changing utf-8 coding to categorical variables		
	for column in list(df.select_dtypes(include=['object']).columns):
		df[column] = df[column].str.decode('utf-8').astype('category')

	#Fixing ethnicity discrepancies
	df.ethnicity = df.ethnicity.str.replace('?',"Unknown").str.replace(' ','_').str.replace('\'Middle_Eastern_\'',"Middle_Eastern" ).str.replace('\'South_Asian\'','South_Asian').str.replace('others','Others').astype('category')

	#Fill in missing age with median age
	df.age = df.age.fillna(value=df.age.median())

	#Fixing relation discrepancies

	relation_mapper = {'Parent':'Family Member',
                   'Relative':'Family Member',
                   '\'Health care professional\'':'Health care professional',
                   '?':'Unknown',
                   'Self':'Self',
                   'self':'Self'}

	df.relation = df.relation.map(relation_mapper).astype('category')
	df.relation = df.relation.fillna("Unknown")

	#Fixing various binary inputs
	jaun_mapper = {'yes':1,'no':0}
	aut_mapper = {'yes':1,'no':0}
	class_mapper = {'YES':1,'NO':0}
	app_mapper = {'yes':1,'no':0}

	mapper_list = list([jaun_mapper,aut_mapper,class_mapper,app_mapper])

	for x, y in zip(['jaundice', 'autism','Class/ASD','used_app_before'],mapper_list):
    		df[x] = df[x].map(y)

	df = df.drop('country_of_res',axis=1)
	
	return df

def df_get_uniform_dummies(df,astype='int64'):

	assert isinstance(df,pd.DataFrame),"Only works with Pandas DataFrame"	

	new_df = pd.get_dummies(df,drop_first=True,columns=['gender','ethnicity','age_range','relation'],prefix={'gender':'gen','ethnicity':'eth','age_range':'age','relation':'rel'})

	new_df = new_df.select_dtypes(include=['category','int64','float64','uint8']).apply(pd.to_numeric).astype(astype)
	return new_df
