ADDING THIS LINE TO TEST
# STS-Random-forest
## Semantic textual Similarity tool which generates random forest model
## How to use this STS tool:
--------------------------
1. Install Anaconda 2.7 
2. install brown corpus,wordnet, punkt modules of nltk as below
	1. nltk.download('punkt')
	2. nltk.download('brown')
	3. nltk.download('wordnet')
3. browse your training dataset file
	1. training dataset should contain tab seperated sentences with its actual value at the end.
4. Create directory with same name as training dataset file and place a file named 'actual.txt' in it.
5. press all the buttons on by one to create that features.
6. press combineall for creating csv file combining all features.
7. now generate model.
	Note: you should create a directory 'models' where the training dataset exists.
8. input two sentences and press find desgree of equivalence.
9. the screen will display the degree ranging from 0 to 5.
