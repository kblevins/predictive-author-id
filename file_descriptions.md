# File descriptions

#### 6_fasttext_predictions

Contains files related to a stretch activity trying a different machine learning library for author id



####Data_Output

* equal_length_test.csv
  * random subset of test data with equal length subset for each author
* equal_length_train.csv		
  * random subset of train data with equal length subset for each author
* probabilities.csv			
  * the probability for each sentence that each author wrote it
* test_anovas.csv				
  * anova results for test dataset
* test_with_author.csv		
  * the test dataset with an assigned author, based on the author with 								the highest probability in the probabilities table
* train_anovas.csv			
  * anova results for train dataset	



#### Data_Raw

* test.csv			
  * a subset of the test dataset to use in testing code
* test_full.csv			
  * the full test dataset
* train.csv			
  * a subset of the train dataset to use in testing code
* train_full.csv			
  * the full train dataset - to be used in developing algorithm to apply to test dataset in order to identify authors



#### Plots

* test_plots			
  * plots related to data analysis of the test dataset
* train_plots			
  * plots related to data analysis of the train dataset



#### QC_Freq_Method

Contains files related to testing frequency method with subsets of training data



#### Sandbox					

A testing ground for peices of code that were used to build the final data exploration & analysis files



#### Main folder

* 1_data_cleanup_train.ipynb		
  * creates random subset of train data with equal length subset for each author
* 2_data_exploration_train.ipynb		
  * analyses of train dataset, including data exploration, clean up, plots related to sentence structure, vocabulary, and punctuation, as well as ANOVA analyses
* 3_data_analysis.ipynb			
  * final analysis to assign authors to the test dataset
* 4_data_cleanup_test.ipynb		
  * creates random subset of test data with equal length subsets for each author 
* 5_data_exploration_test.ipynb		
  * analyses of test dataset with assigned authors to compare to the same analyses performed on the train dataset


* file_descriptions.md			
  * description of the contents of the repo
* ML_Trail.ipynb
  * kaggle classifier incorporating n-grams, part of speech, and word frequency
* Predictive_Author_Presentation.pptx	
  * final presentation for project
* README.md				
  * description of project and repo contents
* requirements.txt			
  * list of required packages to run code
* Summary.docx				
  * Summary of project & results



