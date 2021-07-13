The functionality present within the current solution is based off of the code provided to us in the lab tutorials, 
though with modification and cleaning as described in my report. Included within this source folder 
is the organised training and testing data used. IT IS NOT my own work, only the reorganisation. These
datasets are properly referenced in my report, please refer to that for the source. 

The program and all testing/training data is stored in \HumanMotion\

In order to change components:

SIFT, SURF, FLANN and Brute Force
Uncomment the component you wish to use and Comment out the redundent component

SVM and Normal Bayes
Above main() is a variable called classifier. 
Setting this to 0 makes the system use SVM and setting it to 1 uses Normal Bayes

To change dataset:
The path to the training and testing text files that list all of the files will need to be changed. 

"train.txt" and "test.txt" -> the dataset given in Lab Tutorial's 16 and 18

"train_1.txt" and "test_1.txt" -> the HMDB database

"train_2.txt" and "test_2.txt" -> the database created by C. Schuldt, I. Laptev and B. Caputo


 - UP776193