"""

@author: Jinal Shah

This script will serve as unit tests for the Preprocessing

"""
import unittest
import pandas as pd
import preprocessing

class TestPreprocessing(unittest.TestCase):

    # Testing the drop function
    def test_drop(self):
        dummy_data = {'id':[0,1],'location':['None','USA'],'keyword':['None','hurricane'],'text':['some random text','some random text']}
        dummy_data = pd.DataFrame(dummy_data)
        expectation = pd.DataFrame({'text':['some random text','some random text']})
        preprocess = preprocessing.Preprocessing()
        returned = preprocess.drop_columns(dummy_data)


        self.assertTrue(expectation.equals(returned),'Expectation Does Not Equal Returned!')
    
    # Testing the preprocessing
    def test_preprocessing(self):
        # Test 1: We have only words
        dummy_data = {'id':[0,1],'location':['None','USA'],'keyword':['None','hurricane'],'text':['some random text','1 random text']}
        dummy_data = pd.DataFrame(dummy_data)
        preprocess = preprocessing.Preprocessing()
        returned = preprocess.preprocess_data(dummy_data)

        expectation = [['random','text'],['1','random','text']]
        self.assertEqual(returned,expectation)

        # Test 2: We have a link
        dummy_data = {'id':[0,1],'location':['None','USA'],'keyword':['None','hurricane'],'text':['Hi there: https:somerandom/text','some random text']}
        dummy_data = pd.DataFrame(dummy_data)
        preprocess = preprocessing.Preprocessing()
        returned = preprocess.preprocess_data(dummy_data)

        expectation = [['hi'],['random','text']]
        self.assertEqual(returned,expectation)

        # Test 3: We have characters before the words
        dummy_data = {'id':[0,1],'location':['None','USA'],'keyword':['None','hurricane'],'text':['@some #random text','some *random text!!!!']}
        dummy_data = pd.DataFrame(dummy_data)
        preprocess = preprocessing.Preprocessing()
        returned = preprocess.preprocess_data(dummy_data)

        expectation = [['random','text'],['random','text']]
        self.assertEqual(returned,expectation)

        # Test 4: Some words aren't defined in Unicode
        dummy_data = {'id':[0,1],'location':['None','USA'],'keyword':['None','hurricane'],'text':['Barbados #Bridgetown JAMAICA ÛÒ Two cars set ablaze:','some *random text']}
        dummy_data = pd.DataFrame(dummy_data)
        preprocess = preprocessing.Preprocessing()
        returned = preprocess.preprocess_data(dummy_data)

        expectation = [['barbados','bridgetown','jamaica','two','cars','set','ablaze'],['random','text']]
        self.assertEqual(returned,expectation)

        # Test 5: A simple example in the data
        dummy_data = {'id':[0,1],'location':['None','USA'],'keyword':['None','hurricane'],'text':['London is cool ;)','some *random text']}
        dummy_data = pd.DataFrame(dummy_data)
        preprocess = preprocessing.Preprocessing()
        returned = preprocess.preprocess_data(dummy_data)

        expectation = [['london','cool'],['random','text']]
        self.assertEqual(returned,expectation)


# Main Method to run the tests
if __name__ == '__main__':
    unittest.main()