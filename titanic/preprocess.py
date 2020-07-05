import os
import pandas as pd
from util import bcolors as bc

class PreprocessTitanic():
    """
    """
    def __init__(self):
        self.datasetPath = os.path.join(os.getcwd(), "titanic", "titanic")
        self.dirs = os.listdir(self.datasetPath)
        self.groundtruthfilepath = os.path.join(self.datasetPath, self.dirs[2])
        self.trainfilepath = os.path.join(self.datasetPath, self.dirs[1])
        self.testfilepath = os.path.join(self.datasetPath, self.dirs[0])
        print(bc.OKGREEN + bc.INFO + "Directories {} Files {}" .format(self.datasetPath, self.dirs) + bc.ENDC)

        # Load Data train/test
        self.loadtrainData()
        self.loadtestData()
        self.loadgroundtruthData()

    def loadtrainData(self):
        self.train_df = pd.read_csv(self.trainfilepath)
        print(bc.OKGREEN + bc.INFO + "loadtrainData" + bc.ENDC)
        print("Train data head:", self.train_df.head())

    def loadtestData(self):
        self.test_df = pd.read_csv(self.testfilepath)
        print(bc.OKGREEN + bc.INFO + "loadtestData" + bc.ENDC)
        print("Test data head:", self.test_df.head())

    def loadgroundtruthData(self):
        self.val_df = pd.read_csv(self.groundtruthfilepath)
        print(bc.OKGREEN + bc.INFO + "loadgroundtruthData" + bc.ENDC)
        print("Test data head:", self.val_df.head())

    def exploreData(self):
        print(bc.OKGREEN + bc.INFO + "exploreData Train" + bc.ENDC)
        print(self.train_df.info())
        print(bc.OKGREEN + bc.INFO + "exploreData Test" + bc.ENDC)
        print(self.test_df.info())

    def descibeData(self):
        """
        What is the distribution of categorical features?:
            Names are unique across the dataset (count=unique=891)
            Sex variable as two possible values with 65% male (top=male, freq=577/count=891).
            Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin.
            Embarked takes three possible values. S port used by most passengers (top=S)
            Ticket feature has high ratio (22%) of duplicate values (unique=681).
        """
        print(bc.OKGREEN + bc.INFO + "descibeData" +bc.ENDC)
        print(self.train_df.describe())
        print(bc.OKGREEN + bc.INFO + "descibeData" +bc.ENDC)
        print(self.train_df.describe(include=['O']))


    """
    Assumtions based on data analysis
        We arrive at following assumptions based on data analysis done so far. We may validate these assumptions further before taking appropriate actions.

    Correlating.
        We want to know how well does each feature correlate with Survival. We want to do this early in our project and match these quick correlations with modelled correlations later in the project.

    Completing.
        We may want to complete Age feature as it is definitely correlated to survival.
        We may want to complete the Embarked feature as it may also correlate with survival or another important feature.

    Correcting.
        Ticket feature may be dropped from our analysis as it contains high ratio of duplicates (22%) and there may not be a correlation between Ticket and survival.
        Cabin feature may be dropped as it is highly incomplete or contains many null values both in training and test dataset.
        PassengerId may be dropped from training dataset as it does not contribute to survival.
        Name feature is relatively non-standard, may not contribute directly to survival, so maybe dropped.

    Creating.
        We may want to create a new feature called Family based on Parch and SibSp to get total count of family members on board.
        We may want to engineer the Name feature to extract Title as a new feature.
        We may want to create new feature for Age bands. This turns a continous numerical feature into an ordinal categorical feature.
        We may also want to create a Fare range feature if it helps our analysis.

    Classifying.
        We may also add to our assumptions based on the problem description noted earlier:
            Women (Sex=female) were more likely to have survived.
            Children (Age<?) were more likely to have survived.
            The upper-class passengers (Pclass=1) were more likely to have survived.
    """

    def analizeData(self):
        print(bc.OKGREEN + bc.INFO + "analizeData Pclass" + bc.ENDC)
        print(self.train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
        print(bc.OKGREEN + bc.INFO + "analizeData Sex" + bc.ENDC)
        print(self.train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
        print(bc.OKGREEN + bc.INFO + "analizeData SibSp" + bc.ENDC)
        print(self.train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
        print(bc.OKGREEN + bc.INFO + "analizeData Parch" + bc.ENDC)
        print(self.train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

