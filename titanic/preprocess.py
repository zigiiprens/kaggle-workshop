import os
from util import bcolors as bc

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


class PreprocessTitanic():
    """
    """
    def __init__(self):
        print(bc.HEADER + "PreprocessTitanic starting ..." + bc.ENDC)
        self.datasetPath = os.path.join(os.getcwd(), "titanic", "titanic")
        self.dirs = os.listdir(self.datasetPath)
        self.groundtruthfilepath = os.path.join(self.datasetPath, self.dirs[2])
        self.trainfilepath = os.path.join(self.datasetPath, self.dirs[1])
        self.testfilepath = os.path.join(self.datasetPath, self.dirs[0])
        print(bc.OKGREEN + bc.INFO + "Directories {} Files {}" .format(self.datasetPath, self.dirs) + bc.ENDC)

        self.train_df = None
        self.test_df = None

        # Load Data train/test
        self.loadtrainData()
        self.loadtestData()
        self.loadgroundtruthData()
        self.featuresData()
        self.combine = [self.train_df, self.test_df]

    def loadtrainData(self):
        self.train_df = pd.read_csv(self.trainfilepath)
        print(bc.OKGREEN + bc.INFO + "loadtrainData" + bc.ENDC)
        print(self.train_df.head())

    def loadtestData(self):
        self.test_df = pd.read_csv(self.testfilepath)
        print(bc.OKGREEN + bc.INFO + "loadtestData" + bc.ENDC)
        print(self.test_df.head())

    def loadgroundtruthData(self):
        self.val_df = pd.read_csv(self.groundtruthfilepath)
        print(bc.OKGREEN + bc.INFO + "loadgroundtruthData" + bc.ENDC)
        print(self.val_df.head())

    def featuresData(self):
        print(bc.OKGREEN + bc.INFO + "featuresData Train" + bc.ENDC)
        print(self.train_df.columns.values)
        print(bc.OKGREEN + bc.INFO + "featuresData Test" + bc.ENDC)
        print(self.test_df.columns.values)
        print(bc.OKGREEN + bc.INFO + "featuresData Val" + bc.ENDC)
        print(self.val_df.columns.values)

    def exploreData(self):
        print(bc.OKGREEN + bc.INFO + "exploreData Train" + bc.ENDC)
        print(self.train_df.info())
        print(bc.OKGREEN + bc.INFO + "exploreData Test" + bc.ENDC)
        print(self.test_df.info())

    def describeData(self):
        """
        What is the distribution of categorical features?:
            Names are unique across the dataset (count=unique=891)
            Sex variable as two possible values with 65% male (top=male, freq=577/count=891).
            Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin.
            Embarked takes three possible values. S port used by most passengers (top=S)
            Ticket feature has high ratio (22%) of duplicate values (unique=681).
        """
        print(bc.OKGREEN + bc.INFO + "describeData" +bc.ENDC)
        print(self.train_df.describe())
        print(bc.OKGREEN + bc.INFO + "describeData" +bc.ENDC)
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

    def analizeDataByVisual(self):
        print(bc.OKGREEN + bc.INFO + "analizeDataByVisual Age/Survived" + bc.ENDC)
        """
        Analyze by visualizing data
            Now we can continue confirming some of our assumptions using visualizations for analyzing the data.

        Correlating numerical features
            Let us start by understanding correlations between numerical features and our solution goal (Survived).
            A histogram chart is useful for analyzing continous numerical variables like Age where banding or ranges will help identify useful patterns. The histogram can indicate distribution of samples using automatically defined bins or equally ranged bands. This helps us answer questions relating to specific bands (Did infants have better survival rate?)
            Note that x-axis in historgram visualizations represents the count of samples or passengers.

        Observations.
            Infants (Age <=4) had high survival rate.
            Oldest passengers (Age = 80) survived.
            Large number of 15-25 year olds did not survive.
            Most passengers are in 15-35 age range.

        Decisions.
            This simple analysis confirms our assumptions as decisions for subsequent workflow stages.

            We should consider Age (our assumption classifying #2) in our model training.
            Complete the Age feature for null values (completing #1).
            We should band age groups (creating #3).
        """
        g = sns.FacetGrid(self.train_df, col='Survived')
        g.map(plt.hist, 'Age', bins=20).fig.show()

        """
        Correlating numerical and ordinal features
            We can combine multiple features for identifying correlations using a single plot. This can be done with numerical and categorical features which have numeric values.

        Observations.
            Pclass=3 had most passengers, however most did not survive. Confirms our classifying assumption #2.
            Infant passengers in Pclass=2 and Pclass=3 mostly survived. Further qualifies our classifying assumption #2.
            Most passengers in Pclass=1 survived. Confirms our classifying assumption #3.
            Pclass varies in terms of Age distribution of passengers.
        
        Decisions.
            Consider Pclass for model training.
        """
        grid = sns.FacetGrid(self.train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
        grid.map(plt.hist, 'Age', alpha=.5, bins=20).fig.show()
        grid.add_legend()

        """
        Correlating categorical features
            Now we can correlate categorical features with our solution goal.

        Observations.
            Female passengers had much better survival rate than males. Confirms classifying (#1).
            Exception in Embarked=C where males had higher survival rate. This could be a correlation between Pclass and Embarked and in turn Pclass and Survived, not necessarily direct correlation between Embarked and Survived.
            Males had better survival rate in Pclass=3 when compared with Pclass=2 for C and Q ports. Completing (#2).
            Ports of embarkation have varying survival rates for Pclass=3 and among male passengers. Correlating (#1).
        
        Decisions.
            Add Sex feature to model training.
            Complete and add Embarked feature to model training.
        """
        grid = sns.FacetGrid(self.train_df, row='Embarked', size=2.2, aspect=1.6)
        grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep').fig.show()
        grid.add_legend()

        """
        Correlating categorical and numerical features
            We may also want to correlate categorical features (with non-numeric values) and numeric features. We can consider correlating Embarked (Categorical non-numeric), Sex (Categorical non-numeric), Fare (Numeric continuous), with Survived (Categorical numeric).

        Observations.
            Higher fare paying passengers had better survival. Confirms our assumption for creating (#4) fare ranges.
            Port of embarkation correlates with survival rates. Confirms correlating (#1) and completing (#2).
        
        Decisions.
            Consider banding Fare feature.
        """
        grid = sns.FacetGrid(self.train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
        grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None).fig.show()
        grid.add_legend()

        input(bc.WARNING + "Press any key to close..." + bc.ENDC)

    def correctionData(self):
        print(bc.OKGREEN + bc.INFO + "correctionData Ticket/Cabin droping" + bc.ENDC)
        """
        Correcting by dropping features¶
            This is a good starting goal to execute. By dropping features we are dealing with fewer data points. Speeds up our notebook and eases the analysis.
            Based on our assumptions and decisions we want to drop the Cabin (correcting #2) and Ticket (correcting #1) features.
            Note that where applicable we perform operations on both training and testing datasets together to stay consistent.
        """
        print(bc.OKBLUE + "Before" +str(self.train_df.shape) + str(self.test_df.shape) + str(self.combine[0].shape) + str(self.combine[1].shape) + bc.ENDC)

        self.train_df = self.train_df.drop(['Ticket', 'Cabin'], axis=1)
        self.test_df = self.test_df.drop(['Ticket', 'Cabin'], axis=1)
        self.combine = [self.train_df, self.test_df]

        print(bc.OKBLUE + "After" +str(self.train_df.shape) + str(self.test_df.shape) + str(self.combine[0].shape) + str(self.combine[1].shape)+ bc.ENDC)

    def createData(self):
        """
        Creating new feature extracting from existing
            We want to analyze if Name feature can be engineered to extract titles and test correlation between titles and survival, before dropping Name and PassengerId features.

            In the following code we extract Title feature using regular expressions. The RegEx pattern (\w+\.) matches the first word which ends with a dot character within Name feature. The expand=False flag returns a DataFrame.

        Observations.
            When we plot Title, Age, and Survived, we note the following observations.
            Most titles band Age groups accurately. For example: Master title has Age mean of 5 years.
            Survival among Title Age bands varies slightly.
            Certain titles mostly survived (Mme, Lady, Sir) or did not (Don, Rev, Jonkheer).
        
        Decision.
            We decide to retain the new Title feature for model training.
        """
        print(bc.OKGREEN + bc.INFO + "createData All" + bc.ENDC)
        # TODO: include the data creation here
