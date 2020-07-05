import preprocess
import classifier

def main():
    preprocessing   = preprocess.PreprocessTitanic()
    classifing      = classifier.ClassifyTitanic()

    #Preprocess mode
    preprocessing.exploreData()
    preprocessing.descibeData()
    preprocessing.analizeData()


if __name__ == "__main__":
    main()