
import utils
def main():
    dir = 'data_speech_commands_v0.02'
    train_dirname = 'train_audio'
    validation_dirname = 'valid_audio'

    train_dirpath = os.path.join(dir, train_dirname)
    trainDir = train_dirpath + '/'

    validation_dirpath = os.path.join(dir, validation_dirname)
    validDir = validation_dirpath + '/'

    test_dirpath = os.path.join(dir, test_dirname)
    testDir = test_dirpath + '/'

    trainDataSet = utils.buildDataSet(trainDir)
    validDataSet = utils.buildDataSet(validDir)
    testDataSet = utils.buildDataSet(testDir)

    hmmModels = utils.train_GMMHMM(trainDataSet)
    with open("hmmgmm_model.pkl", "wb") as file: pickle.dump(hmmModels, file)

    utils.predict_GMMHMM(trainDataSet,hmmModels,type='train')
    del trainDataSet
    utils.predict_GMMHMM(validDataSet,hmmModels,type='validation')
    del validDataSet

if __name__ == '__main__':
    main()
