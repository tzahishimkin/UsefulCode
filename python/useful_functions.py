




def get_all_tfrecords(x):
    '''
    :param x: The list of directories which contain tfrecords
    :return: Returns all the file names with a specific signature (*.tfrecords) in the list of directories
    '''
    flatten = lambda x: list(itertools.chain.from_iterable(x)) #Loop over a list
    train_filenames = flatten([glob.glob(os.path.join(f, "*.tfrecords")) if os.path.isdir(f) else [f] for f in args.train])
    #Use LMDB - Explain how to install and use in python:
    # TF save and load checkpoint:
