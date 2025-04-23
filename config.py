from os import path, getenv

version = '1.9.0'

#patch_shape = (64, 192, 192)
patch_shape = (5, 192, 192)

working_resolution = (2, 0.6, 0.6)

pixelMaxValue = 3096
pixelMinValue = -1000
# patch_shape = (128, 128, 128)
# working_resolution = (1.0, 1.0, 1.0)

# Data directory (input / output)
basedir = getenv('VERSEG_BASEDIR', '/data')
datadir = path.join(basedir, 'datasets')
expdir = path.join(basedir, 'experiments')

# External code base (undefined = feature disabled)
extdir = getenv('VERSEG_EXTDIR', path.join(basedir, 'code', 'devel'))
overviewPath = 'overview.csv'

# Print out version number when executed directly
if __name__ == '__main__':
    print(version)
