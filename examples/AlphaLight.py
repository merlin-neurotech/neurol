from neurol import streams
from neurol.connect_device import get_lsl_EEG_inlets
from neurol.BCI import generic_BCI, automl_BCI
from neurol import BCI_tools
from neurol.models import classification_tools

# TODO: try blink to turn on/off

from phue import Bridge #HUE



b = Bridge('192.168.2.50')  #192.168.2.116 (home) - 192.168.0.154 (nates top)
b.connect()
lights = b.get_light_objects()
k = 5
k=k/10
for light in lights:
    light.xy = [0.1000, k]
print("Light default set")


k = 5


def changeLight(input_):
    global k
    if (input_ == 'Non-concentrated'):
        print(input_)
        if (k >= 9):
            k = 9.0
        else:
            k = k + .4
    else:
        print(input_)
        if (k <= 1):
            k = 1.0
        else:
            k = k - .4

    k = k/10  # put it into proper format so it works w light funct
    for light in lights:
        light.xy = [0.1000, k]
    k = k * 10

#-------------------------------------------
# region define BCI behaviour

# we defined a calibrator which returns the 65th percentile of alpha wave
# power over the 'AF7' and 'AF8' channels of a muse headset after recording for 10 seconds
# and using epochs of 1 second seperated by 0.25 seconds.
clb = lambda stream:  BCI_tools.band_power_calibrator(stream, ['AF7', 'AF8'], 'muse', bands=['alpha_low', 'alpha_high'],
                                                        percentile=9, recording_length=10, epoch_len=1, inter_window_interval=0.25)


# define a transformer that corresponds to the choices we made with the calibrator
gen_tfrm = lambda buffer, clb_info: BCI_tools.band_power_transformer(buffer, clb_info, ['AF7', 'AF8'], 'muse',
                                                                    bands=['alpha_low', 'alpha_high'], epoch_len=1)

# Again, we define a classifier that matches the choices we made
# we use a function definition instead of a lambda expression since we want to do slightly more with


def clf(clf_input, clb_info):

    # use threshold_clf to get a binary classification
    binary_label = classification_tools.threshold_clf(
        clf_input, clb_info, clf_consolidator='all')

    # decode the binary_label into something more inteligible for printing
    label = classification_tools.decode_prediction(
        binary_label, {True: 'Concentrated', False: 'Non-concentrated'})

    return label
# endregion
#-------------------------------------------

# GET EEG STREAM
# gets first inlet, assuming only one EEG streaming device is connected
inlet = get_lsl_EEG_inlets()[0]

# we ask the stream object to manage a buffer of 1024 samples from the inlet
stream = streams.lsl_stream(inlet, buffer_length=1024)


# DEFINE GENERIC BCI
alpha_light_generic = generic_BCI(
    clf, transformer=gen_tfrm, action=changeLight, calibrator=clb)


# DEFINE AUTO_ML BCI parameters
ml_epoch_len = 50
ml_recording_length = 10
ml_n_states = 2
ml_transformer = lambda x: x[:,1] # takes the first two channels

from sklearn import svm
ml_model = svm.SVC()
alpha_light_automl = automl_BCI(ml_model, ml_epoch_len, ml_n_states, ml_transformer, action=print)



usrChoice = int(input("Choose your BCI.\n1) Generic BCI\n2) AutoML BCI\n"))

# RUN BCI
try:
    # inlet -> calibrate -> transform -> classify -> action
    if usrChoice == 1:
    # CALIBRATE GENERIC BCI
        alpha_light_generic.calibrate(stream)
        alpha_light_generic.run(stream)
    elif usrChoice == 2:
        alpha_light_automl.build_model(stream, ml_recording_length)
        alpha_light_automl.run(stream)
except KeyboardInterrupt:
    print('\n\nBCI Ended')

""" Todo:
- Calculated threshold value per individual - check out BCI workshop
- incorporate the blink classifier - Where can we find the blink classifier?
- pip installable the extra tools (from AI team)
- clean and document (requirements.txt)"""


# add pynput to requirements# insa
