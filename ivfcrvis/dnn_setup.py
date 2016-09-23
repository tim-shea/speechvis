'''Example script showing how to use stateful RNNs
to model long sequences efficiently.
'''
import numpy
from matplotlib import pyplot
from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from keras.optimizers import RMSprop
from scipy.io import wavfile
from features import logfbank
import theano


recording = Recording('/media/tim/LittleDrive/ivfcr/', 'e20151210_145625_010585')
recording.read_recording()
fbanks = recording.frequency_banks()
dfbanks = numpy.diff(fbanks, axis=0)

# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = 50
epochs = 10
# number of elements ahead that are used to make the prediction
lahead = 1

print('Generating Data')
xLength = batch_size * int(0.75 * len(fbanks) / batch_size)
dx = numpy.concatenate((numpy.zeros((1, 1, 26)), dfbanks[:xLength - 1, :, :]))
x = numpy.concatenate((fbanks[:xLength, :, :], dx), axis=2)
print('Input shape:', x.shape)

expected_output = numpy.zeros((len(x), 52))
for i in range(len(x) - lahead):
    #expected_output[i, j] = numpy.mean(x[i + 1:i + lahead + 1, 0, j])
    expected_output[i, :] = x[i + lahead, 0, :]
print('Output shape: ', expected_output.shape)

print('Creating Model')
model = Sequential()
#model.add(LSTM(200,
#model.add(SimpleRNN(100,
#               batch_input_shape=(batch_size, tsteps, 26),
#               return_sequences=True,
#               stateful=True))
#model.add(LSTM(100,
rnn = SimpleRNN(100, batch_input_shape=(batch_size, tsteps, 52),
                return_sequences=False, stateful=True, dropout_W=0.05)
model.add(rnn)
model.add(Dense(52))
rmsprop = RMSprop(lr=0.0005)
model.compile(loss='mse', optimizer=rmsprop)

print('Training')
for i in range(epochs):
    print('Epoch', i, '/', epochs)
    model.fit(x,
              expected_output,
              batch_size=batch_size,
              verbose=1,
              nb_epoch=1,
              shuffle=False)
    model.reset_states()

print('Predicting')
testOffset = len(x)
testLength = batch_size * int(0.25 * len(fbanks) / batch_size)
dtest_input = numpy.concatenate((numpy.zeros((1, 1, 26)), dfbanks[testOffset:testOffset + testLength - 1, :, :]))
test_input = numpy.concatenate((fbanks[testOffset:testOffset + testLength, :, :], dtest_input), axis=2)
encoder = Sequential()
encoder.add(rnn)
test_hidden = encoder.predict(test_input, batch_size=batch_size)
test_output = model.predict(test_input, batch_size=batch_size)

print('Plotting Results')
pyplot.subplot(411)
pyplot.imshow(test_input.reshape(len(test_input), 52).transpose()[:, :500],
              aspect='auto', origin='lower', interpolation='nearest')
pyplot.subplot(412)
pyplot.imshow(test_hidden.transpose()[:, :500],
              aspect='auto', origin='lower', interpolation='nearest')
pyplot.subplot(413)
pyplot.imshow(test_output.transpose()[:, :500],
              aspect='auto', origin='lower', interpolation='nearest')
pyplot.subplot(414)
error = numpy.sqrt(numpy.sum(numpy.square(test_input.reshape((testLength, 52)) - test_output), axis=1))
pyplot.plot(error[:500])
pyplot.show()

index, starts, ends = recording.filter_speaker('CHN')
rnns = numpy.zeros((len(index) * 24, 100))
a = 0
for i, s in zip(range(len(index)), starts):
    r = int(s * 40) - testOffset
    if r < 0:
        a = i
    elif r + 24 < len(test_hidden):
        rnns[i * 24:i * 24 + 24, :] = test_hidden[r:r + 24, :]
    elif r + 24 >= len(test_hidden):
        rnns = rnns[(a + 1) * 24:i * 24, :]
        index = index[a + 1:i]
        starts = starts[a + 1:i]
        ends = ends[a + 1:i]
        break

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
v = pca.fit_transform(rnns)

ica = FastICA(n_components=3)
v = ica.fit_transform(rnns)

tsne = TSNE(n_components=2)
v = tsne.fit_transform(rnns)

x = numpy.reshape(v[:,0], (len(v) / 24, 24)).transpose()
y = numpy.reshape(v[:,1], (len(v) / 24, 24)).transpose()
z = numpy.reshape(v[:,2], (len(v) / 24, 24)).transpose()

def plotVocalSpace():
    pyplot.subplot(221)
    pyplot.scatter(x, y, c='k', marker='.', edgecolors='face', alpha=0.1)
    pyplot.ylabel('{0:.2f}% Variance'.format(100 * pca.explained_variance_ratio_[1]))
    pyplot.subplot(222)
    pyplot.scatter(z, y, c='k', marker='.', edgecolors='face', alpha=0.1)
    pyplot.xlabel('{0:.2f}% Variance'.format(100 * pca.explained_variance_ratio_[2]))
    pyplot.subplot(223)
    pyplot.scatter(x, z, c='k', marker='.', edgecolors='face', alpha=0.1)
    pyplot.xlabel('{0:.2f}% Variance'.format(100 * pca.explained_variance_ratio_[0]))


def plotVocalTrajectory(i):
    pyplot.subplot(221)
    pyplot.plot(x[:, i], y[:, i], lw=1.5, label='Voc {}'.format(index[i]))
    pyplot.subplot(222)
    pyplot.plot(z[:, i], y[:, i], lw=1.5, label='Voc {}'.format(index[i]))
    pyplot.subplot(223)
    pyplot.plot(x[:, i], z[:, i], lw=1.5, label='Voc {}'.format(index[i]))
