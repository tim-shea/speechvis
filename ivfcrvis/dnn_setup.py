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


# Load the recording audio and convert it to Mel-frequency spectrum
recording = Recording('/media/tim/LittleDrive/ivfcr/', 'e20151210_145625_010585')
recording.read_recording()
fbanks = recording.frequency_banks()
dfbanks = numpy.diff(fbanks, axis=0)

# Set network parameters
tsteps = 1
batch_size = 50
epochs = 5
lahead = 1

# Generate the training input
print('Generating Data')
xLength = batch_size * int(0.75 * len(fbanks) / batch_size)
x = fbanks[:xLength, :, :]
print('Input shape:', x.shape)

# Generate the training output
#expected_output = numpy.zeros((len(x), 52))
#for i in range(len(x) - 5):
#    expected_output[i, 0:26] = numpy.mean(x[i + 1:i + 5, 0, 0:26], axis=0)
#    expected_output[i, 26:52] = numpy.var(x[i + 1:i + 5, 0, 0:26], axis=0)
#expected_output = (expected_output - numpy.mean(expected_output, axis=0)) / numpy.std(expected_output, axis=0)
expected_output = numpy.zeros((len(x), 78))
for i in range(len(x) - 8):
    expected_output[i, 0:26] = x[i + 2, 0, :] - x[i, 0, :]
    expected_output[i, 26:52] = x[i + 4, 0, :] - x[i, 0, :]
    expected_output[i, 52:78] = x[i + 8, 0, :] - x[i, 0, :]
print('Output shape: ', expected_output.shape)

# Create the recurrent neural network
print('Creating Model')
model = Sequential()
rnn1 = LSTM(
           100, batch_input_shape=(batch_size, tsteps, 26),
           return_sequences=True, stateful=True, dropout_W=0.05)
model.add(rnn1)
rnn2 = LSTM(100, return_sequences=False, stateful=True, dropout_W=0.05)
model.add(rnn2)
model.add(Dense(78))
rmsprop = RMSprop(lr=0.001)
model.compile(loss='mse', optimizer=rmsprop)

# Train the network
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

# Generate the test input and output
print('Predicting')
testOffset = len(x)
testLength = batch_size * int(0.25 * len(fbanks) / batch_size)
test_input = fbanks[testOffset:testOffset + testLength,:,:]
#test_expected = numpy.zeros((len(test_input), 52))
#for i in range(len(test_input) - 5):
#    test_expected[i, 0:26] = numpy.mean(test_input[i + 1:i + 5, 0, 0:26], axis=0)
#    test_expected[i, 26:52] = numpy.var(test_input[i + 1:i + 5, 0, 0:26], axis=0)
#test_expected = (test_expected - numpy.mean(test_expected, axis=0)) / numpy.std(test_expected, axis=0)
test_expected = numpy.zeros((len(test_input), 78))
for i in range(len(test_input) - 8):
    test_expected[i, 0:26] = test_input[i + 2, 0, :] - test_input[i, 0, :]
    test_expected[i, 26:52] = test_input[i + 4, 0, :] - test_input[i, 0, :]
    test_expected[i, 52:78] = test_input[i + 8, 0, :] - test_input[i, 0, :]

# Create an encoder model to get hidden layer activity
encoder = Sequential()
encoder.add(rnn1)
encoder.add(rnn2)

# Generate the hidden layer and predictions for the test data
test_hidden = encoder.predict(test_input, batch_size=batch_size)
test_output = model.predict(test_input, batch_size=batch_size)

# Plot the direct results for an arbitrary segment
print('Plotting Results')
start = 7500
end = 8000
pyplot.figure()
pyplot.subplot(411)
pyplot.imshow(test_input.reshape(len(test_input), 26).transpose()[:, start:end],
              aspect='auto', origin='lower', interpolation='nearest')
pyplot.subplot(412)
pyplot.imshow(test_hidden.transpose()[:, start:end],
              aspect='auto', origin='lower', interpolation='nearest')
pyplot.subplot(413)
pyplot.imshow(test_output.transpose()[:, start:end],
              aspect='auto', origin='lower', interpolation='nearest')
pyplot.subplot(414)
error = numpy.sqrt(numpy.sum(numpy.square(test_expected[:,0:26] - test_output[:,0:26]), axis=1))
pyplot.plot(error[start:end])
pyplot.show()

def filter_data_by_speaker(data, offset, index, starts, ends):
    filteredData = numpy.zeros((0, 100))
    filteredIndex = numpy.zeros((0, 3))
    for i, start, end in zip(index, starts, ends):
        a = int(start * 40) - offset
        b = int(end * 40) - offset
        if a >= 0 and b < len(data):
            filteredIndex = numpy.vstack((filteredIndex,
                            [i, len(filteredIndex), len(filteredIndex) + (b - a)]))
            filteredData = numpy.concatenate((filteredData, data[a:b, :]), axis=0)
    return filteredData, filteredIndex

def plot_vocal_kde(v):
    from scipy import stats
    v = v[numpy.random.rand(len(v)) < 0.05, :]
    x = v[:, 0]
    y = v[:, 1]
    xmin, xmax = -4, 8
    ymin, ymax = -3, 4
    # Peform the kernel density estimate
    xx, yy = numpy.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = numpy.vstack([xx.ravel(), yy.ravel()])
    values = numpy.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    f = numpy.reshape(kernel(positions).T, xx.shape)
    fig = pyplot.figure()
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # Contourf plot
    cfset = ax.contourf(xx, yy, f, cmap='Blues')
    ## Or kernel density estimate plot instead of the contourf plot
    #ax.imshow(numpy.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
    # Contour plot
    cset = ax.contour(xx, yy, f, colors='k')
    # Label plot
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('Y1')
    ax.set_ylabel('Y0')

def plot_vocal_space(v, c):
    v = v[numpy.random.rand(len(v)) < 0.05, :]
    pyplot.subplot(221)
    pyplot.scatter(v[:,0], v[:,1], c=c, marker='.', edgecolors='face')
    pyplot.ylabel('{0:.2f}% Variance'.format(100 * pca.explained_variance_ratio_[1]))
    pyplot.subplot(222)
    pyplot.scatter(v[:,2], v[:,1], c=c, marker='.', edgecolors='face')
    pyplot.xlabel('{0:.2f}% Variance'.format(100 * pca.explained_variance_ratio_[2]))
    pyplot.subplot(223)
    pyplot.scatter(v[:,0], v[:,2], c=c, marker='.', edgecolors='face')
    pyplot.xlabel('{0:.2f}% Variance'.format(100 * pca.explained_variance_ratio_[0]))


def plot_vocal_trajectory(v, a, b, i):
    pyplot.subplot(221)
    pyplot.plot(v[a:b, 0], v[a:b, 1], lw=1.5, label='Voc {}'.format(i))
    pyplot.subplot(222)
    pyplot.plot(v[a:b, 2], v[a:b, 1], lw=1.5, label='Voc {}'.format(i))
    pyplot.subplot(223)
    pyplot.plot(v[a:b, 0], v[a:b, 2], lw=1.5, label='Voc {}'.format(i))

# Filter the hidden layer activity by LENA speaker label
index, starts, ends = recording.filter_speaker('CHN')
chnData, chnIndex = filter_data_by_speaker(test_hidden, testOffset, index, starts, ends)

index, starts, ends = recording.filter_speaker('FAN')
fanData, fanIndex = filter_data_by_speaker(test_hidden, testOffset, index, starts, ends)

index, starts, ends = recording.filter_speaker('MAN')
manData, manIndex = filter_data_by_speaker(test_hidden, testOffset, index, starts, ends)

index, starts, ends = recording.filter_speaker('NON')
nonData, nonIndex = filter_data_by_speaker(test_hidden, testOffset, index, starts, ends)

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(chn)

#ica = FastICA(n_components=3)
#v = ica.fit_transform(rnns)

#tsne = TSNE(n_components=2)
#v = tsne.fit_transform(rnns)

vChn = pca.transform(chnData)
vFan = pca.transform(fanData)
vMan = pca.transform(manData)
vNon = pca.transform(nonData)

pyplot.figure()
plot_vocal_space(vChn, 'grey')
for i in [24343, 24804, 25008, 25120]:
    r = numpy.where(chnIndex[:,0] == i)[0]
    plot_vocal_trajectory(vChn, int(chnIndex[r,1]), int(chnIndex[r,2]), i)    

pyplot.figure()
plot_vocal_space(vChn, 'grey')
for i in [26126, 26179, 26376, 26626]:
    r = numpy.where(chnIndex[:,0] == i)[0]
    plot_vocal_trajectory(vChn, int(chnIndex[r,1]), int(chnIndex[r,2]), i)
