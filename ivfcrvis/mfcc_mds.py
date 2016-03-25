import numpy
import matplotlib.pyplot as pyplot
from sklearn.decomposition import FastICA
from ivfcrvis import recording as rec
from ivfcrvis.mfcc_analysis import speaker_mfccs
import scipy.spatial as spatial


def fmt(i, x, y):
    import subprocess
    command = 'start {root}/{subject_id}/CHN/{voc}.wav'.format(root='E:/ivfcr', subject_id=rec.Recording.ids[0], voc=i)
    print(command)
    subprocess.call(command, shell=True)
    return 'Voc: {i:.0f}\nx: {x:0.2f}\ny: {y:0.2f}'.format(i=i, x=x, y=y)


class FollowDotCursor(object):
    """Display the x,y location of the nearest data point.
    """
    def __init__(self, ax, index, x, y, tolerance=5, formatter=fmt, offsets=(-20, 20)):
        self._points = numpy.column_stack((x, y))
        self._index = index
        self.offsets = offsets
        self.scale = x.ptp()
        self.scale = y.ptp() / self.scale if self.scale else 1
        self.tree = spatial.cKDTree(self.scaled(self._points))
        self.formatter = formatter
        self.tolerance = tolerance
        self.ax = ax
        self.fig = ax.figure
        #self.ax.xaxis.set_label_position('top')
        self.dot = ax.scatter(
            [x.min()], [y.min()], s=130, color='green', alpha=0.7)
        self.annotation = self.setup_annotation()
        #pyplot.connect('motion_notify_event', self)
        pyplot.connect('button_press_event', self)

    def scaled(self, points):
        points = numpy.asarray(points)
        return points * (self.scale, 1)

    def __call__(self, event):
        ax = self.ax
        # event.inaxes is always the current axis. If you use twinx, ax could be
        # a different axis.
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
        elif event.inaxes is None:
            return
        else:
            inv = ax.transData.inverted()
            x, y = inv.transform([(event.x, event.y)]).ravel()
        annotation = self.annotation
        i, x, y = self.snap(x, y)
        annotation.xy = x, y
        annotation.set_text(self.formatter(i, x, y))
        self.dot.set_offsets((x, y))
        bbox = ax.viewLim
        event.canvas.draw()

    def setup_annotation(self):
        """Draw and hide the annotation box."""
        annotation = self.ax.annotate(
            '', xy=(0, 0), ha = 'right',
            xytext = self.offsets, textcoords = 'offset points', va = 'bottom',
            bbox = dict(
                boxstyle='round,pad=0.5', fc='yellow', alpha=0.75),
            arrowprops = dict(
                arrowstyle='->', connectionstyle='arc3,rad=0'))
        return annotation

    def snap(self, x, y):
        """Return the value in self.tree closest to x, y."""
        dist, idx = self.tree.query(self.scaled((x, y)), k=1, p=1)
        try:
            return self._index[idx], self._points[idx][0], self._points[idx][1]
        except IndexError:
            # IndexError: index out of bounds
            return self._points[0]


if __name__ == "__main__":
    recording = rec.Recording('E:/ivfcr', rec.Recording.ids[0])
    index, _, _ = recording.filter_speaker('CHN')
    starts, ends, mfccs = speaker_mfccs(recording, 'CHN')
    ica = FastICA(n_components=2)
    x = ica.fit_transform(mfccs)
    fig, ax = pyplot.subplots()
    comp1_label = 'Component 1'
    comp2_label = 'Component 2'
    cursor = FollowDotCursor(ax, index, x[:,0], x[:,1], formatter=fmt, tolerance=20)
    pyplot.plot(x[:100,0], x[:100,1])
    pyplot.xlabel(comp1_label)
    pyplot.ylabel(comp2_label)

    pyplot.title('Fast ICA of {0} MFCCs'.format(len(index)))
