import numpy as np
from sklearn.decomposition import PCA as sklearnPCA

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import ArrowStyle, FancyArrowPatch

import code


class RandomData(object):
    def __init__(self):
        mean_0 = np.array([0, 0, 0])
        covariance_0 = np.identity(3)
        class_0_samples = np.random.multivariate_normal(
            mean_0,
            covariance_0,
            20
        ).T

        mean_1 = np.array([2, 1, 1])
        covariance_1 = np.identity(3) * 0.8
        class_1_samples = np.random.multivariate_normal(
            mean_1,
            covariance_1,
            20
        ).T

        self.class_samples = [
            {'classname': 'class_0', 'samples': class_0_samples},
            {'classname': 'class_1', 'samples': class_1_samples},
        ]

        self.all_samples = np.concatenate(
            [c['samples'] for c in self.class_samples],
            axis=1
        )

        self.mean_x = np.mean(self.all_samples[0,:])
        self.mean_y = np.mean(self.all_samples[1,:])
        self.mean_z = np.mean(self.all_samples[2,:])
        self.overall_mean = np.array([
            [self.mean_x],
            [self.mean_y],
            [self.mean_z],
        ])

        scatter_matrix = np.zeros((3, 3))
        for datum_idx in range(self.all_samples.shape[1]):
            datum = self.all_samples[:,datum_idx].reshape((3, 1))
            delta = datum - self.overall_mean
            datum_contribution = delta.dot(delta.T)
            scatter_matrix += datum_contribution
        self.eig_vals, self.eig_vecs = np.linalg.eig(scatter_matrix)

        sorted_eigs = sorted(
            [
                {
                    'val': abs(self.eig_vals[eig_idx]),
                    'vec': self.eig_vecs[:,eig_idx]
                }
                for eig_idx in range(len(self.eig_vals))
            ],
            key=lambda x: x['val'],
            reverse=True
        )
        self.new_axes = np.hstack([
            eig['vec'].reshape(3, 1) for eig in sorted_eigs[:2]
        ])

        self.transformed_class_samples = [
            {
                'classname': c['classname'],
                'samples': self.new_axes.T.dot(c['samples'])
            }
            for c in self.class_samples
        ]
        self.transformed_samples = self.new_axes.T.dot(self.all_samples)

    def showOriginalData(self):

        # plot original 3D data

        fig = plt.figure(figsize=(8, 8))
        plt.rcParams['legend.fontsize'] = 10
        plt.title('Samples for class 0 and class 1')

        ax = fig.add_subplot(111, projection='3d', label='Axe')

        class_0_samples = self.class_samples[0]['samples']
        ax.plot(
            class_0_samples[0,:],
            class_0_samples[1,:],
            class_0_samples[2,:],
            'o',
            markersize=8,
            color='blue',
            alpha=0.5,
            label=self.class_samples[0]['classname']
        )

        class_1_samples = self.class_samples[1]['samples']
        ax.plot(
            class_1_samples[0,:],
            class_1_samples[1,:],
            class_1_samples[2,:],
            '^',
            markersize=8,
            color='green',
            alpha=0.5,
            label=self.class_samples[1]['classname']
        )

        ax.legend(loc='upper right')

        ax.plot(
            [self.mean_x],
            [self.mean_y],
            [self.mean_z],
            'o',
            color='red',
            alpha=0.5
        )

        for eig_vec in self.eig_vecs.T:
            arrow = Arrow3D(
                [self.mean_x, eig_vec[0]],
                [self.mean_y, eig_vec[1]],
                [self.mean_z, eig_vec[2]],
                mutation_scale=20,
                lw=3,
                arrowstyle='-|>',
                color='r'
            )
            ax.add_artist(arrow)

        plt.show()

    def showTransformedData(self):

        # plot transformed 2D data

        class_0_samples = self.transformed_class_samples[0]['samples']

        class_1_samples = self.transformed_class_samples[1]['samples']

        plt.plot(
            class_0_samples[0,:],
            class_0_samples[1,:],
            'o',
            markersize=7,
            color='blue',
            alpha=0.5,
            label=self.transformed_class_samples[0]['classname']
        )
        plt.plot(
            class_1_samples[0,:],
            class_1_samples[1,:],
            '^',
            markersize=7,
            color='green',
            alpha=0.5,
            label=self.transformed_class_samples[1]['classname']
        )
        plt.xlim([-4,4])
        plt.ylim([-4,4])
        plt.xlabel('x_values')
        plt.ylabel('y_values')
        plt.legend()
        plt.title('Transformed samples with class labels')

        plt.show()

        # compare against scikit

        sklearn_pca = sklearnPCA(n_components=2)
        sklearn_transf = sklearn_pca.fit_transform(self.all_samples.T)

        plt.plot(
            sklearn_transf[0:20,0],
            sklearn_transf[0:20,1],
            'o',
            markersize=7,
            color='blue',
            alpha=0.5,
            label=self.transformed_class_samples[0]['classname']
        )
        plt.plot(
            sklearn_transf[20:40,0],
            sklearn_transf[20:40,1],
            '^',
            markersize=7,
            color='green',
            alpha=0.5,
            label=self.transformed_class_samples[1]['classname']
        )

        plt.xlabel('x_values')
        plt.ylabel('y_values')
        plt.xlim([-4,4])
        plt.ylim([-4,4])
        plt.legend()
        plt.title(
            'Transformed samples with class labels from matplotlib.mlab.PCA()'
        )

        plt.show()



class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def main():

    rd = RandomData()

    rd.showOriginalData()

    rd.showTransformedData()



if __name__ == '__main__':
  main()
