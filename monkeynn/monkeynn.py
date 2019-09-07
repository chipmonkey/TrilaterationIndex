import numpy as np
from scipy.spatial import distance

class monkeynn:
    def __init__(self, P):
        """ChipMonkey Approximate Nearest Neighbor
        Input: P - numpy.ndarray
        P describes a list of multidimensional points
        P.shape is (n, D) where:
        D is the dimension
        n is the number of records in the base population
        P[i] is an nx1 numpy array of the coordinates of P_i
        ReferencePoints is a randomly generated DxD ndarray
        """
        self.P=P
        (self.n, self.D) = P.shape

        # Array of minimum and maximum values per dimension:
        minD = np.amin(P,0)
        maxD = np.amax(P,0)
        self.DRanges = np.dstack((minD, maxD))[0]

        # Generate D random D-dimensional points:
        nReferencePoints - ceil(log(self.D))
        self.ReferencePoints = self._randomPoints(self.D,
                                                  self.DRanges,
                                                  nReferencePoints)

        self.myIndex = np.zeros(nReferencePoints, self.n)
        for i, rp in enumerate(ReferencePoints):
            for j, myP in enumerate(P):
                myIndex[i, j] = distance.euclidean(self.ReferencPoints[i],
                                                   self.P[j])
        print("Myshape: {}".format(myIndex.shape))

    def _randomPoints(self, D, DRanges, n):
        """Generate n random D-dimensional points
        Dranges is a (D,2) shaped array listing the
        min and max values for each dimension in the generated point
        """
        widths = DRanges[:,1] - DRanges[:,0]
        rpoints = DRanges[:,1] + widths * np.random.random(size=(n, D))
        return rpoints

    def whoami(self):
        print("Population is {}x{}".format(self.n, self.D))
        print(distance.euclidean(self.ReferencePoints[0],
                                 self.P[0]))

    def queries(self, Q, m):
        """Perform nearest neighbor queries
        Input: Q - numpy.ndarray
        Q describes a list of multidimensional points
        Q.shape is (n, D) where:
        D is the dimension P.D must == Q.D
        n is the number of points in Q
        Q[i] is an mx1 numpy array of the coordinates of Q_i
        this function returns R a (Qxm) np array such that R[i]
        represents the indexes in P of the top m
        nearest neighbors to Q[i] in P
        """
        return 0
