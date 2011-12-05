from backbone import *
import math

class TestPoint:
    def setUp(self):
        self.pt1 = Point(1, 2, 3)
        self.pt2 = Point()
        self.pt3 = Point(2, 2.0, 3.0)
        self.pt4 = Point(3, 3.0, 4.0)
        
    def test_point(self):
        assert self.pt1.id == 1
        assert self.pt1.x == 2
        assert self.pt1.y == 3
        
    def test_default_constructor(self):
        assert self.pt2.id == 0
        assert self.pt2.x == 0
        assert self.pt2.y == 0
        
    def test_equal_points(self):
        assert self.pt1 == self.pt3
        
    def test_not_equal_points(self):
        assert self.pt1 != self.pt2
        
    def test_distance(self):
        assert self.pt4.distance(self.pt2) == 5.0
        assert self.pt1.distance(self.pt3) == 0.0
        
    def test_frequency(self):
        assert self.pt1.frequency(self.pt2) == 0
        assert self.pt3.frequency(self.pt4) == 0
        
class TestCluster:
    def setUp(self):
        self.centroid1 = Point(x=1.0, y=2.0)
        self.cluster1 = Cluster(self.centroid1)
        
        self.centroid2 = Point(x=5.0, y=5.0)
        self.cluster2 = Cluster(self.centroid2)
        
        self.pt1 = Point(1, 0, 0)
        self.pt2 = Point(2, 1, 1)
        self.pt3 = Point(3, 2, 2)
        self.pt4 = Point(4, 3, 3)
        self.pt5 = Point(5, 4, 4)
        self.pt6 = Point(6, 5, 5)
        
    def test_cluster(self):
        assert self.cluster1.centroid == Point(x=1, y=2)
        assert self.cluster2.centroid == Point(x=5, y=5)
    
    def test_calc_centroid(self):
        self.cluster1.add_point(self.pt1)
        self.cluster1.add_point(self.pt2)
        self.cluster1.add_point(self.pt3)
        self.cluster1.add_point(self.pt4)
        
        self.cluster1.centroid = self.cluster1.calc_centroid()
        #print self.cluster1.centroid
        assert self.cluster1.centroid == Point(x = 1.5, y = 1.5)
        
        self.cluster1.add_point(self.pt5)
        self.cluster1.centroid = self.cluster1.calc_centroid()
        assert self.cluster1.centroid == Point(x = 2, y = 2)
        
        self.cluster2.add_point(self.pt5)
        self.cluster2.add_point(self.pt6)
        self.cluster2.centroid = self.cluster2.calc_centroid()
        assert self.cluster2.centroid == Point(x = 4.5, y = 4.5)
        
    def test_update(self):
        self.cluster1.add_point(self.pt1)
        self.cluster1.add_point(self.pt2)
        self.cluster1.add_point(self.pt3)
        self.cluster1.add_point(self.pt4)
        
        self.cluster1.centroid = self.cluster1.calc_centroid()
        
        self.cluster1.add_point(self.pt5)

        assert self.cluster1.update() == math.sqrt((2.0-1.5)**2 + (2.0-1.5)**2)
        
    def test_radius(self):
        self.cluster1.add_point(self.pt1)
        self.cluster1.add_point(self.pt2)
        self.cluster1.add_point(self.pt3)
        self.cluster1.add_point(self.pt4)
        
        self.cluster1.centroid = self.cluster1.calc_centroid()
        self.cluster1.update()
        assert self.cluster1.radius == math.sqrt((1.5 ** 2) * 2) + 0.1
        
    def test_equal_clusters(self):
        assert self.cluster1 == Cluster(Point(x = 1, y = 2))
        assert self.cluster2 == Cluster(Point(x = 5.0, y = 5.0))
        
    def test_not_equal_cluster(self):
        assert self.cluster1 != self.cluster2
    
def test_create_points():
    pts = create_points(25, 100)
    assert len(pts) == 100
    
    for idx, pt in enumerate(pts):
        assert (idx + 1) == pt.id
        assert 26 > pt.x >= 0 <= pt.y < 26

def test_create_points_have_no_dups():
    pts = create_points(25, 100)
    
    for idx, pt in enumerate(pts):
        if idx < (len(pts) - 1):
            for pt2 in pts[idx+1:]:
                assert pt != pt2
    
def test_create_clusters():
    clsts = create_clusters(25, 10)
    assert len(clsts) == 10
    
    for idx, clst in enumerate(clsts):
        assert clst.label == chr(96+idx+1)
        assert clst.centroid.x >= 0 and clst.centroid.y < 26 and \
            clst.centroid.y >= 0 and clst.centroid.y < 26

def test_kmeans():
    pt1 = Point(1, 0, 0)
    pt2 = Point(2, 1, 1)
    pt3 = Point(3, 2, 2)
    pt4 = Point(4, 3, 3)
    pt5 = Point(5, 4, 4)
    pt6 = Point(6, 5, 5)
    
    centroid1 = Point(x=1.0, y=2.0)
    cluster1 = Cluster(centroid1)
    
    centroid2 = Point(x=5.0, y=5.0)
    cluster2 = Cluster(centroid2)
    
    kmeans([pt1, pt2, pt3, pt4, pt5, pt6], [cluster1, cluster2])
    assert len(cluster1.points) == 4
    assert cluster1.centroid == Point(x = 1.5, y = 1.5)
    
    assert len(cluster2.points) == 2
    assert cluster2.centroid == Point(x = 4.5, y = 4.5)

def test_backbone_network():
    pt1 = Point(1, 0, 0)
    pt2 = Point(2, 1, 1)
    pt3 = Point(3, 2, 2)
    pt4 = Point(4, 3, 3)
    pt5 = Point(5, 4, 4)
    pt6 = Point(6, 5, 5)
    
    centroid1 = Point(x=1.0, y=2.0)
    cluster1 = Cluster(centroid1)
    cluster1.label = 'a'
    
    centroid2 = Point(x=5.0, y=5.0)
    cluster2 = Cluster(centroid2)
    cluster2.label = 'b'

    kmeans([pt1, pt2, pt3, pt4, pt5, pt6], [cluster1, cluster2])
    graph = {}

    create_backbone_network(graph, [cluster1, cluster2], 5)
    assert cluster1.neighbour.keys()[0] == cluster2
    assert cluster2.neighbour.keys()[0] == cluster1

    assert graph[cluster1.label][cluster2.label] == cluster1.neighbour.values()[0]
    assert graph[cluster2.label][cluster1.label] == cluster2.neighbour.values()[0]


#def test_find_cluster():
#    clusters = create_clusters(10, 3)
#
#    assert clusters[0] == _find_cluster(clusters, 'a')
#    assert clusters[1] == _find_cluster(clusters, 'b')
#    assert clusters[2] == _find_cluster(clusters, 'c')

def test_shortest_paths():
    pass

def test_inter_cost():
    pt1 = Point(1, 0, 0)
    pt2 = Point(2, 1, 1)
    pt3 = Point(3, 2, 2)
    pt4 = Point(4, 3, 3)
    pt5 = Point(5, 4, 4)
    pt6 = Point(6, 5, 5)

    centroid1 = Point(x=1.0, y=2.0)
    cluster1 = Cluster(centroid1)
    cluster1.label = 'a'

    centroid2 = Point(x=5.0, y=5.0)
    cluster2 = Cluster(centroid2)
    cluster2.label = 'b'

    kmeans([pt1, pt2, pt3, pt4, pt5, pt6], [cluster1, cluster2])
    create_backbone_network({}, [cluster1, cluster2], [pt1, pt2, pt3, pt4, pt5, pt6])
    assert inter_cost(cluster1) == 6
    assert inter_cost(cluster2) == 0

def test_intra_cost():
    pt1 = Point(1, 0, 0)
    pt2 = Point(2, 1, 1)
    pt3 = Point(3, 2, 2)
    pt4 = Point(4, 3, 3)
    pt5 = Point(5, 4, 4)
    pt6 = Point(6, 5, 5)

    centroid1 = Point(x=1.0, y=2.0)
    cluster1 = Cluster(centroid1)
    cluster1.label = 'a'

    centroid2 = Point(x=5.0, y=5.0)
    cluster2 = Cluster(centroid2)
    cluster2.label = 'b'

    kmeans([pt1, pt2, pt3, pt4, pt5, pt6], [cluster1, cluster2])
    create_backbone_network({}, [cluster1, cluster2], [pt1, pt2, pt3, pt4, pt5, pt6])
    assert intra_cost([pt1, pt2, pt3, pt4, pt5, pt6], cluster1) == 10
    assert intra_cost([pt1, pt2, pt3, pt4, pt5, pt6], cluster2) == 10

def test_door_matt_cost():
    pt1 = Point(1, 0, 0)
    pt2 = Point(2, 1, 1)
    pt3 = Point(3, 2, 2)
    pt4 = Point(4, 3, 3)
    pt5 = Point(5, 4, 4)
    pt6 = Point(6, 5, 5)

    centroid1 = Point(x=1.0, y=2.0)
    cluster1 = Cluster(centroid1)
    cluster1.label = 'a'

    centroid2 = Point(x=5.0, y=5.0)
    cluster2 = Cluster(centroid2)
    cluster2.label = 'b'

    kmeans([pt1, pt2, pt3, pt4, pt5, pt6], [cluster1, cluster2])
    G = {}
    Cache = {}
    create_backbone_network(G, [cluster1, cluster2], [pt1, pt2, pt3, pt4, pt5, pt6])
    find_all_shortest_paths([cluster1, cluster2], Cache, G)
    assert door_matt_cost([cluster1, cluster2], cluster1, Cache) == 0
    assert door_matt_cost([cluster1, cluster2], cluster2, Cache) == 0
