#!/opt/python/bin/python

import timeit
import datetime
import numpy
import math
import sys
import matplotlib.pyplot as plot
from prioritydict import PriorityDictionary

# shortest path table, its structure is this:
# key: (clst1, clst2), clst1 and clst2 are the label of clusters
# the order doesn't matter, because forward and backward are the same
# value: shortest path instance
SP_TABLE = {}

# final network graph, it is created in order to calculate
# shortest path between two given points
GRAPH = {}

class Point(object):
    """
    represents a point in the NxN plane.
    """

    def __init__(self, id = 0, x = 0.0, y = 0.0):
        """
        constructor
        id is the identification of current point, if the point is used to represent centroid, id is always 0
        x indicates x coordinate
        y indicates y coordinate
        """
        self.id = id
        self.x = x
        self.y = y

    def distance(self, pt):
        """ computes euclidean distance between two points. """
        return math.sqrt((self.x - pt.x) ** 2 + (self.y - pt.y) ** 2)

    def frequency(self, pt):
        """ computes frequency between two points. """
        return math.floor(abs(self.id - pt.id) / 2)

    def __eq__(self, pt):
        """ whether two points are the same. """
        return self.x == pt.x and self.y == pt.y

    def __repr__(self):
        """ used for debugging. """
        return "Point " + str(self.id) + " (" + str(self.x) + ", " + str(self.y) + ")"

class Cluster(object):
    """ it represents a cluster/classification which has classified points. """
    def __init__(self, centroid):
        """
        constructor
        label means the identification of current cluster
        centroid is the centroid point of current cluster
        points means all the points which are belong to current cluster
        radius is used to draw plot.
        """
        self.label = ''
        self.centroid = centroid
        self.points = []
        self.radius = 0.0   # used to draw plot
        self.neighbour = {}
        self.inter_cost = 0
        self.intra_cost = 0
        self.dm_cost = 0

    def add_point(self, pt):
        """ adds new point to the cluster. """
        self.points.append(pt)

    def calc_centroid(self):
        """ calculates new centroid based on classified points. """
        sumX = 0.0
        sumY = 0.0
        dis = 0.0
        for p in self.points:
            sumX += p.x
            sumY += p.y
            d = p.distance(self.centroid)
            if dis < d: dis = d
        # radius is the longest distance within points
        self.radius = dis + 0.1
        size = len(self.points)
        if size:
            return Point(x=float(sumX)/size, y=float(sumY)/size)
        else:
            return self.centroid

    def update(self):
        """ updates new centroid and calculates diff between old and new centroid. """
        old_centroid = self.centroid
        self.centroid = self.calc_centroid()
        return old_centroid.distance(self.centroid)

    def __eq__(self, clst):
        """ whether two clusters are the same. """
        return self.centroid == clst.centroid

    def __repr__(self):
        """ used for debugging. """
        return "cluster '" + self.label + "' at " + str(self.centroid)

def create_points(N, M):
    """
    create NxN plane with M points randomly distributed.
    N: domain size of plane
    M: number of points
    """
    arr = numpy.random.randint(1, N+1, size=(M, 2))
    idx = 0
    coords = []
    points = []
    
    for ele in arr:
        if (ele[0], ele[1]) not in coords:
            idx += 1
            coords.append((ele[0], ele[1]))
    
    while idx < M:
        missed = numpy.random.randint(1, N+1, size=(M-idx, 2))
        for ele in missed:
            if (ele[0], ele[1]) not in coords:
                idx += 1
                coords.append((ele[0], ele[1]))

    # creates real points in the plane
    idx = 0
    for coord in coords:
        idx += 1
        points.append(Point(id=idx, x=coord[0], y=coord[1]))

    return points

def _cluster_name(index):
    """
    generates cluster's name based on given number,
    from a to z, if index is larger than 26,
    then use aa to az, ba to bz, etc.
    """
    if index < 26: return chr(97+index)
    else: return 'a'+chr(71+index)

def create_clusters(N, K):
    """ 
    create randomly distributed k clusters.
    N: domain size of plane
    K: number of clusters
    """
    clusters = []
    centroids = create_points(N, K)
    for idx, centroid in enumerate(centroids):
        cluster = Cluster(centroid)
        cluster.label = _cluster_name(idx)
        clusters.append(cluster)
    return clusters

def _empty_clusters(clusters):
    """ removes all the points in the cluster. """
    for clst in clusters:
        clst.points = []

def kmeans(points, clusters, threshold=1e-10):
    """
    1. put each point into nearest cluster
    2. calculate new centroid for each cluster
    3. repeat step 2 and step 3 until diff is less than threshold
    """
    diff = threshold + 1.0
    while diff > threshold:
        _empty_clusters(clusters)
        for pt in points:
            min_dis = sys.maxint
            min_cls = None
            for clst in clusters:
                dis = pt.distance(clst.centroid)
                if min_dis > dis:
                    min_dis = dis
                    min_cls = clst
            min_cls.add_point(pt)
        diff = 0.0

        for clst in clusters:
            diff += clst.update()

def create_backbone_network(G, clusters, distance):
    """ 
    creates the backbone network with given clusters.
    G: the final graph
    clusters: stable k clusters
    distance: if distance between two centroids are less or equal to distance,
              should be connected
    """
    G.clear()   # clear the graph before assigning new components
    for clst in clusters:
        neighbour = {}
        for peer in clusters:
            dis = clst.centroid.distance(peer.centroid)
            if clst != peer and dis <= distance:
                neighbour[peer.label] = dis
                clst.neighbour[peer] = dis
        G[clst.label] = neighbour

def _find_cluster(clusters, label):
    """ finds the exact cluster by given label"""
    for clst in clusters:
        if clst.label == label: return clst
    return None

def _dijkstra(G, start, end=None):
    """ Dijkstra's shortest path algorithm. """
    distances = {}
    predecessors = {}
    Q = PriorityDictionary()
    Q[start] = 0

    for pt in Q:
        distances[pt] = Q[pt]
        if pt == end: break

        for neigh in G[pt]:
            pnLength = distances[pt] + G[pt][neigh]
            if neigh in distances:
                if pnLength < distances[neigh]:
                    raise ValueError, "Dijkstra: found better path to already-final vertex"
            elif neigh not in Q or pnLength < Q[neigh]:
                Q[neigh] = pnLength
                predecessors[neigh] = pt
    return distances, predecessors

def _shortest_path(G, start, end, sp_cache):
    """
    finds a single shortest path from given start point to the given end point.
    first check whether it is in the cached shortest path table (SP_TABLE),
    if it has been calculated before, directly pick it there;
    otherwise calculate the new shortest path and store it to the cache table.
    """
    if (start, end) in SP_TABLE:
        return sp_cache[(start, end)]
    elif (end, start) in SP_TABLE:
        return sp_cache[(end, start)]
    else:
        D, P = _dijkstra(G, start, end)
        path = []
        temp = end
        while 1:
            path.append(end)
            if end == start: break
            end = P[end]
        path.reverse()
        sp_cache[(start, temp)] = path
        return path
    
def find_all_shortest_paths(clusters, sp_cache, G):
    """
    finds all shortest paths and stores them in the shortest path table.
    sp_cache means the shortest path table
    G is the graph
    """
    sp_cache.clear()    # clear cached table for new size of clusters
    for src in clusters:
        for dest in clusters:
            if src != dest:
                _shortest_path(G, src.label, dest.label, sp_cache)

def inter_cost(cluster):
    """
    calculates the inter network cost.
    each two points in the same cluster will be connected
    by the centroid of the cluster.
    cluster: given cluster
    """
    def _p2p(point):
        _freq_sum = 0
        for pt in cluster.points:
            if point != pt:
                _freq_sum += point.frequency(pt)
        return _freq_sum

    return int(sum(map(_p2p, cluster.points)))

def intra_cost(points, cluster):
    """
    calculates the intra network cost.
    each two points belong to different clusters.
    clusters: the whole network
    cluster: given cluster
    """
    def _p2p(point):
        _freq_sum = 0
        for pt in points:
            if point != pt and pt not in cluster.points:
                _freq_sum += point.frequency(pt)
        return _freq_sum
    return int(sum(map(_p2p, cluster.points)))

def _c2c_cost(sclst, eclst):
    """
    calculates the communication frequency between two clusters
    sclst: source cluster
    eclst: end cluster
    """
    def _c2c(point):
        _c_sum = 0
        for pt in eclst.points:
            _c_sum += point.frequency(pt)
        return _c_sum
    return int(sum(map(_c2c, sclst.points)))

def door_matt_cost(clusters, cluster, sp_cache):
    """
    calculates the door matt effect cost.
    cluster: given cluster which participates as a stepping stone
    in a communication of points.
    """
    # checks shortest path table, finds all path larger than 3,
    # then the cluster between will be used as door matt.
    dm_sum = 0
    for path in sp_cache.values():
        if (len(path) > 2) and (cluster.label in path[1:-1]):
            src = _find_cluster(clusters, path[0])
            dest = _find_cluster(clusters, path[-1])
            dm_sum += _c2c_cost(src, dest)
    return int(dm_sum)

def total_cost(clusters):
    """
    calculates total cost of the backbone network.
    """
    inter = 0
    intra = 0
    dm = 0
    for clst in clusters:
        # print clst.label, "has cost: ", str(clst.inter_cost), str(clst.intra_cost), str(clst.dm_cost)
        inter += clst.inter_cost
        intra += clst.intra_cost
        dm += clst.dm_cost
    total = inter + intra + dm
    #iic = inter + intra
    #print "inter " + str(inter) + " intra " + str(intra) + " dm " + str(dm) + " total " + str(total) + " iic " + str(iic)
    print str(inter) + "\t" + str(intra) + "\t" + str(dm) + "\t" + str(total) # + " in " + str(inr)
    return inter, intra, dm, total

def draw_plot(points, clusters):
    """
    draws plot with points and clusters.
    Each centroid is a triangle connected by its neighbours.
    Each cluster covers its points with a circle.
    """
    plot.figure().clear() # clean the canvas, ready for new draw
    plot.title("k-means Backbone Network")
    plot.axis([-1, 26, -1, 26])
    plot.xticks(range(-1, 26, 1))
    plot.yticks(range(-1, 26, 1))

    for pt in points:
        # draws the point id
        plot.text(pt.x+0.1, pt.y+0.1, str(pt.id))

    for clst in clusters:
        color = tuple(numpy.random.rand(1, 3)[0])
        cir = plot.Circle((clst.centroid.x, clst.centroid.y), radius=clst.radius, alpha=0.3, fc=color)
        plot.gca().add_patch(cir)
        plot.plot([clst.centroid.x], [clst.centroid.y], '^')
        plot.plot([pt.x for pt in clst.points], [pt.y for pt in clst.points], 'o')

        _draw_connections(plot, clusters, SP_TABLE)
    plot.grid(True)
    #plot.show()
    plot.savefig("./" + str(len(clusters)) + " - " + str(datetime.datetime.now()) + ".png", format="png")

def _draw_connections(plot, clusters, sp_cache):
    """
    draws the connection between two clusters.
    """
    for path in sp_cache:
        clsts = [_find_cluster(clusters, step) for step in sp_cache[path]]
        plot.plot([c.centroid.x for c in clsts], [c.centroid.y for c in clsts], '-k', linewidth=0.09)

def draw_cost_plot(hori, verts, iic=False):
    """
    draws a plot has the costs of different network.
    """
    plot.figure().clear()
    if not iic: title = "Network cost"
    else: title = "Network cost with IIC"
    plot.title(title)
    plot.xticks(range(-1, 100, 1))

    # normally only draw inter-cost, intra-cost, door matt effect and total cost.
    _draw_line(plot, hori, verts[0], 'g', 'IEC')
    _draw_line(plot, hori, verts[1], 'b', 'IAC')
    _draw_line(plot, hori, verts[2], 'r', 'DMC')
    _draw_line(plot, hori, verts[3], 'k', 'TOTAL')

    # if want to watch inter-intra cost
    if iic:
        v = map(lambda iec, iac: iec + iac, verts[0], verts[1])
        _draw_line(plot, hori, v, 'm', 'IIC')
    
    plot.grid(True)
    if iic:
        plot.savefig("./" + str(datetime.datetime.now()) + "iic.png", format='png')
    else:
        plot.savefig("./" + str(datetime.datetime.now()) + "normal.png", format='png')
    #plot.show()
    
def _draw_line(plot, hori, vert, color, text):
    """
    draws a plot line, with given x-axis and y-axis values, color and annotation text.
    """
    plot.plot(hori, vert, '-o'+color)
    plot.text(hori[-1]-3, vert[-1]+2, text, color=color)
    
def draw_door_matts(hori, verts):
    """
    draws a plot shows how distance affects door matt effect.
    """
    plot.figure().clear()
    plot.title("Door Matt Effect")
    plot.xticks(hori)
    _draw_line(plot, hori, verts, 'b', 'door matt effect')
    plot.grid(True)
    plot.savefig("./door_matt_eff.png", format='png')

def distance_dmc(distances, Ks, points):
    """
    calculates total "door matt effect" of the network,
    but the network is created by different distances.
    """
    doors = []
    for d in distances:
        dmc = []
        for k in Ks:
            print "==========================", k, "=========================="
            clusters = create_clusters(25, k)

            kmeans(points, clusters)
    #        print "Finished creating kmeans algorithm"

            create_backbone_network(GRAPH, clusters, d)
    #        print "Finished creating backbone network"

            find_all_shortest_paths(clusters, SP_TABLE, GRAPH)
    #        print "Finished finding all shortest paths"

            for clst in clusters:
                clst.inter_cost = inter_cost(clst)
                clst.intra_cost = intra_cost(points, clst)
                clst.dm_cost = door_matt_cost(clusters, clst, SP_TABLE)

            ret = total_cost(clusters)
            dmc.append(ret[2])
        doors.append(sum(dmc))
    draw_door_matts(map(lambda d: float(format(d, ".4g")), distances), doors)
    
DEBUG = False

def main():
    if DEBUG:
        Ks = [10]
    else:
        Ks = [3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 30]
        #Ks = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        #Ks = range(40, 51)
        #Ks = [10]
    
    # iec = []
    # iac = []
    # dmc = []
    # costs = []
    
    distances = map(lambda x: math.sqrt(x) * 25 / 2.0, range(2, 11))
    
    points = create_points(25, 100)
    
    iec = []
    iac = []
    dmc = []
    costs = []
    for k in Ks:
     print "==========================", k, "=========================="
     clusters = create_clusters(25, k)

     kmeans(points, clusters)
    #        print "Finished creating kmeans algorithm"

     create_backbone_network(GRAPH, clusters, math.sqrt(2)*25/2.0)
    #        print "Finished creating backbone network"

     find_all_shortest_paths(clusters, SP_TABLE, GRAPH)
    #        print "Finished finding all shortest paths"

     for clst in clusters:
         clst.inter_cost = inter_cost(clst)
         clst.intra_cost = intra_cost(points, clst)
         clst.dm_cost = door_matt_cost(clusters, clst, SP_TABLE)

     ret = total_cost(clusters)
     iec.append(ret[0])
     iac.append(ret[1])
     dmc.append(ret[2])
     costs.append(ret[3])
     #draw_plot(points, clusters)
    draw_cost_plot(Ks, [iec, iac, dmc, costs])
    draw_cost_plot(Ks, [iec, iac, dmc, costs], iic=True)
#    distance_dmc(distances, Ks, points)

if __name__ == '__main__':
    if DEBUG:
        t = timeit.Timer("main()", "from __main__ import main")
        print t.timeit(5)
    else:
        main()
